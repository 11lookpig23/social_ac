import torch
import numpy as np
from torch import nn
from gym.spaces import Discrete, Box
from envs.SocialDilemmaENV.social_dilemmas.envir.cleanup import CleanupEnv
from parallel_env_process import envs_dealer
# from PGagent import IAC, Centralised_AC, Law
# from network import Centralised_Critic

class env_wrapper():
    def __init__(self,env):
        self.env = env

    def step(self,actions,need_argmax=True):
        def action_convert(action,need_argmax):
            # action = list(action.values())
            act = {}
            for i in range(len(action)):
                if need_argmax:
                    act["agent-%d"%i] = np.argmax(action[i],0)
                else:
                    act["agent-%d"%i] = action[i]
            return act
        n_state_, n_reward, done, info = self.env.step(action_convert(actions,need_argmax))
        n_state_ = np.array([state.reshape(-1) for state in n_state_.values()])
        n_reward = np.array([reward for reward in n_reward.values()])
        return n_state_/255., n_reward, done, info

    def reset(self):
        n_state = self.env.reset()
        return np.array([state.reshape(-1) for state in n_state.values()])/255.

    def seed(self,seed):
        self.env.seed(seed)

    def render(self):
        self.env.render()

    @property
    def observation_space(self):
        return Box(0., 1., shape=(675,), dtype=np.float32)

    @property
    def action_space(self):
        return Discrete(9)

    @property
    def num_agents(self):
        return self.env.num_agents

def make_parallel_env(n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = env_wrapper(CleanupEnv(num_agents=4))
            # env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env()
    return envs_dealer([get_env_fn(i) for i in range(n_rollout_threads)])


class Agents():
    def __init__(self,agents,exploration=0.5):
        self.num_agent = len(agents)
        self.agents = agents
        self.exploration = exploration
        self.epsilon = 0.95


    def choose_action(self,state,is_prob=False):
        actions = {}
        agentID = list(state.keys())
        i = 0
        if is_prob:
            for agent, s in zip(self.agents, state.values()):
                actions[agentID[i]] = agent.choose_action(s/255.,is_prob).detach()
                i += 1
            return actions
        else:
            for agent, s in zip(self.agents, state.values()):
                actions[agentID[i]] = int(agent.choose_action(s.reshape(-1)/255.).cpu().detach().numpy())
                i += 1
            return actions

    def update(self, state, reward, state_, action):
        for agent, s, r, s_,a in zip(self.agents, list(state), list(reward), list(state_), list(action)):
            agent.update(s.reshape(-1)/255.,r,s_.reshape(-1)/255.,a)

    def save(self, file_name):
        for i, ag in zip(range(self.num_agent), self.agents):
            torch.save(ag.policy, file_name + "pg" + str(i) + ".pth")

# class Social_Agents():
#     def __init__(self,agents,agentParam):
#         self.Law = social_agent(agentParam)
#         self.agents = agents
#         self.n_agents = len(agents)
#
#     def select_masked_actions(self, state):
#         actions = []
#         for i, ag in zip(range(self.n_agents), self.agents):
#             masks, prob_mask = self.Law.select_action(state[i])
#             self.Law.prob_social.append(prob_mask)  # prob_social is the list of masks for each agent
#             pron_mask_copy = prob_mask  # deepcopy(prob_mask)
#             action, prob_indi = ag.select_masked_action(state[i], pron_mask_copy)
#             self.Law.pi_step.append(prob_indi)  # pi_step is the list of unmasked policy(prob ditribution) for each agent
#             actions.append(action)
#         return actions
#
#     def update(self, state, reward, state_, action):
#         for agent, s, r, s_,a in zip(self.agents, state, reward, state_, action):
#             agent.update(s,r,s_,a)
#
#     def update_law(self):
#         self.Law.update(self.n_agents)
#
#     def push_reward(self, reward):
#         for i, ag in zip(range(self.n_agents), self.agents):
#             ag.rewards.append(reward[i])
#         self.Law.rewards.append(sum(reward))


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma, i):
    bs = [s[i] for s in bs]
    ba = [a[i] for a in ba]
    br = [r[i] for r in br]
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )