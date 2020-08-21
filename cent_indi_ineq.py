import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from gather_env import GatheringEnv
from PGagent import  IAC,Centralised_AC
from network import Centralised_Critic
from copy import deepcopy
#from logger import Logger
from torch.utils.tensorboard import SummaryWriter
# from envs.ElevatorENV import Lift
from multiAG import CenAgents,Agents
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from envtest import envSocialDilemma,envLift

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', default=False, action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

'''

##### env: gathering
CentQ = True
if CentQ:
    useCenCritc = True
n_agents = 2
state_dim = 400
action_dim = 8
env = GatheringEnv(2,"default_small2")
torch.manual_seed(args.seed)
envfolder = "gathering/"
model_name = "gathering_ineq"
file_name = "save_weight/" +envfolder+ model_name
ifload = False
save_eps = 10
ifsave_model = True
agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device,"ifload":ifload,"filename": file_name}
n_episode = 201
n_steps = 1000
line = 10
'''

'''
##### env: Lift
CentQ = True
if CentQ:
    useCenCritc = True
n_agents = 10
height = 5

env = envLift(n_agents,height)
torch.manual_seed(args.seed)
envfolder = "elevator/"
model_name = "iac_ag10h5_sparse_rw_"#"centSumR_ag5h4_"#"lift_iac"
file_name = "save_weight/" +envfolder+ model_name
ifload = False
save_eps = 20
ifsave_model = True
agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device,"ifload":ifload,"filename": file_name}
n_episode = 401
n_steps = 210#200
state_dim = 4*height+1
action_dim = 3
'''
#env.seed(args.seed)
'''
CentQ = True
if CentQ:
    useCenCritc = True
n_agents = 5

env = envSocialDilemma("cleanup",n_agents)
torch.manual_seed(args.seed)
envfolder = "cleanup/"
model_name = "clean_ineq_"#"clean_centSumR_indi_"#"lift_iac"
file_name = "save_weight/" +envfolder+ model_name
ifload = False
save_eps = 10
ifsave_model = True
agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device,"ifload":ifload,"filename": file_name}
n_episode = 201
n_steps = 1000#200
#state_dim = 4*height+1
state_dim = 675
action_dim = 9

'''

CentQ = True
if CentQ:
    useCenCritc = True
n_agents = 5

env = envSocialDilemma("harvest",n_agents)
torch.manual_seed(args.seed)
envfolder = "harvest/"
model_name = "harvest_ineq_"#"clean_centSumR_indi_"#"lift_iac"
file_name = "save_weight/" +envfolder+ model_name
ifload = False
save_eps = 10
ifsave_model = False#True
agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device,"ifload":ifload,"filename": file_name}
n_episode = 11#201
n_steps = 1000#200
state_dim = 675
action_dim = 8

def inequity(n_er,ri,I):
    alpha = 5  # when env is cleanup: alpha =0
    beta = 0   # when env is cleanup: beta = 0.05
    alphaterm = [  max(erj-n_er[I],0)  for erj in n_er ]
    betaterm =  [  max(n_er[I]-erj,0)  for erj in n_er ]
    eq_rw = ri - alpha*sum(alphaterm)/(n_agents-1)- beta*sum(betaterm)/(n_agents-1)
    return eq_rw


def add_para(id):
    agentParam["id"] = str(id)
    return agentParam

def main():
    # agent = PGagent(agentParam)
    lamd = 1
    writer = SummaryWriter('runs/'+envfolder+model_name)
    ######  law only:  
    # multiPGCen = CenAgents([Centralised_AC(action_dim,state_dim,add_para(i),useLaw=False,useCenCritc=useCenCritc,num_agent=n_agents) for i in range(n_agents)],state_dim,agentParam)  # create PGagents as well as a social agent
    multiPG = Agents([IAC(action_dim,state_dim,add_para(i),useLaw=False,useCenCritc=useCenCritc,num_agent=n_agents) for i in range(n_agents)])  # create PGagents as well as a social agent
    for i_episode in range(n_episode):
        n_state, ep_reward = env.reset_linear(), 0  # reset the env
        n_er = [0 for i in range(n_agents) ]
        for t in range(n_steps):
            
            #actions = multiPGCen.choose_actions(n_state)
            actions = multiPG.choose_actions(n_state)
            n_state_, n_reward, _, _ = env.step_linear(actions)  # interact with the env
            if args.render and i_episode%10==0 and i_episode>0:  # render or not
                env.render()
            ep_reward += sum(n_reward)  # record the total reward
            n_er = [ er*args.gamma*lamd+r for (er,r) in zip(n_er,n_reward) ]
            n_eqreward = [inequity(n_er,ri,I) for (ri,I) in zip(n_reward,range(n_agents))]
            if CentQ:
                multiPG.update_cent(n_state, n_eqreward, n_state_, actions)
            else:
                multiPG.update(n_state, n_eqreward, n_state_, actions)
            '''
            if CentQ:
                multiPG.update_cent(n_state, n_reward, n_state_, actions)
            else:
                multiPG.update(n_state, n_reward, n_state_, actions)
            '''
            ######  law only:
            # multiPGCen.update_share(n_state, n_reward, n_state_, actions)
            n_state = n_state_

        running_reward = ep_reward

        #writer.add_scalar("ep_reward", ep_reward, i_episode)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
            # logger.scalar_summary("ep_reward", ep_reward, i_episode)
        if i_episode % save_eps == 0 and i_episode > 11 and ifsave_model:
            ######  law only:
            # multiPGCen.save(file_name)
            multiPG.save(file_name)
            #pass


if __name__ == '__main__':
    main()
