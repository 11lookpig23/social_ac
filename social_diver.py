import argparse
import gym
import numpy as np
from itertools import count
from Coin_game import CoinGameVec,coin_wrapper
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
from fishery import Fishery
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
##### env: coin
CentQ = True
if CentQ:
    useCenCritc = True
n_agents = 2
env = coin_wrapper()
torch.manual_seed(args.seed)
envfolder = "coins/"
model_name = "iac_coins"
file_name = "save_weight/" +envfolder+ model_name
ifload = False
save_eps = 20
ifsave_model = False#True
agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device,"ifload":ifload,"filename": file_name}
n_episode = 101
n_steps = 20
state_dim = 17
action_dim = 4
#0., 0., 0., 1., 1.,   0., 0., 0., 0., 1.,   0., 0., 0., 0., 0., 0.
'''
##### env: Lift
'''
CentQ = True
if CentQ:
    useCenCritc = True
n_agents = 3
height = 5

env = envLift(n_agents,height)
torch.manual_seed(args.seed)
envfolder = "elevator/"
model_name = "iac_ag10h5_sparse_rw_"#"centSumR_ag5h4_"#"lift_iac"
file_name = "save_weight/" +envfolder+ model_name
ifload = False
save_eps = 20
ifsave_model = False#True
agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device,"ifload":ifload,"filename": file_name}
n_episode = 401
n_steps = 210#200
state_dim = 4*height+1
action_dim = 3
'''
#env.seed(args.seed)

CentQ = True
if CentQ:
    useCenCritc = True
n_agents = 5

env = envSocialDilemma("cleanup",n_agents)
torch.manual_seed(args.seed)
envfolder = "cleanup/"
model_name = "clean_diver_"#"clean_centSumR_indi_"#"lift_iac"
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
model_name = "harvest_indi_"#"clean_centSumR_indi_"#"lift_iac"
file_name = "save_weight/" +envfolder+ model_name
ifload = False
save_eps = 10
ifsave_model = True
agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device,"ifload":ifload,"filename": file_name}
n_episode = 201
n_steps = 1000#200
state_dim = 675
action_dim = 8

'''
'''
CentQ = True
if CentQ:
    useCenCritc = True
n_agents = 2

env = Fishery()
torch.manual_seed(args.seed)
envfolder = "fish/"
model_name = "fish_indi_"#"clean_centSumR_indi_"#"lift_iac"
file_name = "save_weight/" +envfolder+ model_name
ifload = False
save_eps = 10
ifsave_model = False
agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device,"ifload":ifload,"filename": file_name}
n_episode = 201
n_steps = 300#1000
state_dim = 25
action_dim = 4
'''
import math
mean_svo = 75*math.pi/180
std_svo = math.pi/24
SVO_theta = [np.random.normal(mean_svo,std_svo,1)[0] for i in range(n_agents)]#[15*math.pi/180,30*math.pi/180,45*math.pi/180,60*math.pi/180,75*math.pi/180]
w_clean = 0.1

def inequity(n_er,ri,I):
    alpha = 0
    beta = 0.5#0.05
    alphaterm = [  max(erj-n_er[I],0)  for erj in n_er ]
    betaterm =  [  max(n_er[I]-erj,0)  for erj in n_er ]
    eq_rw = ri - alpha*sum(alphaterm)/(n_agents-1)- beta*sum(betaterm)/(n_agents-1)
    return eq_rw

def rw_angle(r_n,I):
    other_rw = (sum(r_n)-r_n[I])/(n_agents-1)
    if r_n[I]==0:
        theta = math.pi/2
    else:
        theta = math.atan(other_rw/r_n[I])
    return theta

def redistribute_rw(r_n,I):
    u = r_n[I]-w_clean*abs(SVO_theta[I]-rw_angle(r_n,I))
    return u

def add_para(id):
    agentParam["id"] = str(id)
    return agentParam

def main():
    # agent = PGagent(agentParam)
    writer = SummaryWriter('runs/'+envfolder+model_name)
    ######  law only:  
    # multiPGCen = CenAgents([Centralised_AC(action_dim,state_dim,add_para(i),useLaw=False,useCenCritc=useCenCritc,num_agent=n_agents) for i in range(n_agents)],state_dim,agentParam)  # create PGagents as well as a social agent
    multiPG = Agents([IAC(action_dim,state_dim,add_para(i),useLaw=False,useCenCritc=useCenCritc,num_agent=n_agents) for i in range(n_agents)])  # create PGagents as well as a social agent
    for i_episode in range(n_episode):
        n_state, ep_reward = env.reset_linear(), 0
        #n_state, ep_reward = env.reset(), 0  # reset the env
        for t in range(n_steps):
            #actions = multiPGCen.choose_actions(n_state)
            actions = multiPG.choose_actions(n_state)
            n_state_, n_reward, _, _ = env.step_linear(actions)
            #n_state_, n_reward, _, _ = env.step(actions)  # interact with the env
            if args.render and i_episode%10==0 and i_episode>0:  # render or not
                env.render()
            ep_reward += sum(n_reward)  # record the total reward
            #n_eqreward = [inequity(n_er,ri,I) for (ri,I) in zip(n_reward,range(n_agents))]
            n_diver_rw = [redistribute_rw(n_reward,i) for i in range(n_agents) ]
            if CentQ:
                multiPG.update_cent(n_state, n_diver_rw, n_state_, actions)
            else:
                multiPG.update(n_state, n_diver_rw, n_state_, actions)
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

        writer.add_scalar("ep_reward", ep_reward, i_episode)
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

