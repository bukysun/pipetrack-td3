import gym
import gym_duckietown
from gym import wrappers
import torch
from env import launch_env
from duckietown_rl.ddpg import DDPG
from src.td3 import TD3
from duckietown_rl.utils import evaluate_policy
from args import get_td3_args_test
from wrappers import NormalizeWrapper, ImgWrapper, SoftActionWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper, SteeringToWheelVelWrapper, FrameStack, \
    CropWrapper
import numpy as np
import time
import matplotlib.pyplot as plt

args = get_td3_args_test()

experiment = 2 #args.experiment
seed = args.seed
policy_name = "TD3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


file_name = "{}_{}_{}_{}".format(
    policy_name,
    experiment,
    args.exp_label,
    args.seed
)

# Launch the env with our helper function
env = launch_env(seed=111, map_name=args.map_name)

# Wrappers
env = wrappers.Monitor(env, './videos/test/' + file_name + '/', force=True, video_callable=lambda x:True)
env = CropWrapper(env)
env = ResizeWrapper(env)
#env = FrameStack(env, args.frame_stack_k)
env = NormalizeWrapper(env)
env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
env = SteeringToWheelVelWrapper(env)
#env = SoftActionWrapper(env)
#env = ActionWrapper(env)
#env = DtRewardWrapper(env) # not during testing

state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize policy
policy = TD3(state_dim, action_dim, max_action, net_type=args.net_type, args=args)

policy.load("{}_{}".format(file_name, args.model_label), directory="./pytorch_models/"+file_name)

with torch.no_grad():
    total_rews = []
    for i in range(10):
        obs = env.reset()
        #env.render()
        rewards = []
        for _ in range(150):
            action = policy.predict(np.array(obs), is_training=False)
            #action = [0.4, -1]
            #action[-1] = -1
            print(action)
            obs, rew, done, misc = env.step(action)
            print(rew)
            rewards.append(rew)
            # time.sleep(0.01)
            #env.render()
            if done:
                break
        print ("total episode reward:",np.sum(rewards))
        total_rews.append(np.sum(rewards))

plt.plot(range(len(total_rews)), total_rews)
plt.xlabel("numbers")
plt.ylabel("total reward of one episode")
plt.savefig("{}_{}.png".format(file_name, args.model_label))
