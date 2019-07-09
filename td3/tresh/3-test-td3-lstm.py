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
    DtRewardWrapper, ActionWrapper, ResizeWrapper, SteeringToWheelVelWrapper, FrameStack
import numpy as np
import time
import matplotlib.pyplot as plt

args = get_td3_args_test()
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")


experiment = 2 #args.experiment
seed = args.seed
policy_name = "TD3"


file_name = "{}_{}_{}_{}".format(
    policy_name,
    experiment,
    args.exp_label,
    args.seed
)

# Launch the env with our helper function
env = launch_env(seed=112, map_name=args.map_name)

# Wrappers
#env = wrappers.Monitor(env, './videos/test/'+ args.savefolder  + '/', force=True, video_callable=lambda x: True)
env = ResizeWrapper(env)
#env = FrameStack(env, args.frame_stack_k)
env = NormalizeWrapper(env)
env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
env = SteeringToWheelVelWrapper(env)
#env = SoftActionWrapper(env)
#env = ActionWrapper(env)
env = DtRewardWrapper(env) # not during testing

state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize policy
policy = TD3(state_dim, action_dim, max_action, net_type=args.net_type, args=args, device=device)
policy.load("{}_{}".format(file_name, args.model_label), directory="./pytorch_models{}/".format(args.epi)+file_name)

with torch.no_grad():
    total_rews = []
    for i in range(30):
        obs = env.reset()
        hx = np.zeros(256)
        cx = np.zeros(256)
        env.render()
        rewards = []
        for _ in range(500):
            action, hx, cx = policy.predict(np.array(obs), hx, cx, is_training=False)
            #action = [0.4, -1]
            #action[-1] = -1
            action = (action + np.random.normal(
                0,
                args.noise,
                size=env.action_space.shape[0])
            ).clip(env.action_space.low, env.action_space.high)
            print("action:{}".format(action))
            obs, rew, done, misc = env.step(action)
            print("real_reward:{}".format(rew))
            rewards.append(rew)
            # time.sleep(0.01)
            env.render()
            if done:
                break
        print ("total episode reward:",np.sum(rewards))
        total_rews.append(np.sum(rewards))
    print('average episode reward:{}'.format(np.mean(total_rews)))
plt.plot(range(len(total_rews)), total_rews)
plt.xlabel("numbers")
plt.ylabel("total reward of one episode")
plt.savefig("test_data/{}_{}_{}_{}.png".format(file_name, args.model_label, args.epi, args.noise))
