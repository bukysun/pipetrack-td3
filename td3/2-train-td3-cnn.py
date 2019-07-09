import random
import time

import numpy as np
import torch
import gym
import os

import rospy
from env.ros_utils import launch_from_py

from make_env import launch_env
from args import get_td3_args_train
from src.td3 import TD3
# from src.replaybuffer import ReplayBuffer
from utils import seed, evaluate_policy, ReplayBuffer
from wrappers import *
from gym import wrappers
from memory import PriReplayMemory

# launch ros node
launch = launch_from_py("auv", "/home/uwsim/uwsim_ws/install_isolated/share/RL/launch/turns.launch")
launch.start()
rospy.loginfo("auv started!")
rospy.sleep(10)

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
policy_name = "TD3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_td3_args_train()

time_str = time.strftime("%Y-%m-%d-%H:%M", time.localtime(time.time()))
file_name = "{}_{}_{}".format(
    policy_name,
    args.exp_label,
    time_str
)

if not os.path.exists("./results"):
    os.makedirs("./results")
if args.save_models and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

#record args
with open("results/{}.arg".format(file_name), 'w') as fw:
    arg_strs = []
    arg_dict = vars(args)
    for k in arg_dict.keys():
        arg_strs.append("{}:{}".format(k, arg_dict[k]))
    fw.write("\n".join(arg_strs))

# Launch the env with our helper function
env = launch_env(seed=args.seed)

# Wrappers
#env = wrappers.Monitor(env, './videos/train/' + time_str + '/', force=True)
env = ResizeWrapper(env)
#env = FrameStack(env, k = args.frame_stack_k)
env = NormalizeWrapper(env)
env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
env = DtRewardWrapper(env)


# Set seeds
seed(args.seed)

state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])


# Initialize policy
policy = TD3(state_dim, action_dim, max_action, net_type=args.net_type, args = args)
if args.load_model:
    policy.load("TD3_2_best_start_0_best", "./pytorch_models/TD3_2_best_start_0")
    print("load suceed!")

if not args.priority_replay:
    replay_buffer = ReplayBuffer(args.replay_buffer_max_size)
else:
    replay_buffer = PriReplayMemory(args, args.replay_buffer_max_size)

# Evaluate untrained policy
#evaluations= [evaluate_policy(env, policy)]


total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
episode_reward = None
best_eval_rew = -np.float("Inf")
best_eval_index = 0
while total_timesteps < args.max_timesteps:

    if done:
        # Reset environment
        time.sleep(0.1)
        obs = env.reset()
        time.sleep(0.1)
        env.step([0,0])
        if total_timesteps != 0:
            print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                total_timesteps, episode_num, episode_timesteps, episode_reward))
            policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau,
                         args.policy_noise, args.noise_clip, args.policy_freq)

        # Evaluate episode
        if timesteps_since_eval >= args.eval_freq:
            timesteps_since_eval %= args.eval_freq
            evaluations.append(evaluate_policy(env, policy))

            if args.save_models:
                policy.save("{}_{}".format(file_name, total_timesteps//args.eval_freq), directory="./pytorch_models/{}".format(file_name))
                if evaluations[-1] > best_eval_rew:
                    best_eval_rew = evaluations[-1]
                    best_eval_index = total_timesteps//args.eval_freq
                    policy.save("{}_{}".format(file_name, "best"), directory="./pytorch_models/{}".format(file_name))
            np.savez("./results/{}.npz".format(file_name), evaluations)

        # Reset environment
        time.sleep(0.1)
        obs = env.reset()
        time.sleep(0.1)
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # Select action randomly or according to policy
    if total_timesteps < args.start_timesteps:
        action = env.action_space.sample()
    else:
        action = policy.predict(np.array(obs), is_training=True)
        print action
        if args.expl_noise != 0:
            action = (action + np.random.normal(
                0,
                args.expl_noise,
                size=env.action_space.shape[0])
            ).clip(env.action_space.low, env.action_space.high)

    # Perform action
    #print("check action:{}".format(action))
    new_obs, reward, done, _ = env.step(action)

    if episode_timesteps >= args.env_timesteps:
        done = True

    done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
    episode_reward += reward

    # Store data in replay buffer
    replay_buffer.add(obs, new_obs, action, reward, done_bool)

    obs = new_obs

    episode_timesteps += 1
    total_timesteps += 1
    timesteps_since_eval += 1

# Final evaluation
evaluations.append(evaluate_policy(env, policy))

if args.save_models:
    policy.save("{}_final".format(file_name), directory="./pytorch_models/{}".format(file_name))
    print("best evaluation index:{}".format(best_eval_index))
np.savez("./results/{}.npz".format(file_name), evaluations)

launch.shutdown()
