from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
from baselines import logger
import baselines.common.tf_util as U
from env.env_util import make_env
from args import get_ppo_rnd_args_train
import rospy
from env.ros_utils import launch_from_py
import os
from ppo_rnd import PPO_RND, traj_segment_generator, flatten_lists
from collections import deque
from mpi4py import MPI
import numpy as np
import time
from statistics import stats

policy_name = "ppo_rnd"

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d, %d" %(500, 70)
args = get_ppo_rnd_args_train()

#launch ros node
launch = launch_from_py("auv", "/home/uwsim/uwsim_ws/install_isolated/share/RL/launch/basic.launch")
launch.start()
rospy.loginfo("auv started!")
rospy.sleep(10)

rank = MPI.COMM_WORLD.Get_rank()
sess = U.single_threaded_session()
sess.__enter__()

task_name = "{}.{}.{}.{}".format(
    policy_name,
    args.env_id,
    args.taskname,
    args.seed
)

tensorboard_dir = osp.join(args.log_dir, task_name)
ckpt_dir = osp.join(args.checkpoint_dir, task_name)


if rank == 0:
    logger.configure()
else:
    logger.configure(format_strs=[])

workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank() if args.seed is not None else None
set_global_seeds(workerseed)

env = make_env(args.env_id, seed=args.seed, frame_stack=False, save_camera=False, remove_dyn=False, no_cnn=False)()
policy = PPO_RND(env.observation_space, env.action_space, args.policy_type, args)

seg_gen = traj_segment_generator(policy, env, args.timesteps_per_actorbatch, stochastic = True)

episodes_so_far = 0
timesteps_so_far = 0
iters_so_far = 0
lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
ext_rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
int_rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

distbuffer = deque(maxlen=100)
tstart = time.time()
writer = U.FileWriter(tensorboard_dir)
loss_stats = stats(["pol_surr", "pol_entpen", "vf_ext_loss", "vf_int_loss", "kl", "ent", "aux_loss"]
)
ep_stats = stats(["Reward_Ext", "Reward_Int", "Episode_Length", "Episode_This_Iter", "Distance"])

while timesteps_so_far < args.max_timesteps:
    # Save model
    if iters_so_far % args.save_per_iter == 0 and iters_so_far > 0 and ckpt_dir is not None:
        U.save_state(os.path.join(ckpt_dir, task_name), counter=iters_so_far)

    logger.log2("********** Iteration %i ************"%iters_so_far)

    seg = seg_gen.next()
    losses = policy.train(seg, args.optim_batchsize, args.optim_epochs)

    lrlocal = (seg["ep_lens"], seg["ep_rets_ext"], seg["ep_rets_int"], seg["ep_dists"]) # local values
    listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
    lens, rews_ext, rews_int, dists = map(flatten_lists, zip(*listoflrpairs))

    lenbuffer.extend(lens)
    ext_rewbuffer.extend(rews_ext)
    int_rewbuffer.extend(rews_int)
    #rewbuffer.extend(list(np.array(rews_ext) + np.array(rews_int)))
    distbuffer.extend(dists)
    logger.record_tabular("eplenmean", np.mean(lenbuffer))
    #logger.record_tabular("eprewmean", np.mean(rewbuffer))
    logger.record_tabular("epextrewmean", np.mean(ext_rewbuffer))
    logger.record_tabular("epintrewmean", np.mean(int_rewbuffer))
    logger.record_tabular("epthisiter", len(lens))
    logger.record_tabular("epdistmean", np.mean(distbuffer))

    episodes_so_far += len(lens)
    timesteps_so_far += sum(lens)
    iters_so_far += 1

    logger.record_tabular("EpisodesSoFar", episodes_so_far)
    logger.record_tabular("TimestepsSoFar", timesteps_so_far)
    logger.record_tabular("TimeElapsed", time.time() - tstart)
    if MPI.COMM_WORLD.Get_rank()==0:
        logger.dump_tabular()
        loss_stats.add_all_summary(writer, losses, iters_so_far)
        ep_stats.add_all_summary(writer, [np.mean(ext_rewbuffer), np.mean(int_rewbuffer), np.mean(lenbuffer), len(lens), np.mean(distbuffer)], iters_so_far)



env.close()
launch.shutdown()
