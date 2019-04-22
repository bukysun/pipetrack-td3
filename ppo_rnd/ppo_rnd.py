from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time, os, sys
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from cnn_policy import CoordConvPolicy
from rnd_cnn import RND
import copy
from mpi_util import RunningMeanStd



class PPO_RND(object):
    def __init__(self, ob_space, ac_space, policy_type, args):
        self.gamma = args.gamma
        self.lam = args.lam
        self.adam_epsilon=args.adam_epsilon
        self.clip_param = args.clip_param
        self.entcoeff = args.entcoeff
        self.optim_stepsize = args.optim_stepsize
        self.int_coeff = args.int_coeff
        self.ext_coeff = args.ext_coeff
        
        self.ob_space = ob_space
        self.ac_space = ac_space
        
        self.policy_type = policy_type
        if self.policy_type == "coord_cnn":
            self.pi = CoordConvPolicy("pi", self.ob_space, self.ac_space, args.hidden_size, args.num_hid_layers, args.kind)
            self.oldpi = CoordConvPolicy("oldpi", self.ob_space, self.ac_space, args.hidden_size, args.num_hid_layers, args.kind)

        self.int_rew = RND("rnd_int_rew", self.pi.ob, args) 
        self.rff_int = RewardForwardFilter(args.gamma)
        self.rff_rms_int = RunningMeanStd(comm=MPI.COMM_SELF, use_mpi=True)


        self.build_graph()
        U.initialize()
        self.adam.sync()

    def build_graph(self): 
        atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
        ret_ext = tf.placeholder(dtype=tf.float32, shape=[None]) # Extrinsic return
        ret_int = tf.placeholder(dtype=tf.float32, shape=[None]) # Intrinsic return
        
        lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
        clip_param = self.clip_param * lrmult # Annealed clipping parameter epsilon

        ob = self.pi.ob
        ac = self.pi.pdtype.sample_placeholder([None])

        kloldnew = self.oldpi.pd.kl(self.pi.pd)
        ent = self.pi.pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        pol_entpen = (-self.entcoeff) * meanent

        ratio = tf.exp(self.pi.pd.logp(ac) - self.oldpi.pd.logp(ac)) # pnew / pold
        surr1 = ratio * atarg # surrogate from conservative policy iteration
        surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
        pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
        vf_ext_loss = tf.reduce_mean(tf.square(self.pi.vpred_ext - ret_ext))
        vf_int_loss = tf.reduce_mean(tf.square(self.pi.vpred_int - ret_int))
        vf_loss = vf_ext_loss + vf_int_loss
        total_loss = pol_surr + pol_entpen + vf_loss + self.int_rew.aux_loss
        
        self.losses = [pol_surr, pol_entpen, vf_ext_loss, vf_int_loss, meankl, meanent, self.int_rew.aux_loss]
        self.loss_names = ["pol_surr", "pol_entpen", "vf_ext_loss", "vf_int_loss", "kl", "ent", "aux_loss"]

        var_list = self.pi.get_trainable_variables() + self.int_rew.get_trainable_variables()
        
        self.lossandgrad = U.function([ac, atarg, ret_ext, ret_int, lrmult] + ob, self.losses + [U.flatgrad(total_loss, var_list)])
        self.compute_losses = U.function([ac, atarg, ret_ext, ret_int, lrmult] + ob, self.losses)

        self.adam = MpiAdam(var_list, epsilon = self.adam_epsilon)

        self.assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(self.oldpi.get_variables(), self.pi.get_variables())])


    def train(self, seg, optim_batchsize, optim_epochs):
        #normalize the reward
        rffs_int = np.array([self.rff_int.update(rew) for rew in seg["rew_int"]])
        self.rff_rms_int.update(rffs_int.ravel())
        seg["rew_int"] = seg["rew_int"] / np.sqrt(self.rff_rms_int.var)
        
        cur_lrmult = 1.0
        add_vtarg_and_adv(seg, self.gamma, self.lam)
        ob, unnorm_ac, atarg_ext, tdlamret_ext, atarg_int, tdlamret_int = seg["ob"], seg["unnorm_ac"], seg["adv_ext"], seg["tdlamret_ext"], seg["adv_int"], seg["tdlamret_int"]
        vpredbefore_ext, vpredbefore_int = seg["vpred_ext"], seg["vpred_int"] # predicted value function before udpate
        atarg_ext = (atarg_ext - atarg_ext.mean()) / atarg_ext.std() # standardized advantage function estimate
        atarg_int = (atarg_int - atarg_int.mean()) / atarg_int.std()
        atarg = self.int_coeff * atarg_int + self.ext_coeff * atarg_ext 

        d = Dataset(dict(ob=ob, ac=unnorm_ac, atarg=atarg, vtarg_ext=tdlamret_ext, vtarg_int=tdlamret_int), shuffle=not self.pi.recurrent)

        if hasattr(self.pi, "ob_rms"): self.pi.update_obs_rms(ob) # update running mean/std for policy
        if hasattr(self.int_rew, "ob_rms"): self.int_rew.update_obs_rms(ob)   #update running mean/std for int_rew
        self.assign_old_eq_new() # set old parameter values to new parameter values
        logger.log2("Optimizing...")
        logger.log2(fmt_row(13, self.loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                lg = self.lossandgrad(batch["ac"], batch["atarg"], batch["vtarg_ext"], batch["vtarg_int"], cur_lrmult, *zip(*batch["ob"].tolist()))
                new_losses, g = lg[:-1], lg[-1]
                self.adam.update(g, self.optim_stepsize * cur_lrmult)
                losses.append(new_losses)
            logger.log2(fmt_row(13, np.mean(losses, axis=0)))
        
        logger.log2("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = self.compute_losses(batch["ac"], batch["atarg"], batch["vtarg_ext"], batch["vtarg_int"], cur_lrmult, *zip(*batch["ob"].tolist()))
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log2(fmt_row(13, meanlosses))

        for (lossval, name) in zipsame(meanlosses, self.loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_ext_before", explained_variance(vpredbefore_ext, tdlamret_ext))
        return meanlosses


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred_ext = np.append(seg["vpred_ext"], seg["nextvpred_ext"])
    vpred_int = np.append(seg["vpred_int"], seg["nextvpred_int"])
    T = len(seg["rew_ext"])
    seg["adv_ext"] = gaelam_ext = np.empty(T, 'float32')
    seg["adv_int"] = gaelam_int = np.empty(T, 'float32')
    rew_ext = seg["rew_ext"]
    rew_int = seg["rew_int"]
    lastgaelam_ext = 0
    lastgaelam_int = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta_ext = rew_ext[t] + gamma * vpred_ext[t+1] * nonterminal - vpred_ext[t]
        gaelam_ext[t] = lastgaelam_ext = delta_ext + gamma * lam * nonterminal * lastgaelam_ext
        delta_int = rew_int[t] + gamma * vpred_int[t+1] * nonterminal - vpred_int[t]
        gaelam_int[t] = lastgaelam_int = delta_int + gamma * lam * nonterminal * lastgaelam_int

    seg["tdlamret_ext"] = seg["adv_ext"] + seg["vpred_ext"]
    seg["tdlamret_int"] = seg["adv_int"] + seg["vpred_int"]

def traj_segment_generator(agent, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    time.sleep(0.1)
    ob = env.reset()
    time.sleep(0.1)

    cur_ep_ret_ext = 0 # return in current episode
    cur_ep_ret_int = 0 # intrinsic reward in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets_ext = [] # returns of completed episodes in this segment
    ep_rets_int = [] # returns of intrinsic rewards of completed episode in this segment
    ep_lens = [] # lengths of ...
    ep_dists = [] # distance of ...

    # Initialize history arrays
    if isinstance(ob, tuple):
        obs = np.array([[np.array(ob[0]), np.array(ob[1])] for _ in range(horizon)])
    else:
        obs = np.array([ob for _ in range(horizon)])
    rews_ext = np.zeros(horizon, 'float32')
    vpreds_ext = np.zeros(horizon, 'float32')
    rews_int = np.zeros(horizon, 'float32')
    vpreds_int = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    unnorm_acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    states = []
    if agent.pi.recurrent:
        states = [state for _ in range(horizon)]

    while True:
        prevac = ac
        if agent.pi.recurrent:
            ac, vpred_ext, state_out = agent.pi.act(stochastic, ob, state)
        else:
            ac, unnorm_ac, vpred_ext, vpred_int = agent.pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew_ext" : rews_ext, "rew_int": rews_int, "vpred_ext" : vpreds_ext, "vpred_int":vpreds_int, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred_ext": vpred_ext * (1 - new), "nextvpred_int": vpred_int * (1 - new),
                    "ep_rets_ext" : ep_rets_ext, "ep_rets_int":ep_rets_int, "ep_lens" : ep_lens, "state": states, "ep_dists": ep_dists, 
                    "unnorm_ac":unnorm_acs}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets_ext = []
            ep_rets_int = []
            ep_lens = []
            ep_dists = []
        i = t % horizon
        if isinstance(ob, tuple):
            obs[i] = [np.array(ob[0]), np.array(ob[1])]
        else:
            obs[i] = ob
        vpreds_ext[i] = vpred_ext
        vpreds_int[i] = vpred_int
        news[i] = new
        acs[i] = ac
        unnorm_acs[i] = unnorm_ac
        prevacs[i] = prevac
        if agent.pi.recurrent:
            states[i] = state
            state = state_out

        ob, rew_ext, new, info = env.step(ac)
        rews_ext[i] = rew_ext
        rew_int = agent.int_rew.predict(ob)
        rews_int[i] = rew_int

        cur_ep_ret_ext += rew_ext
        cur_ep_ret_int += rew_int
        cur_ep_len += 1
        if new:
            ep_rets_ext.append(cur_ep_ret_ext)
            ep_rets_int.append(cur_ep_ret_int)
            ep_lens.append(cur_ep_len)
            ep_dists.append(info["distance"])
            cur_ep_ret_ext = 0
            cur_ep_ret_int = 0
            cur_ep_len = 0
           
            # delay for ros to reset the environment
            time.sleep(0.1)
            ob = env.reset()
            state = agent.pi.get_initial_state()
            time.sleep(0.1)
        t += 1


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma
    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
