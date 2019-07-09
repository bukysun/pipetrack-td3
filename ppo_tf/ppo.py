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
from mlp_policy import MlpPolicy
import copy



class PPO(object):
    def __init__(self, ob_space, ac_space, policy_type, args):
        self.gamma = args.gamma
        self.lam = args.lam
        self.adam_epsilon=args.adam_epsilon
        self.clip_param = args.clip_param
        self.entcoeff = args.entcoeff
        self.optim_stepsize = args.optim_stepsize
        
        self.ob_space = ob_space
        self.ac_space = ac_space
        
        self.policy_type = policy_type
        if self.policy_type == "coord_cnn":
            self.pi = CoordConvPolicy("pi", self.ob_space, self.ac_space, args.hidden_size, args.num_hid_layers, args.kind)
            self.oldpi = CoordConvPolicy("oldpi", self.ob_space, self.ac_space, args.hidden_size, args.num_hid_layers, args.kind)
        elif self.policy_type == "dense":
            self.pi = MlpPolicy("pi", self.ob_space, self.ac_space, args.hidden_size, args.num_hid_layers) 
            self.oldpi = MlpPolicy("oldpi", self.ob_space, self.ac_space, args.hidden_size, args.num_hid_layers) 

        self.build_graph()
        U.initialize()
        self.adam.sync()

    def build_graph(self): 
        atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
        
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
        vf_loss = tf.reduce_mean(tf.square(self.pi.vpred - ret))
        total_loss = pol_surr + pol_entpen + vf_loss
        
        self.losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
        self.loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

        var_list = self.pi.get_trainable_variables()
        
        self.lossandgrad = U.function([ac, atarg, ret, lrmult] + ob, self.losses + [U.flatgrad(total_loss, var_list)])
        self.compute_losses = U.function([ac, atarg, ret, lrmult] + ob, self.losses)

        self.adam = MpiAdam(var_list, epsilon = self.adam_epsilon)

        self.assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(self.oldpi.get_variables(), self.pi.get_variables())])

    def fix_ob2feed(self, ob):
        if ob.shape[-1] == 2:
            ret = zip(*ob.tolist())
            return ret
        else:
            return [ob]


    def train(self, seg, optim_batchsize, optim_epochs):
        cur_lrmult = 1.0
        add_vtarg_and_adv(seg, self.gamma, self.lam)
        ob, unnorm_ac, atarg, tdlamret = seg["ob"], seg["unnorm_ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        d = Dataset(dict(ob=ob, ac=unnorm_ac, atarg=atarg, vtarg=tdlamret), shuffle=not self.pi.recurrent)

        if hasattr(self.pi, "ob_rms"): self.pi.update_obs_rms(ob) # update running mean/std for policy
        self.assign_old_eq_new() # set old parameter values to new parameter values
        logger.log2("Optimizing...")
        logger.log2(fmt_row(13, self.loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                lg = self.lossandgrad(batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, *self.fix_ob2feed(batch["ob"]))
                new_losses, g = lg[:-1], lg[-1]
                self.adam.update(g, self.optim_stepsize * cur_lrmult)
                losses.append(new_losses)
            logger.log2(fmt_row(13, np.mean(losses, axis=0)))
        
        logger.log2("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = self.compute_losses(batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, *self.fix_ob2feed(batch["ob"]))
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log2(fmt_row(13, meanlosses))

        for (lossval, name) in zipsame(meanlosses, self.loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        return meanlosses


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    time.sleep(0.1)
    ob = env.reset()
    time.sleep(0.1)

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...
    ep_dists = [] # distance of ...

    # Initialize history arrays
    if isinstance(ob, tuple):
        obs = np.array([[np.array(ob[0]), np.array(ob[1])] for _ in range(horizon)])
    else:
        obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    unnorm_acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    states = []
    if pi.recurrent:
        states = [state for _ in range(horizon)]

    while True:
        prevac = ac
        if pi.recurrent:
            ac, vpred, state_out = pi.act(stochastic, ob, state)
        else:
            ac, unnorm_ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "state": states, "ep_dists": ep_dists, 
                    "unnorm_ac":unnorm_acs}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            ep_dists = []
        i = t % horizon
        if isinstance(ob, tuple):
            obs[i] = [np.array(ob[0]), np.array(ob[1])]
        else:
            obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        unnorm_acs[i] = unnorm_ac
        prevacs[i] = prevac
        if pi.recurrent:
            states[i] = state
            state = state_out

        ob, rew, new, info = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            ep_dists.append(info["distance"])
            cur_ep_ret = 0
            cur_ep_len = 0
           
            # delay for ros to reset the environment
            time.sleep(0.1)
            ob = env.reset()
            state = pi.get_initial_state()
            time.sleep(0.1)
        t += 1

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
