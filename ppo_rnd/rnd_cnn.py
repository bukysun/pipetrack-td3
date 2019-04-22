import numpy as np
import tensorflow as tf
from utils import fc, conv, ortho_init
from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U

def to2d(x):
    size = 1
    for shapel in x.get_shape()[1:]: size *= shapel.value
    return tf.reshape(x, (-1, size))


class RND(object):
    def __init__(self, name, ph_ob, args):
        self.convfeat = args.convfeat
        self.rep_size = args.rep_size
        self.enlargement = args.enlargement
        self.proportion_of_exp_used_for_predictor_update = args.proportion_of_exp_used_for_predictor_update
        self.scope = name

        with tf.variable_scope(self.scope):
            self.build_graph = self.build_graph(ph_ob)


    def build_graph(self, ph_ob):
        ob = ph_ob[-1]
        assert len(ob.shape.as_list()) == 4 #B, H, W, C
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape = ob.shape.as_list()[1:3] + [1])

        ob_norm = ob[:, :, :, -1:]
        ob_norm = tf.cast(ob_norm, tf.float32)
        ob_norm = tf.clip_by_value((ob_norm - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        
        # Random target network
        xr = tf.nn.leaky_relu(conv(ob_norm, "c1r", nf=self.convfeat*1, rf=8, stride=4, init_scale=np.sqrt(2)))
        xr = tf.nn.leaky_relu(conv(xr, 'c2r', nf=self.convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
        xr = tf.nn.leaky_relu(conv(xr, 'c3r', nf=self.convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
        rgbr = [to2d(xr)]
        X_r = fc(rgbr[0], 'fc1r', nh=self.rep_size, init_scale=np.sqrt(2))

        # Predictor network
        xrp = tf.nn.leaky_relu(conv(ob_norm, 'c1rp_pred', nf=self.convfeat, rf=8, stride=4, init_scale=np.sqrt(2)))
        xrp = tf.nn.leaky_relu(conv(xrp, 'c2rp_pred', nf=self.convfeat * 2, rf=4, stride=2, init_scale=np.sqrt(2)))
        xrp = tf.nn.leaky_relu(conv(xrp, 'c3rp_pred', nf=self.convfeat * 2, rf=3, stride=1, init_scale=np.sqrt(2)))
        rgbrp = to2d(xrp)

        X_r_hat = tf.nn.relu(fc(rgbrp, 'fc1r_hat1_pred', nh=256 * self.enlargement, init_scale=np.sqrt(2)))
        X_r_hat = tf.nn.relu(fc(X_r_hat, 'fc1r_hat2_pred', nh=256 * self.enlargement, init_scale=np.sqrt(2)))
        X_r_hat = fc(X_r_hat, 'fc1r_hat3_pred', nh=self.rep_size, init_scale=np.sqrt(2))

        self.feat_var = tf.reduce_mean(tf.nn.moments(X_r, axes=[0])[1])
        self.max_feat = tf.reduce_max(tf.abs(X_r))
        self.int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(X_r) - X_r_hat), axis=-1, keep_dims=True)

        targets = tf.stop_gradient(X_r)
        # self.aux_loss = tf.reduce_mean(tf.square(noisy_targets-X_r_hat))
        self.aux_loss = tf.reduce_mean(tf.square(targets - X_r_hat), -1)

        mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
        mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
        self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)
        self._predictor = U.function([ob], [self.int_rew])
        
    def predict(self, ob):
        obf = ob[-1]
        if obf.shape == 3:
            obf = np.expand_dims(obf, 0)
        int_rew = self._predictor(obf)[0]
        return int_rew

    def update_obs_rms(self, ob):
        obf = np.array(zip(*ob.tolist())[1])
        self.ob_rms.update(obf)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

