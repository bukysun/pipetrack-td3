import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
from baselines.common.mpi_running_mean_std import RunningMeanStd
import numpy as np
from CoordConv import AddCoords

class CnnPolicy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space, hid_size, num_hid_layers, kind='large'):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, hid_size, num_hid_layers, kind)
            self.scope = tf.get_variable_scope().name
            self.recurrent = False

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, kind):
        assert isinstance(ob_space, tuple)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None
        
        ob_p = U.get_placeholder(name="ob_physics", dtype=tf.float32, shape=[sequence_length] + list(ob_space[0].shape))
        ob_f= U.get_placeholder(name="ob_frames", dtype=tf.float32, shape=[sequence_length]+list(ob_space[1].shape))

        self.ob = [ob_p, ob_f]
        #process ob_p
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape = ob_space[0].shape)
        obpz = tf.clip_by_value((ob_p - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            

        #process ob_f
        x = ob_f / 255.0
            
        x = self.img_encoder(x, kind)
        
        ob_last = tf.concat((obpz, x), axis=-1)

        with tf.variable_scope("vf"):
            last_out = ob_last
            for i in range(num_hid_layers):
                last_out = tf.nn.relu(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope("pol"):
            last_out = ob_last
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            logits = tf.layers.dense(last_out, pdtype.param_shape()[0], name='logits', kernel_initializer=U.normc_initializer(0.01))
            self.pd = pdtype.pdfromflat(logits)

        self.state_in = []
        self.state_out = []
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob_p, ob_f], [ac, self.vpred])

    def img_encoder(self, x, kind):
        if kind == 'small': # from A3C paper
            x = tf.nn.relu(U.conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 256, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        elif kind == 'large': # Nature DQN
            x = tf.nn.relu(U.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 512, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        else:
            raise NotImplementedError
        return x

    
    def act(self, stochastic, ob):
        ob1, ob2 = ob
        ob2 = np.array(ob2)
        ac1, vpred1 = self._act(stochastic, ob1, ob2)
        norm_ac1 = np.tanh(ac1)
        return norm_ac1[0], ac1[0], vpred1[0]
    
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

    def update_obs_rms(self, ob):
        obp = np.array(zip(*ob.tolist())[0])
        self.ob_rms.update(obp)


class CnnPoolPolicy(CnnPolicy):
    def img_encoder(self, x, kind):
        if kind == 'small': # from A3C paper
            x = max_pool(tf.nn.relu(U.conv2d(x, 16, "l1", [8, 8], [1, 1], pad="VALID")), 4)
            x = max_pool(tf.nn.relu(U.conv2d(x, 32, "l2", [4, 4], [1, 1], pad="VALID")), 2)
            x = U.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 256, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        elif kind == 'large': # Nature DQN
            x = max_pool(tf.nn.relu(U.conv2d(x, 32, "l1", [8, 8], [1, 1], pad="VALID")), 4)
            x = max_pool(tf.nn.relu(U.conv2d(x, 64, "l2", [4, 4], [1, 1], pad="VALID")), 2)
            x = tf.nn.relu(U.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 512, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        else:
            raise NotImplementedError
        return x

class CnnSoftmaxPolicy(CnnPolicy):
    def img_encoder(self, img, kind):
        cov_layer0 = tf.nn.relu(U.conv2d(img, 32, "l1", [8, 8], [2, 2], pad="VALID"))
        cov_layer1 = tf.nn.relu(U.conv2d(cov_layer0, 64, "l2", [4, 4], [1, 1], pad="VALID"))
        cov_layer2 = tf.nn.relu(U.conv2d(cov_layer1, 64, "l3", [3, 3], [1, 1], pad="VALID"))
        # spatial weight
        _, num_rows, num_cols, num_fp = cov_layer2.get_shape()
        num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]
        x_map = np.empty([num_rows, num_cols], np.float32)
        y_map = np.empty([num_rows, num_cols], np.float32)
         
        for i in range(num_rows):
            for j in range(num_cols):
                x_map[i, j] = (i - num_rows / 2.0) / num_rows
                y_map[i, j] = (j - num_cols / 2.0) / num_cols
        x_map = tf.convert_to_tensor(x_map)
        y_map = tf.convert_to_tensor(y_map)
        x_map = tf.reshape(x_map, [num_rows * num_cols])
        y_map = tf.reshape(y_map, [num_rows * num_cols])
        
        # rearrange features to be [batch_size, num_fp, num_rows, num_cols]
        features = tf.reshape(tf.transpose(cov_layer2, [0,3,1,2]),[-1, num_rows*num_cols])
        softmax = tf.nn.softmax(features)

        fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
        fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)

        fp = tf.reshape(tf.concat([fp_x, fp_y], 1), [-1, num_fp*2])
        return fp

class CoordConvPolicy(CnnPolicy):
    def img_encoder(self, img, kind, mode="input"):
        """mode denote where add the coord conv:
            "input" means add only after input tensor
            "all" means add after all-level tensors
        """
        _, num_rows, num_cols, _ = img.get_shape().as_list()
        addcoord = AddCoords(x_dim=num_cols,
                              y_dim=num_rows,
                              with_r=False,
                              skiptile=True)
        img_coord = addcoord(img)
        x = tf.nn.relu(U.conv2d(img_coord, 32, "l1", [8, 8], [4, 4], pad="VALID"))
        x = tf.nn.relu(U.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
        x = tf.nn.relu(U.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
        
        x = U.flattenallbut0(x)
        x = tf.nn.relu(tf.layers.dense(x, 512, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        return x
        
class CnnOnlyPolicy(CoordConvPolicy):
    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, kind):
        print type(ob_space)
        assert isinstance(ob_space, gym.spaces.box.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None
        
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        self.ob = [ob]

        #process ob_
        x = ob / 255.0
            
        ob_last = self.img_encoder(x, kind)
        
        with tf.variable_scope("vf"):
            last_out = ob_last
            for i in range(num_hid_layers):
                last_out = tf.nn.relu(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope("pol"):
            last_out = ob_last
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            logits = tf.layers.dense(last_out, pdtype.param_shape()[0], name='logits', kernel_initializer=U.normc_initializer(0.01))
            self.pd = pdtype.pdfromflat(logits)

        self.state_in = []
        self.state_out = []
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode()) # XXX
        self._act = U.function([stochastic, ob], [ac, self.vpred])
    
    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob)
        return ac1[0], vpred1[0]



def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding = "SAME")



if __name__ == "__main__":
    from env.env_util import make_env
    import rospy
    rospy.init_node("sample")

    env = make_env("PipelineTrack-v1")()

    pol = CnnPolicy("pi", env.observation_space, env.action_space, hid_size=256, num_hid_layers=1)
    ob = env.reset()

    sess = U.single_threaded_session()
    sess.__enter__()
    
    U.initialize()
    
    a, v = pol.act(True, ob)
    print(a)
    print(v)






        



