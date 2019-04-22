import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from auv_uwsim import AuvUwsim
import cv2

class PipelineTrackEnv(gym.Env):

    def __init__(self, no_cnn = False):
        self.dynamics = AuvUwsim() 
        self.no_cnn = no_cnn
        # observation space
        if self.no_cnn:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (7,), dtype=np.float) 
        else:
            ob_space1 = spaces.Box(low=-np.inf, high=np.inf, shape = (5,), dtype=np.float)
            ob_space2 = spaces.Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8) 
            self.observation_space = (ob_space1, ob_space2)

        # action space
        self.action_space = spaces.Box(low = -np.zeros(2), high = np.ones(2))
        self.seed()
        
        # The pipeline coordinate
        p = np.array([0.56, 1.8])
        self.pipe_unit_dir = p / np.sqrt(np.sum(np.square(p)))


    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return ([seed])

    def reset(self):
        self.state, self.camera, self.feat = self.dynamics.reset_sim()
        self.init_pos = np.array(self.state[:2])
        return self._get_obs()

    def step(self, action):
        #print("action:", action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.state, self.camera, reward, done, self.feat = self.dynamics.frame_step(action)
        # infomation data
        u, rew_no_u = None, None
        if self.feat is not None:
            u, rew_no_u = self.feat[-2:]
        cur_pos = np.array(self.state[:2])
        project_dist = np.dot((cur_pos - self.init_pos), self.pipe_unit_dir) 
        info = {"u":u, "rew_no_u":rew_no_u, "distance":project_dist}
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert self.camera != [], "get null img"
        x, y, psi, u, v, r = self.state
        ret_state = [np.cos(psi), np.sin(psi), u, v, r]
        if self.no_cnn:
            if self.feat is None:
                self.feat = [0, 0]
            ret_state.extend(list(self.feat[:2]))
            return ret_state
        else:
            return ret_state, self.camera

