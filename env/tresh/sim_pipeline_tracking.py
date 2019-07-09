import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from auv_uwsim import AuvUwsim
import cv2

class SimplePipelineTrackEnv(gym.Env):

    def __init__(self, save_camera = False):
        self.dynamics = AuvUwsim()
        
        # set parameter for saving camera
        self.save_camera = save_camera
        if self.save_camera:
            self.fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self.video_dir = "saved_camera/video"
            self.video_index = 0

        self.observation_space = spaces.Box(low = -np.inf, high=np.inf, shape = (7,), dtype = np.float)

        # action space
        self.action_space = spaces.Box(low = -np.array(np.ones(2)), high = np.array(np.ones(2)))
        self.seed()

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return ([seed])

    def reset(self):
        self.state, self.camera, self.feat = self.dynamics.reset_sim()
        if self.save_camera:                        
            self.camera_rec = [self.camera]
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.state, self.camera, reward, done, info = self.dynamics.frame_step(action)
        self.feat = info["feat"]
        if self.save_camera:
            self.camera_rec.append(self.camera)
            if done:
                video_file = self.video_dir + "%d.avi"%self.video_index
                print("generate " + video_file)
                video_writer = cv2.VideoWriter(video_file, self.fourcc, 30, self.camera.shape[:-1][::-1])
                for frame in self.camera_rec:
                    video_writer.write(frame)
                video_writer.release()
                self.video_index += 1

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert self.camera != [], "get null img"
        x, y, psi, u, v, r = self.state
        ret_state = [np.cos(psi), np.sin(psi), u, v, r]
        if self.feat is None:
            ret_state.extend([0, 0])
        else:
            ret_state.extend(list(self.feat))
        return ret_state

