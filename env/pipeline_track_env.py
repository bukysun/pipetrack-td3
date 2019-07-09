import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from auv_uwsim import AuvUwsim
import cv2

class PipelineTrackEnv(gym.Env):
    "The pipeline tracking env with localization for reward."
    def __init__(self, seed=None):
        self.dynamics = AuvUwsim() 
        self.observation_space = spaces.Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8) 

        # action space
        self.action_space = spaces.Box(low = np.zeros(2), high = np.ones(2))
        self.seed(seed) 

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return ([seed])

    def reset(self):
        self.state, self.camera, self.info = self.dynamics.reset_sim()
        return self._get_obs()

    def step(self, action):
        #print("action:", action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.state, self.camera, reward, done, info = self.dynamics.frame_step(action)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert self.camera != [], "get null img"
        x, y, psi, u, v, r = self.state[:6]
        ret_state = [np.cos(psi), np.sin(psi), u, v, r]
        return self.camera 

if __name__ == "__main__":
    from ros_utils import launch_from_py
    import rospy
    import time
    launch = launch_from_py("auv", "/home/uwsim/uwsim_ws/install_isolated/share/RL/launch/turns.launch")
    launch.start()
    rospy.loginfo("auv started!")
    rospy.sleep(10)
    
    env = PipelineTrackEnv()
    for i in range(10):
        time.sleep(0.1)
        s = env.reset()
        time.sleep(0.1)
        done = False
        while not done:
            a = [0.5, 0.5]
            s, reward, done, _ = env.step(a)
            print s.shape



