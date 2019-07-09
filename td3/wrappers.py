import gym
from gym import spaces
import numpy as np
from collections import deque
import cv2


class CropWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(CropWrapper, self).__init__(env)

    def observation(self, observation):
        return observation[int(observation.shape[0]//3):,:,:]
    

class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype)
        self.shape = shape

    def observation(self, observation):
        from scipy.misc import imresize
        return imresize(observation, self.shape)

class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k 
        self.gray_scale = True
        self.frames = deque([], maxlen=k)
        obsp = self.observation_space

        self.observation_space = spaces.Box(low=0, high=255, shape=(obsp.shape[:-1] + (obsp.shape[-1] * k,)), dtype=obsp.dtype)

    def reset(self):
        ob = self.env.reset()
        if self.gray_scale:
            ob = cv2.cvtColor(ob, cv2.COLOR_RGB2GRAY)
        assert len(ob.shape) == 2       #(height, width)
        for _ in range(self.k):
            self.frames.append(ob)
        ob = np.array(list(self.frames)).transpose(1, 2, 0) 
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        if self.gray_scale:
            ob = cv2.cvtColor(ob, cv2.COLOR_RGB2GRAY)
        assert len(ob.shape) == 2       #(height, width)
        self.frames.append(ob)
        ob = np.array(list(self.frames)).transpose(1, 2, 0)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.array(list(self.frames)).transpose(1, 2, 0)
	
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)

class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10
        elif reward > 0:
            reward += 10
        else:
            reward += 4

        return reward

class StableRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(StableRewardWrapper, self).__init__(env)

    def reset(self):
        self.action_old = None
        return self.env.reset()

    def step(self, action):
        action = np.array(action)
        if self.action_old is None:
            self.action_old = action
        ob, rew, done, misc = self.env.step(action)
        # add penalty for the change of action
        action_penalty = np.linalg.norm(action - self.action_old)
        new_rew = rew - 0.01 * action_penalty
        print("test for action change in rew:{}".format(action_penalty))
        return ob, new_rew, done, misc
        
        

# this is needed because at max speed the duckie can't turn anymore
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)


    def action(self, action):
        action_ = [action[0] * 0.8, action[1]]
        return action_



