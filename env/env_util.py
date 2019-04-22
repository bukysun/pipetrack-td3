import gym 
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
import cv2
from collections import deque
import os 
import time

_REGISTERED = False

class CameraRecEnv(gym.Wrapper):
    def __init__(self, env, save_path):
        """Record the frames of camera as video."""
        gym.Wrapper.__init__(self, env)
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.video_index = 0
        self.init_time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time()))
 

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        self.frame_rec = [self.env.unwrapped.camera]
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.frame_rec.append(self.env.unwrapped.camera)
        if done:    #if True, record the frames in frame_rec
            print(done)
            print(len(self.frame_rec))
            video_path = self.get_video_name(self.save_path)
            video_writer = cv2.VideoWriter(video_path, self.fourcc, 30, self.env.unwrapped.camera.shape[:-1][::-1])
            for frame in self.frame_rec:
                video_writer.write(frame)
            video_writer.release()
            self.video_index += 1
        return obs, rew, done, info

    def get_video_name(self, path_name):
        return os.path.join(path_name, "video_" + self.init_time_str + "_%d.avi"%self.video_index)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = [0,0]

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        assert isinstance(self.observation_space, tuple)
        ob_space1, _ = self.observation_space

        if self.grayscale:
            ob_space2 = spaces.Box(low=0, high=255,
                shape=(self.height, self.width, 1), dtype=np.uint8)
        else:
            ob_space2 = spaces.Box(low=0, high=255,
                shape=(self.height, self.width, 3), dtype=np.uint8)
        self.observation_space = (ob_space1, ob_space2)

    def observation(self, observation):
        assert isinstance(observation, tuple)
        state, frame = observation
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return state, frame


class RemoveDyn(gym.ObservationWrapper):
    def __init__(self, env):
        """remove the dynamics in the observation"""
        gym.ObservationWrapper.__init__(self, env)
        assert isinstance(self.observation_space, tuple)
        self.observation_space = self.observation_space[1]
    
    def observation(self, observation):
        assert isinstance(observation, tuple)
        state, frame = observation
        return frame

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
        self.frames = deque([], maxlen=k)
        assert isinstance(env.observation_space, tuple)
        shp1, shp2 = env.observation_space

        shp2 = spaces.Box(low=0, high=255, shape=(shp2.shape[:-1] + (shp2.shape[-1] * k,)), dtype=shp2.dtype)
        self.observation_space = (shp1, shp2)

    def reset(self):
        ob = self.env.reset()
        self.state = ob[0]
        for _ in range(self.k):
            self.frames.append(ob[1])
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.state = ob[0]
        self.frames.append(ob[1])
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return self.state, LazyFrames(list(self.frames))


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


def make_env(env_id, seed=0, frame_stack=False, remove_dyn = False, save_camera = False, save_path = "saved_camera", *args, **kwargs):
    global _REGISTERED
    if not _REGISTERED:
        register(id="PipelineTrack-v1", entry_point="env.pipeline_track_env:PipelineTrackEnv", kwargs=kwargs)
        register(id="SimPipelineTrack-v1", entry_point="env.sim_pipeline_tracking:SimplePipelineTrackEnv", kwargs=kwargs)
        _REGISTERED = True

    def _thunk():
        env = gym.make(env_id)
        env.seed(seed)
        if env_id == "PipelineTrack-v1":
            env = NoopResetEnv(env, noop_max=10)
            if save_camera:
                env = CameraRecEnv(env, save_path)
            if not kwargs['no_cnn']:
                env = WarpFrame(env)
                if frame_stack:
                    env = FrameStack(env, 4)
                if remove_dyn:
                    env = RemoveDyn(env)
        return env 
    return _thunk


