import gym

def launch_env(id=None, seed=123):
    env = None
    if id is None:
        from env.pipeline_track_env import PipelineTrackEnv
        env = PipelineTrackEnv(seed)    
    else:
        env = gym.make(id)

    return env
