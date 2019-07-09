import random
import numpy as np


# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self,max_size):
        self.storage = []
        self.max_size = max_size

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, state, next_state, action, reward, done, hx, cx, hx_n, cx_n):
        state = (state * 255).astype(np.uint8)
        next_state = (next_state * 255).astype(np.uint8)
        
        if len(self.storage) < self.max_size:
            self.storage.append((state, next_state, action, reward, done, hx, cx, hx_n, cx_n))
        else:
            # Remove random element in the memory beforea adding a new one
            self.storage.pop(random.randrange(len(self.storage)))
            self.storage.append((state, next_state, action, reward, done, hx, cx, hx_n, cx_n))


    def sample(self, batch_size=100, flat=True):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, next_states, actions, rewards, dones, hxs, cxs, hx_ns, cx_ns = [], [], [], [], [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done, hx, cx, hx_n, cx_n = self.storage[i]
            state = state.astype(np.float32) / 255.0
            next_state = next_state.astype(np.float32) / 255.0

            if flat:
                states.append(np.array(state, copy=False).flatten())
                next_states.append(np.array(next_state, copy=False).flatten())
            else:
                states.append(np.array(state, copy=False))
                next_states.append(np.array(next_state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            dones.append(np.array(done, copy=False))
            hxs.append(np.array(hx, copy=False))
            cxs.append(np.array(cx, copy=False))
            hx_ns.append(np.array(hx_n, copy=False))
            cx_ns.append(np.array(cx_n, copy=False))

        # state_sample, action_sample, next_state_sample, reward_sample, done_sample
        return {
            "state": np.stack(states),
            "next_state": np.stack(next_states),
            "action": np.stack(actions),
            "reward": np.stack(rewards).reshape(-1,1),
            "done": np.stack(dones).reshape(-1,1),
            "hx": np.stack(hxs),
            "cx": np.stack(cxs),
            "hx_n": np.stack(hx_ns),
            "cx_n": np.stack(cx_ns),
        }

