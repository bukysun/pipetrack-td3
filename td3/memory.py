from collections import namedtuple
import numpy as np
import torch

Transition = namedtuple('Transition', ('timestep', 'state', 'next_state', 'action', 'reward', 'terminal'))

# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
  def __init__(self, size):
    self.index = 0
    self.size = size
    self.full = False  # Used to track actual capacity
    self.sum_tree = np.zeros((2 * size - 1, ), dtype=np.float32)  # Initialise fixed size tree with all (priority) zeros
    self.data = np.array([None] * size)  # Wrap-around cyclic buffer
    self.max = 1  # Initial max value to return (1 = 1^omega)

  # Propagates value up tree given a tree index
  def _propagate(self, index, value):
    parent = (index - 1) // 2
    left, right = 2 * parent + 1, 2 * parent + 2
    self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
    if parent != 0:
      self._propagate(parent, value)

  # Updates value given a tree index
  def update(self, index, value):
    self.sum_tree[index] = value  # Set new value
    self._propagate(index, value)  # Propagate value
    self.max = max(value, self.max)

  def append(self, data, value):
    self.data[self.index] = data  # Store data in underlying data structure
    self.update(self.index + self.size - 1, value)  # Update tree
    self.index = (self.index + 1) % self.size  # Update index
    self.full = self.full or self.index == 0  # Save when capacity reached
    self.max = max(value, self.max)

  # Searches for the location of a value in sum tree
  def _retrieve(self, index, value):
    left, right = 2 * index + 1, 2 * index + 2
    if left >= len(self.sum_tree):
      return index
    elif value <= self.sum_tree[left]:
      return self._retrieve(left, value)
    else:
      return self._retrieve(right, value - self.sum_tree[left])

  # Searches for a value in sum tree and returns value, data index and tree index
  def find(self, value):
    index = self._retrieve(0, value)  # Search for index of item from root
    data_index = index - self.size + 1
    return (self.sum_tree[index], data_index, index)  # Return value, data index, tree index

  # Returns data given a data index
  def get(self, data_index):
    return self.data[data_index % self.size]

  def total(self):
    return self.sum_tree[0]


class PriReplayMemory():
    def __init__(self, args, capacity):
        self.capacity = capacity
        self.priority_weight = args.priority_weight  # Initial importance sampling weight beta, annealed to 1 over course of training
        self.priority_exponent = args.priority_exponent
        self.transitions = SegmentTree(capacity)
        self.t = 0

    def add(self, state, next_state, action, reward, terminal):
        self.transitions.append(Transition(self.t, state, next_state, action, reward, terminal), self.transitions.max)  # Store new transition with maximum priority
        self.t = 0 if terminal else self.t + 1 # Start new episodes with t = 0

    def _get_sample_from_segment(self, segment, i):  
        valid = False
        while not valid:
            sample = np.random.uniform(i * segment, (i + 1) * segment)  # Uniformly sample an element from within a segment
            prob, idx, tree_idx = self.transitions.find(sample)  # Retrieve sample from tree with un-normalised probability
            if prob != 0:
                valid = True
        
        trans = self.transitions.get(idx)

        return prob, idx, tree_idx, trans.state, trans.next_state, trans.action, trans.reward, trans.terminal

    def sample(self, batch_size, flat):    
        p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
        segment = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]  # Get batch of valid samples
        probs, idxs, tree_idxs, states, next_states, actions, returns, terminals = zip(*batch)
        states, next_states, actions = np.stack(states), np.stack(next_states), np.stack(actions)
        probs = np.array(probs, dtype=np.float32) / p_total
        terminals, returns = np.array(terminals).reshape(-1, 1), np.array(returns).reshape(-1, 1)
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        weights = weights / weights.max()
        return {"tree_idx":tree_idxs, 
                "state": states,
                "next_state": next_states,
                "action": actions,
                "reward": returns,
                "done": terminals, 
                "weight": weights}
        
    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]


    





