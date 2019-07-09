import math
import functools
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorDense(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorDense, self).__init__()

        state_dim = functools.reduce(operator.mul, state_dim, 1)

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)

        self.max_action = max_action

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * self.tanh(self.l3(x))
        return x

class ActorCNN(nn.Module):
    def __init__(self, action_dim, max_action, args):
        super(ActorCNN, self).__init__()
        self.args = args
        self.max_action = max_action
        # ONLY TRU IN CASE OF DUCKIETOWN:
        flat_size = 32 * 10 * 15

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 32, 7, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1)
        
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)


        #self.dropout = nn.Dropout(.5)
        self.lin1 = nn.Linear(flat_size, args.fc_hid_size)
        self.lin2 = nn.Linear(args.fc_hid_size, action_dim)


        if args.spe_init:
            print("use special initializer for actor")
            self.apply(weights_init)
            self.lin2.weight.data = normalized_columns_initializer(
                self.lin2.weight.data, 0.01)
            self.lin2.bias.data.fill_(0)
        

    def forward(self, x):
        if self.args.add_bn:
            x = self.bn1(self.lr(self.conv1(x)))
            x = self.bn2(self.pool2(self.lr(self.conv2(x))))
            x = self.bn3(self.pool3(self.lr(self.conv3(x))))
            x = self.bn4(self.lr(self.conv4(x)))
        else:
            x = self.lr(self.conv1(x))
            x = self.pool2(self.lr(self.conv2(x)))
            x = self.pool3(self.lr(self.conv3(x)))
            x = self.lr(self.conv4(x))

        x = x.view(x.size(0), -1)  # flatten
        #x = self.dropout(x)
        x = self.lr(self.lin1(x))
        
        #x = self.bn5(x)


        # this is the vanilla implementation
        # but we're using a slightly different one
        # x = self.max_action * self.tanh(self.lin2(x))

        # because we don't want our duckie to go backwards
        x = self.lin2(x)
        #x = self.tanh(x)
        x = self.max_action * self.sigm(x)  # because we don't want the duckie to go backwards

        return x


class CriticDense(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticDense, self).__init__()

        state_dim = functools.reduce(operator.mul, state_dim, 1)

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64 + action_dim, 64)
        self.l3 = nn.Linear(64, 1)

        self.l4 = nn.Linear(state_dim, 64)
        self.l5 = nn.Linear(64 + action_dim, 64)
        self.l6 = nn.Linear(64, 1)

    def forward(self, x, u):
        x1 = F.relu(self.l1(x))
        x1 = F.relu(self.l2(torch.cat([x1, u], 1)))
        x1 = self.l3(x1)

        x2 = F.relu(self.l1(x))
        x2 = F.relu(self.l2(torch.cat([x2, u], 1)))
        x2 = self.l3(x2)

        return x1, x2

    def Q1(self, x, u):
        x1 = F.relu(self.l1(x))
        x1 = F.relu(self.l2(torch.cat([x1, u], 1)))
        x1 = self.l3(x1)

        return x1


class CriticCNN(nn.Module):
    def __init__(self, action_dim, args):
        super(CriticCNN, self).__init__()
        self.args = args


        flat_size = 32 * 10 * 15

        self.lr = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(3, 32, 7, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1)
        self.conv5 = nn.Conv2d(3, 32, 7, stride=2)
        self.conv6 = nn.Conv2d(32, 32, 5, stride=1)
        self.conv7 = nn.Conv2d(32, 32, 3, stride=1)
        self.conv8 = nn.Conv2d(32, 32, 3, stride=1)

        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool6 = nn.MaxPool2d(2)
        self.pool7 = nn.MaxPool2d(2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)
        self.bn8 = nn.BatchNorm2d(32)

        self.lin1 = nn.Linear(flat_size, 256)
        self.lin2 = nn.Linear(256 + action_dim, 128)
        self.lin3 = nn.Linear(128, 1)
        self.lin4 = nn.Linear(flat_size, 256)
        self.lin5 = nn.Linear(256 + action_dim, 128)
        self.lin6 = nn.Linear(128, 1)

        if args.spe_init:
            print("special initializer for critic.")
            self.apply(weights_init)
            self.lin3.weight.data = normalized_columns_initializer(
                self.lin3.weight.data, 1.0)
            self.lin3.bias.data.fill_(0)
            
            self.lin6.weight.data = normalized_columns_initializer(
                self.lin6.weight.data, 1.0)
            self.lin6.bias.data.fill_(0)

        
    def forward(self, states, actions):
        #if self.args.add_bn:
        #    x1 = self.bn1(self.lr(self.conv1(states)))
        #    x1 = self.bn2(self.pool2(self.lr(self.conv2(x1))))
        #    x1 = self.bn3(self.pool3(self.lr(self.conv3(x1))))
        #    x1 = self.bn4(self.lr(self.conv4(x1)))
        #else:
        #    x1 = self.lr(self.conv1(states))
        #    x1 = self.pool2(self.lr(self.conv2(x1)))
        #    x1 = self.pool3(self.lr(self.conv3(x1)))
        #    x1 = self.lr(self.conv4(x1))

        #x1 = x1.view(x1.size(0), -1)  # flatten
        #x1 = self.lr(self.lin1(x1))
        #x1 = self.lr(self.lin2(torch.cat([x1, actions], 1)))  # c
        #x1 = self.lin3(x1)
        x1 = self.Q1(states, actions)

        if self.args.add_bn:
            x2 = self.bn5(self.lr(self.conv5(states)))
            x2 = self.bn6(self.pool6(self.lr(self.conv6(x2))))
            x2 = self.bn7(self.pool7(self.lr(self.conv7(x2))))
            x2 = self.bn8(self.lr(self.conv8(x2)))
        else:
            x2 = self.lr(self.conv5(states))
            x2 = self.pool6(self.lr(self.conv6(x2)))
            x2 = self.pool7(self.lr(self.conv7(x2)))
            x2 = self.lr(self.conv8(x2))

        x2 = x2.view(x2.size(0), -1)  # flatten
        x2 = self.lr(self.lin4(x2))
        x2 = self.lr(self.lin5(torch.cat([x2, actions], 1)))  # c
        x2 = self.lin6(x2)

        return x1, x2

    def Q1(self, states, actions):
        if self.args.add_bn:
            x1 = self.bn1(self.lr(self.conv1(states)))
            x1 = self.bn2(self.pool2(self.lr(self.conv2(x1))))
            x1 = self.bn3(self.pool3(self.lr(self.conv3(x1))))
            x1 = self.bn4(self.lr(self.conv4(x1)))
        else: 
            x1 = self.lr(self.conv1(states))
            x1 = self.pool2(self.lr(self.conv2(x1)))
            x1 = self.pool3(self.lr(self.conv3(x1)))
            x1 = self.lr(self.conv4(x1))

        x1 = x1.view(x1.size(0), -1)  # flatten
        x1 = self.lr(self.lin1(x1))
        x1 = self.lr(self.lin2(torch.cat([x1, actions], 1)))  # c
        x1 = self.lin3(x1)

        return x1


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, net_type, args):
        super(TD3, self).__init__()
        assert net_type in ["cnn", "dense", "densenet"]

        self.state_dim = state_dim
        self.max_action = max_action
        self.args = args

        if net_type == "dense":
            self.flat = True
            self.actor = ActorDense(state_dim, action_dim, max_action).to(device)
            self.actor_target = ActorDense(state_dim, action_dim, max_action).to(device)
        elif net_type == "cnn":
            self.flat = False
            self.actor = ActorCNN(action_dim, max_action, args).to(device)
            self.actor_target = ActorCNN(action_dim, max_action, args).to(device)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        if net_type == "dense":
            self.critic = CriticDense(state_dim, action_dim).to(device)
            self.critic_target = CriticDense(state_dim, action_dim).to(device)
        elif net_type == "cnn":
            self.critic = CriticCNN(action_dim, args).to(device)
            self.critic_target = CriticCNN(action_dim, args).to(device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.cwg_norm = [] 

    def predict(self, state, is_training=False):

        # just making sure the state has the correct format, otherwise the prediction doesn't work
        assert state.shape[0] == 3

        if self.flat:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        else:
            if is_training:
                self.actor.train()
            else:
                self.actor.eval()
            state = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001,
              policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            # Sample replay buffer
            sample = replay_buffer.sample(batch_size, flat=self.flat)
            state = torch.FloatTensor(sample["state"]).to(device)
            action = torch.FloatTensor(sample["action"]).to(device)
            next_state = torch.FloatTensor(sample["next_state"]).to(device)
            done = torch.FloatTensor(1 - sample["done"]).to(device)
            reward = torch.FloatTensor(sample["reward"]).to(device)
            if self.args.priority_replay:
                weights = torch.FloatTensor(sample["weight"]).to(device)
                idxs = sample["tree_idx"]

            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(sample["action"]).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q, reduce=False) + F.mse_loss(current_Q2, target_Q, reduce=False)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            if not self.args.priority_replay:
                critic_loss.mean().backward()
            else:
                (weights * critic_loss).mean().backward()
            # clip the grad
            if self.args.grad_norm:
                #print("add grad norm")
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 50)
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state))
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                if not self.args.priority_replay:
                    actor_loss.mean().backward()
                else:
                    (weights * actor_loss).mean().backward()
                # clip the grad 
                if self.args.grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 50)
                self.actor_optimizer.step()
                
                # check gradient
                cw = list(self.actor.parameters())[-2]
                cwg = cw.grad.cpu().data.numpy().flatten()
                cwg_norm = np.linalg.norm(cwg)
                self.cwg_norm.append(cwg_norm)

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            if self.args.priority_replay:
                replay_buffer.update_priorities(idxs, critic_loss.detach().cpu().numpy())

    def save(self, filename, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.actor.state_dict(), '{}/{}_actor.pth'.format(directory, filename))
        torch.save(self.critic.state_dict(), '{}/{}_critic.pth'.format(directory, filename))
        with open("{}/check_w.pkl".format(directory), "wb") as fr:
            pickle.dump(self.cwg_norm, fr)

    def load(self, filename, directory, for_inference=False):
        self.actor.load_state_dict(torch.load('{}/{}_actor.pth'.format(directory, filename), map_location=device))
        self.critic.load_state_dict(torch.load('{}/{}_critic.pth'.format(directory, filename), map_location=device))
        if for_inference:
            self.actor.eval()
            self.critic.eval()

    def copy_policy(self, target_policy):
        self.actor.load_state_dict(target_policy.actor.state_dict())
        self.critic.load_state_dict(target_policy.critic.state_dict())


