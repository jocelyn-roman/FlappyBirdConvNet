import matplotlib.pyplot as plt

import numpy as np
import cv2

from ple.games.flappybird import FlappyBird
from ple import PLE

# Imports for Gym Flappy Bird
import gym
import gym_ple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

for i in range(10):
        ob = env.reset()
        while True:
            action = env.action_space.sample()
            screen = env.render(mode='rgb_array')
            ob, reward, done, b = env.step(action)
            print(type(ob), type(reward), type(done), type(b))
            print(ob, reward, done, b)
            if done:
                plt.figure()
                plt.imshow(screen)
                plt.title('Example extracted screen')
                plt.show()
                break
from itertools import count

from torch.distributions import Categorical


env = gym.make('FlappyBird-v0')

def visualize_image(x, title):
    # Based on: https://www.quora.com/How-can-l-visualize-cifar-10-data-RGB-using-python-matplotlib\
    plt.imshow(x, cmap='gray')
    plt.title("Image of " + title)
    plt.show()


def pre_processing(raw_image):
    print(raw_image.shape)
    # Limit chosen by try n fail
    limit = 80
    # Moving to the extremes
    raw_image[raw_image > limit] = 255
    raw_image[raw_image <= limit] = 1
    print(raw_image.shape)
    # Inverse
    raw_image[raw_image == 255] = 0
    raw_image[raw_image == 1] = 255
    # Delete floor
    raw_image = np.delete(raw_image, range(404, 512), 1)
    print(raw_image.shape)
    visualize_image(raw_image, "TEST")

    return raw_image

def preprocessing(raw_image):
    imgData = np.asarray(raw_image)
    thresholdedData = (imgData > 200) * 1.0
    visualize_image(raw_image, "TEST")


from PIL import Image

im = Image.open('test2.png')
im = im.convert('L')
#preprocess(np.array(im))



class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(1, 512)
        self.affine2 = nn.Linear(512, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)
print("hi",np.asarray(im).shape)
x=pre_processing(np.array(im))
x = torch.from_numpy(x).float().unsqueeze(0)
print(x.dim())

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 7, kernel_size=5, stride=1)
        self.conv2 = torch.nn.Conv2d(7, 7, kernel_size=5, stride=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = torch.nn.Linear(128, 2)  # fully connected
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1,128)
        x = self.fc1(x)

        return F.softmax(x, dim=1)


policy = Net()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()
gamma = 0.99
log_interval = 10
print(policy.forward(x))
return cv2.resize(cv2.cvtColor(self.obs, cv2.COLOR_RGB2GRAY), (80, 80))

def select_action(state):
    #state = torch.from_numpy(state).float().unsqueeze(0)

    #state = pre_processing(state)
    state = pre_processing(np.array(state))
    state = torch.from_numpy(x).float().unsqueeze(0)
    probs = policy(state)

    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    print(action.item)

    return 0

def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning#512
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            #env.render()
            policy.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

#main()



