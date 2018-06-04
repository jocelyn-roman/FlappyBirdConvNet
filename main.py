import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from itertools import count

from torch.distributions import Categorical
from statistics import mean, median
from collections import Counter

from ple.games.flappybird import FlappyBird
from ple import PLE

import matplotlib.pyplot as plt

gamma = 0.99


def visualize_image(x, title):
    # Based on: https://www.quora.com/How-can-l-visualize-cifar-10-data-RGB-using-python-matplotlib\
    plt.imshow(x, cmap='gray')
    plt.title("Image of " + title)
    plt.show()


def pre_processing(raw_image):
    print(raw_image.shape)
    limit = 80
    raw_image[raw_image > limit] = 255
    raw_image[raw_image <= limit] = 1
    raw_image[raw_image == 255] = 0
    raw_image[raw_image == 1] = 255
    raw_image = np.delete(raw_image, range(404, 512), 1)
    visualize_image(raw_image.T, "TEST")
    return


from PIL import Image
im = Image.open('test2.png')
im = im.convert('L')
pre_processing(np.array(im))


game = FlappyBird()

p = PLE(game, fps=30, display_screen=True,
        reward_values=
        {
            "positive": 1.0,
            "negative": -1.0,
            "tick": 1.0,
            "loss": -5.0,
            "win": 5.0
        },
        force_fps=False)

p.init()
initial_games = 1000
goal_steps = 1000
score_requirement = 50
myKeys = [
    'next_pipe_dist_to_player',
    'next_pipe_bottom_y',
    'player_vel',
    'next_next_pipe_bottom_y',
    'next_pipe_top_y',
    'next_next_pipe_dist_to_player',
    'player_y',
    'next_next_pipe_top_y'
]


class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()

        # Input channels = inputSize, output channels = 256
        self.conv1 = torch.nn.Conv2d(input_size, 256, kernel_size=5, stride=1, padding=1)

        self.conv2 = torch.nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=1)
        self.conv2_drop = nn.Dropout2d()

        # 128 input features, 2 output
        self.fc1 = torch.nn.Linear(128, 2)  # fully connected

    def forward(self, x):
        # Computes the activation of the first convolution
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # Reshape data to input to the input layer of the neural net
        x = x.view(-1, 128)

        # Computes the fully connected layer (activation applied later)
        x = self.fc1(x)

        return F.softmax(x, dim=1)




policy = Net()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()
log_interval = 10
render = True


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


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
        state = p.reset_game()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = p.step(action)
            #state, reward, done, _ = p.act(action)
            if render:
                p.render()
            policy.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward >p.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    pass
                    # main()

def initial_population():
    training = []
    scores = []
    accepted_scores = []
    reward = 0
    for i in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            # Este codigo se debe remplazar para evaluar con la red neuronal
            action = random.randrange(0, 10)
            observation = game.getGameState()
            a = p.act(0)
            print(a)
            # Este codigo visualiza las imagenes de las iteraciones
            screen = p.getScreenGrayscale().T
            # visualize_image(screen, "TEST")
            p. saveScreen("test.png")
            if action == 1:
                # Action set: 119 or None
                reward = p.act(119)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if p.game_over():
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            output = []
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                elif data[1] == 2:
                    output = [1, 0]
                elif data[1] == 3:
                    output = [1, 0]
                elif data[1] == 4:
                    output = [1, 0]
                elif data[1] == 5:
                    output = [1, 0]
                elif data[1] == 6:
                    output = [1, 0]
                elif data[1] == 7:
                    output = [1, 0]
                elif data[1] == 8:
                    output = [1, 0]
                elif data[1] == 9:
                    output = [1, 0]
                elif data[1] == 10:
                    output = [1, 0]

                training.append([[data[0][k] for k in myKeys], output])

        p.reset_game()
        scores.append(score)

    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training


training_data = initial_population()
