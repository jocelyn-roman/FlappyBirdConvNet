# import gym
# import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
# import torchvision.transforms as transforms
# import torch.optim as optim
from torchvision import transforms
from statistics import mean, median
from collections import Counter

from ple.games.flappybird import FlappyBird
from ple import PLE

import matplotlib.pyplot as plt


def visualize_image(x, title):
    # Based on: https://www.quora.com/How-can-l-visualize-cifar-10-data-RGB-using-python-matplotlib\
    plt.imshow(x, cmap='gray')
    plt.title("Image of " + title)
    plt.show()


game = FlappyBird()

p = PLE(game, fps=30, display_screen=True,
        reward_values=
        {
            "positive": 1.0,
            "negative": -1.0,
            "tick": 1.0,
            "loss": 0.0,
            "win": 5.0
        },
        force_fps=True)

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
        self.fc1 = torch.nn.Linear(128, 2) # fully connected

    def forward(self, x):
        # Computes the activation of the first convolution
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # Reshape data to input to the input layer of the neural net
        x = x.view(-1, 128)

        # Computes the fully connected layer (activation applied later)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)


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
            action = random.randrange(0, 10)
            observation = game.getGameState()
            p.act(0)
            # Este codigo visualiza las imagenes de las iteraciones
            screen = p.getScreenGrayscale().T
            # visualize_image(screen, "TEST")
            if action == 1:
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
