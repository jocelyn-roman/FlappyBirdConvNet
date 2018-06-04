import matplotlib.pyplot as plt

import numpy as np

import os, sys

# Imports for PLE Flappy Bird
from ple.games.flappybird import FlappyBird
from ple import PLE

# Imports for Gym Flappy Bird
import gym
import gym_ple

env = gym.make('FlappyBird-v0').unwrapped

for i in range(10):
        ob = env.reset()
        while True:
            action = env.action_space.sample()
            screen = env.render(mode='rgb_array')
            ob, reward, done, _ = env.step(action)
            if done:
                plt.figure()
                plt.imshow(screen)
                plt.title('Example extracted screen')
                plt.show()
                break

env.close()


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
    # Inverse
    raw_image[raw_image == 255] = 0
    raw_image[raw_image == 1] = 255
    # Delete floor
    raw_image = np.delete(raw_image, range(404, 512), 1)
    return raw_image


def main():
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

    for _ in range(10):
        p.reset_game()
        for _ in range(5000):
            p.act(0)
            if p.game_over():
                break

    return


if __name__ == '__main__':
    main()
