import gym
import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import datasets, transforms
from statistics import mean, median
from collections import Counter

from ple.games.flappybird import FlappyBird
game = FlappyBird()
from ple import PLE


p = PLE(game, fps=30, display_screen=True,
        reward_values=
        {
            "positive": 1.0,
            "negative": -1.0,
            "tick": 1.0,
            "loss": 0.0,
            "win": 5.0
        },
            force_fps= True)
p.init()
reward = 0.0
counter = 0
initial_games = 1000
goal_steps = 1000
score_requirement = 50
LR = 1e-5
myKeys = ['next_pipe_dist_to_player','next_pipe_bottom_y','player_vel','next_next_pipe_bottom_y','next_pipe_top_y','next_next_pipe_dist_to_player','player_y','next_next_pipe_top_y']


class Net(nn.Module):
    def __init__(self, inputSize):
        super(Net, self).__init__()

        # Input channels = inputSize, output channels = 256
        self.conv1 = torch.nn.Conv2d(inputSize, 256, kernel_size=5, stride=1, padding=1)

        self.conv2 = torch.nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=1)
        self.conv2_drop = nn.Dropout2d()

        # 128 input features, 2 output
        self.fc1 = torch.nn.Linear(128, 2) # fully conected

    def forward(self, x):
        # Computes the activation of the first convolution
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # Reshape data to input to the input layer of the neural net
        x = x.view(-1, 128)

        # Computes the fully connected layer (activation applied later)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=transform)
n_training_samples = 20000
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))


def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=2)
    return(train_loader)

def trainNet(net, batch_size, n_epochs, learning_rate):

    # Get training data
    train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)


    # Loop for n_epochs
    for epoch in range(n_epochs):

        running_loss = 0.0
        print_every = n_batches // 10
        total_train_loss = 0

        for i, data in enumerate(train_loader, 0):

            # Get inputs
            inputs, labels = data


            outputs = net(inputs)
            outputs.forward()
            outputs.backward()

            # Print statistics
            running_loss += outputs.data[0]
            total_train_loss += outputs.data[0]

            # Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} ".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every))
                # Reset running loss and time
                running_loss = 0.0





def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    reward = 0
    for i in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0,10)
            observation = game.getGameState()
            p.act(0)
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
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
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

                training_data.append([[data[0][k] for k in myKeys], output])

        p.reset_game()
        scores.append(score)



    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data




def train_model(training_data, net):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]),1)
    Y = [i[1] for i in training_data]



    net.fit({'input':X}, {'targets':Y}, n_epoch=2, snapshot_step=500, show_metric=True, run_id='openaistuff')

    return model


training_data = initial_population()
model = train_model(training_data)


scores = []
choices = []

for each_game in range(1000):
    score = 0
    game_memory = []
    prev_obs = []
    p.reset_game()
    for _ in range(goal_steps):
        p.act(0)
        if len(prev_obs) == 0:
            action = random.randrange(0, 10)
        else:
            #action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
            print("Action is ", action)
        if action == 1:
            reward = p.act(119)

        choices.append(action)
        new_observation = np.asarray([game.getGameState()[k] for k in myKeys])
        p.act(0)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if p.game_over():
            break

    scores.append(score)

print('Average Score:', sum(scores) / len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
print(score_requirement)



   #action = agent.pickAction(reward, observation)
   #reward = p.act(action)

""" "

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


if __name__ == '__main__':
    main()"""

