"""Initial self-project CNN."""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  # for showing images
import numpy as np  # for showing images

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

"Load and normalize CIFAR10."

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 4

# pulls in data from CIFAR10 via torchvision and creates a new folder called
# 'data' for it
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

# I believe this pulls the training images into the current session or somehow
# makes them accessible for model use
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

# Does the same but for the testing data
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Now look at some of the images
# commented out because I don't think it works well in a
# non-jupyter-type environment


def imshow(img):
    """Shows image."""
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show


# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# imshow(torchvision.utils.make_grid(images))
# print(" ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))

"Define a CNN."


class Net(nn.Module):
    "Make the Network an object."

    def __init__(self):
        "Initialize."
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        "Go through a cycle."
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

"Define a loss function and optimizer."

# reminders: the learning rate is multiplied by the weight gradient to create
# the weight increment for a single movement in gradient descent... the
# momentum is a factor that multiplies by the weight increment of the previous
# iteration, and then adds to (lr * weight gradient)... so it basically varies
# the learning rate dependent on how much the previous iteration changed
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

"""
Train the network... basically loop over data and give the inputs to the
network and optimize.
"""


def train():
    "Really only doing this because I am getting a weird error without it"
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward +optimizer
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

        print("Finished Training")


PATH = "./cifar_net.pth"
torch.save(net.state_dict(), PATH)


def test_print():
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))
    print("GroundTruth: ", " ".join(f"{classes[labels[j]]:5s}" for j in range(4)))


if __name__ == "__main__":
    train()
    test_print()
