import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#load and normalize dataset

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4
trainset, trainloader = None, None
testset, testloader = None, None

trainset = torchvision.datasets.CIFAR10(root = './data', transform = transform, train = True, download = True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = False, num_workers = 2)
testset = torchvision.datasets.CIFAR10(root = './data', transform = transform, train = False, download = True)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 2)


classes = ('Airplane', 'Automobie', 'Bird', 'Cat',
           'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

it = iter(trainloader)
for i in range(np.random.randint(len(trainloader)-1)):
  it.next()
images, labels = it.next()


# show images

for image in images: imshow(image)


# print labels
print("True Labels:")
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

#define CNN

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1, self.pool = None, None
        self.conv2, self.fc1 = None, None 
        self.fc2, self.fc3 = None, None
        self.conv1 = torch.nn.Conv2d(3, 6, kernel_size = 5, stride = 1)
        self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size = 5, stride = 1)
        self.fc1 = torch.nn.Linear(16*5*5, 120)
        self.fc2 = torch.nn.Linear(120,84)
        self.fc3 = torch.nn.Linear(84,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

net = Net()
print(net)

#define loss function and optimizer

import torch.optim as optim

criterion, optimizer = None, None
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(params = net.parameters(), lr = 0.001, momentum = 0.9)

# training

epochs = 2

for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()
        
        outputs, loss = None, None
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        

        
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')


PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

# testing

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:

        images, labels = data
        total += labels.size(0)
        
        outputs = None
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

