import sys
sys.path.append('../')
from modules.rnn import RNN
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TwoListsDataset(torch.utils.data.Dataset):
    def __init__(self, data, data2):
        self.data = data
        self.data2 = data2
        assert len(data) == len(data2), 'Args must have same length'


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx], self.data2[idx]

def get_q_mimic_dataset():
    data = [torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        torch.tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        torch.tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        torch.tensor([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
    ]


    labels = [torch.tensor([10., -1., -1., -1., -1., -1., -1., -1., -1., -1.]),
        torch.tensor([-1., 10., -1., -1., -1., -1., -1., -1., -1., -1.]), 
        torch.tensor([-1., -1., 10., -1., -1., -1., -1., -1., -1., -1.]), 
        torch.tensor([-1., -1., -1., 10., -1., -1., -1., -1., -1., -1.]), 
        torch.tensor([-1., -1., -1., -1., 10., -1., -1., -1., -1., -1.]), 
        torch.tensor([-1., -1., -1., -1., -1., 10., -1., -1., -1., -1.]), 
        torch.tensor([-1., -1., -1., -1., -1., -1., 10., -1., -1., -1.]), 
        torch.tensor([-1., -1., -1., -1., -1., -1., -1., 10., -1., -1.]), 
        torch.tensor([-1., -1., -1., -1., -1., -1., -1., -1., 10., -1.]), 
        torch.tensor([-1., -1., -1., -1., -1., -1., -1., -1., -1., 10.])
    ]

    return TwoListsDataset(data, labels)

def get_dataset():
    data = [torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        torch.tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        torch.tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        torch.tensor([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
    ]
    labels = torch.tensor(range(10))
    
    return TwoListsDataset(data, labels) # labels are the same as data!


class RNNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.rnn = RNN(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    
    def forward(self, input):
        action_values = self.rnn(input)
        return action_values

class FCClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, num_labels)


    def forward(self, input):
        action_values = self.fc2(F.relu(self.fc1(input)))
        return action_values


def train_test(net, trainloader, testloader, num_epochs, test_every=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        total_loss = 0
        if epoch % test_every == 0:
            test(net, trainloader)

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % test_every == 0:
            print('[epoch %d, iter %5d] loss: %.9f' % (epoch + 1, i + 1, total_loss))
        total_loss=0

    print('Finished Training')

def train_test_mimic_q_learning(net, trainloader, testloader, num_epochs,
        test_every=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        total_loss = 0
        if epoch % test_every == 0:
            test_q(net, trainloader)

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % test_every == 0:
            print('[epoch %d, iter %5d] loss: %.9f' % (epoch + 1, i + 1, total_loss))
        total_loss=0

    print('Finished Training')

def test_q(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = net(inputs)

            out_max = torch.argmax(outputs, 1)
            label_max = torch.argmax(labels, 1)
            total += labels.size(0)
            correct += (out_max == label_max).sum().item()

    print('Accuracy of the network: %d %%' % (
            100 * correct / total))

def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network: %d %%' % (
            100 * correct / total))


def classification():
    dataset = get_dataset()
    input_dim = 20
    output_dim = 10
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    rnn = RNNClassifier(input_dim, output_dim)
    fc = FCClassifier(input_dim, output_dim)
    num_epochs = 1000
    print('fc results: ')
    train_test(net = fc, trainloader=trainloader, testloader=trainloader, 
            num_epochs=num_epochs, test_every=100)
    print('rnn results: ')
    train_test(net = rnn, trainloader=trainloader, testloader=trainloader, 
            num_epochs=num_epochs, test_every=100)


def q_mimic_classification():
    dataset = get_q_mimic_dataset()
    input_dim = 20
    output_dim = 10
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    rnn = RNNClassifier(input_dim, output_dim)
    fc = FCClassifier(input_dim, output_dim)
    num_epochs = 1000
    print('fc results: ')
    train_test_mimic_q_learning(net = fc, trainloader=trainloader, testloader=trainloader, 
            num_epochs=num_epochs, test_every=100)
    print('rnn results: ')
    train_test_mimic_q_learning(net = rnn, trainloader=trainloader, testloader=trainloader, 
            num_epochs=num_epochs, test_every=100)


if __name__ == '__main__':
    q_mimic_classification()
    # classification()









