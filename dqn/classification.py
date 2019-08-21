import sys
sys.path.append('../')
from modules.rnn import RNN
from pytorch_dnc_simon.dnc.dnc import DNC
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce
import numpy as np

class BabyNamesDataset(torch.utils.data.Dataset):
    def __init__(self, pad=True, validation=False):
        self.pad = pad

        boy_path = '/home/salford/boy_names_1k.txt'
        girl_path = '/home/salford/girl_names_1k.txt'

        with open(boy_path, 'r') as f:
            boy_names = f.readlines()
        self.boy_names = list(map(lambda l: l[:-1], boy_names)) # remove \n

        with open(girl_path, 'r') as f:
            girl_names = f.readlines()
        self.girl_names = list(map(lambda l: l[:-1], girl_names)) # remove \n


        self.max_len = max(map(len, self.boy_names + self.girl_names))

        if not validation:
            self.boy_names = self.boy_names[:800]
            self.girl_names = self.girl_names[:800]
        else:
            self.boy_names = self.boy_names[800:]
            self.girl_names = self.girl_names[800:]

        self.names = (list(map(lambda n: ((n, len(n)), 0), self.boy_names))
                + list(map(lambda n: ((n, len(n)), 1), self.girl_names)))


        if pad:
            def pad_name(n):
                return n + ' '*(self.max_len - len(n))

            self.names = list(map(lambda n: ((pad_name(n[0][0]), n[0][1]), n[1]), self.names))

            assert (max(map(lambda name: len(name[0][0]), self.names)) 
                    == min(map(lambda name: len(name[0][0]), self.names)))


        
        def tensorize(name):
            eye = torch.eye(27)
            labels = [' abcdefghijklmnopqrstuvwxyz'.index(char) for char in name]
            one_hot = torch.tensor(eye[labels]).cuda()
            return one_hot


        self.names = list(map(lambda n: ((tensorize(n[0][0]), n[0][1]), n[1]), self.names))


    def __len__(self):
        return len(self.names)


    def __getitem__(self, idx):
        return self.names[idx]
       

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


class DNCClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = 256
        self.dnc = DNC(
                input_size=27, # size of each token
                hidden_size=self.hidden_size,
                output_size=output_dim,
                rnn_type='lstm',
                num_layers=1, # number of RNN layers
                num_hidden_layers=1, # num hidden layers per RNN
                nr_cells=10, # number of memory cells
                cell_size=20,
                read_heads=1,
                gpu_id=0,
                debug=False,
                batch_first=True, # shape of input tensor is (batch, seq_len, token_dim)
        )
        self.fc1 = nn.Linear(self.hidden_size, output_dim)

    def forward(self, input):
        input, input_lengths = input[0], input[1]
        
        output, (controller_hidden, memory, read_vectors) = self.dnc(
                input)
        ht = controller_hidden[0][0]
        # ht is (num_layers, batch_size, hidden_dim)
        assert ht.shape[0] == 1
        x = ht.squeeze(0) # (batch_size, hidden_dim)
        x = self.fc1(x)

        return x


class RNNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = 256
        self.rnn = nn.LSTM(
                input_size=27,
                hidden_size=self.hidden_size,
                num_layers=1,
                batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, self.output_dim)

    
    def forward(self, input):
        input, input_lengths = input[0], input[1]

        packed_input = nn.utils.rnn.pack_padded_sequence(input, input_lengths,
                batch_first=True,
                enforce_sorted=False)

        packed_output, (ht, ct) = self.rnn(packed_input)
        # ht is (num_layers, batch_size, hidden_dim)
        assert ht.shape[0] == 1
        x = ht.squeeze(0) # (batch_size, hidden_dim)
        x = self.fc1(x)

        return x

class FCClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, num_labels)


    def forward(self, input):
        if type(input) == list:
            input = input[0] # for dealing with RNN oriented input

        # flatten the second and third axes
        input = input.view(-1, np.prod(input.shape[1:]))


        action_values = self.fc2(F.relu(self.fc1(input)))
        return action_values


def train_test(net, trainloader, valloader, num_epochs, test_every=10,
        criterion=nn.CrossEntropyLoss(), lr=0.0001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0
        if epoch % test_every == 0:
            percent_correct = test(net, trainloader)
            print('Training accuracy: {}'.format(percent_correct))
            percent_correct = test(net, valloader)
            print('Validation accuracy: {}'.format(percent_correct))
            test(net, valloader)

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            labels = labels.cuda()


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

def train_test_mimic_q_learning(net, trainloader, valloader, num_epochs,
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

def test_q(net, valloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            inputs, labels = data
            outputs = net(inputs)

            out_max = torch.argmax(outputs, 1)
            label_max = torch.argmax(labels, 1)
            total += labels.size(0)
            correct += (out_max == label_max).sum().item()

    print('Accuracy of the network: %d %%' % (
            100 * correct / total))

def test(net, valloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            inputs, labels = data
            labels = labels.cuda()

            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def lstm_classification():
    dataset = get_dataset()
    input_dim = 20
    output_dim = 10
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    rnn = RNNClassifier(input_dim, output_dim)
    fc = FCClassifier(input_dim, output_dim)
    num_epochs = 1000
    print('fc results: ')
    train_test(net = fc, trainloader=trainloader, valloader=trainloader, 
            num_epochs=num_epochs, test_every=100)
    print('rnn results: ')
    train_test(net = rnn, trainloader=trainloader, valloader=trainloader, 
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
    train_test_mimic_q_learning(net = fc, trainloader=trainloader, valloader=trainloader, 
            num_epochs=num_epochs, test_every=100)
    print('rnn results: ')
    train_test_mimic_q_learning(net = rnn, trainloader=trainloader, valloader=trainloader, 
            num_epochs=num_epochs, test_every=100)


def name_classification():
    train_dataset = BabyNamesDataset(pad=True, validation=False)
    val_dataset = BabyNamesDataset(pad=True, validation=True)
    input_dim = train_dataset.max_len*27
    output_dim = 2
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
            shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=32,
            shuffle=False)

    fc = FCClassifier(input_dim, output_dim)
#    print('fc results: ')
#    train_test(net=fc, trainloader=trainloader, valloader=valloader,
#            num_epochs=500, test_every=10)

    rnn = RNNClassifier(input_dim, output_dim).cuda()
    print('rnn results: ')
    train_test(net=rnn, trainloader=trainloader, valloader=valloader,
            num_epochs=500, test_every=1)

if __name__ == '__main__':
    name_classification()









