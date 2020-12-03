import torch.nn as nn
import torch.nn.functional as F


class MLP1(nn.Module):
    """
    Shared Multilayer Perceptron composed of a succession of fully connected layers,
    batch-norms and ReLUs
    INPUT (N x L x C)
    """

    def __init__(self):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.fc2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(num_features=64)

    def forward(self, x):
        x = self.fc1(x)
        x = x.transpose(2, 1)
        x = self.bn1(x)     # BN1d takes [batch_size x channels x seq_len]
        x = x.transpose(2, 1)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.transpose(2, 1)
        x = self.bn2(x)   # BN1d takes [batch_size x channels x seq_len]
        x = x.transpose(2, 1)
        x = F.relu(x)
        return x


class MLP2(nn.Module):
    """
    Second Perceptron in the spatial encoder
    INPUT (N x L x C)
    """

    def __init__(self):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(132, 128)
        self.bn1 = nn.BatchNorm1d(num_features=128)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x.transpose(2, 1))     # BN1d takes [batch_size x channels x seq_len]
        x = F.relu(x.transpose(2, 1))
        return x


class MLP3(nn.Module):
    """
    Multilayer Perceptron number three
    INPUT (N x L)
    """

    def __init__(self):
        super(MLP3, self).__init__()
        self.fc1 = nn.Linear(512, 128)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(num_features=128)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class MLP4(nn.Module):
    """
    Decoder Multilayer Perceptron.
    INPUT (N x L)
    """

    def __init__(self):
        super(MLP4, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.fc3 = nn.Linear(32, 20)
        self.bn3 = nn.BatchNorm1d(num_features=20)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

