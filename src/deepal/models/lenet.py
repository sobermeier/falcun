import torch.nn as nn
from torch.nn import Module


class LeNet5(Module):
    def __init__(self, output_dim=10, input_dim=3):
        super(LeNet5, self).__init__()
        input_lin = 256 if input_dim == 1 else 400
        self.conv1 = nn.Conv2d(input_dim, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(input_lin, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, output_dim)
        # self.relu5 = nn.ReLU()

    def forward(self, x, embedding=False):
        if embedding:
            emb = x
        else:
            y = self.conv1(x)
            y = self.relu1(y)
            y = self.pool1(y)
            y = self.conv2(y)
            y = self.relu2(y)
            y = self.pool2(y)
            y = y.view(y.shape[0], -1)
            y = self.fc1(y)
            y = self.relu3(y)
            y = self.fc2(y)
            emb = self.relu4(y)
        y = self.fc3(emb)
        # y = self.relu5(y)
        return y, emb

    def get_embedding_dim(self):
        return 84

    @staticmethod
    def update_class_counts(class_counts):
        """ Placeholder function """
        print('Just a placeholder')
