import torch.nn as nn
import numpy as np


def linear_sequential(input_dims, hidden_dims, output_dim, p_drop=None):
    dims = [np.prod(input_dims)] + hidden_dims + [output_dim]
    num_layers = len(dims) - 1
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < num_layers - 1:
            layers.append(nn.ReLU())
            if p_drop is not None:
                layers.append(nn.Dropout(p=p_drop))
    return nn.Sequential(*layers)


def convolution_sequential(input_dims, hidden_dims, kernel_dim, p_drop=None):
    channel_dim = input_dims[2]
    dims = [channel_dim] + hidden_dims
    num_layers = len(dims) - 1
    layers = []
    for i in range(num_layers):
        layers.append(nn.Conv2d(dims[i], dims[i + 1], kernel_dim, padding=(kernel_dim - 1) // 2))
        layers.append(nn.ReLU())
        if p_drop is not None:
            layers.append(nn.Dropout(p=p_drop))
        layers.append(nn.MaxPool2d(2, padding=0))
    return nn.Sequential(*layers)


def build_mlp(input_dim,
              hidden_dims,
              output_dim,
              use_batchnorm=False,
              dropout=0,
              add_sigmoid=False,
              add_non_linearity_after=False,
              leaky_relu=False,
              add_dropout_after=True,
              add_dropout_batchnorm_before=True):
    layers = []
    D = input_dim
    if dropout > 0 and add_dropout_batchnorm_before:
        layers.append(nn.Dropout(p=dropout))
    if use_batchnorm and add_dropout_batchnorm_before:
        layers.append(nn.BatchNorm1d(input_dim))

    for dim in hidden_dims:
        layers.append(nn.Linear(D, dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(dim))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        if leaky_relu:
            layers.append(nn.LeakyReLU())
        else:
            layers.append(nn.ReLU())
        D = dim
    layers.append(nn.Linear(D, output_dim))

    if add_dropout_after and dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if add_non_linearity_after:
        layers.append(nn.ReLU())

    if add_sigmoid:
        layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)