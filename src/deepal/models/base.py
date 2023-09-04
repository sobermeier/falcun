from abc import abstractmethod

import torch
import torch.nn as nn

from .utils import linear_sequential, convolution_sequential

# Credits to https://github.com/sharpenb/Posterior-Network/tree/main/src/architectures


class AbstractModel(nn.Module):
    def __init__(self, input_dim, output_dim, linear_hidden_dims, p_drop):
        super().__init__()
        self.input_dims, self.output_dim, self.linear_hidden_dims = input_dim, output_dim, linear_hidden_dims
        self.p_drop = p_drop
        self.emb_dim = linear_hidden_dims[-1]

    @abstractmethod
    def init_model(self):
        pass

    def get_embedding_dim(self):
        return self.emb_dim


class LinearSeq(AbstractModel):
    def __init__(self, input_dim, output_dim, linear_hidden_dims, p_drop):
        super().__init__(input_dim, output_dim, linear_hidden_dims, p_drop)
        self.linear = self.init_model()

    def init_model(self):
        return linear_sequential(
            input_dims=self.input_dims,
            hidden_dims=self.linear_hidden_dims,
            output_dim=self.output_dim,
            p_drop=self.p_drop
        )

    def forward(self, x: torch.FloatTensor, embedding=False):
        if embedding:
            emb = x
        else:
            batch_size = x.size(0)
            feature_extractor = torch.nn.Sequential(*list(self.linear.children())[:-1])
            emb = feature_extractor(x.view(batch_size, -1))

        head = list(self.linear.children())[-1]
        output = head(emb)
        return output, emb


class ConvLinSeq(AbstractModel):
    def __init__(self, input_dim, output_dim, linear_hidden_dims, p_drop, kernel_dim,
                 conv_hidden_dims):
        super().__init__(input_dim, output_dim, linear_hidden_dims, p_drop)
        self.kernel_dim, self.conv_hidden_dims = kernel_dim, conv_hidden_dims
        self.convolutions, self.linear = self.init_model()

    def init_model(self):
        convolutions = convolution_sequential(input_dims=self.input_dims,
                                              hidden_dims=self.conv_hidden_dims,
                                              kernel_dim=self.kernel_dim,
                                              p_drop=self.p_drop)
        lin_input_dims = [self.conv_hidden_dims[-1] * (self.input_dims[0] // 2 ** len(self.conv_hidden_dims)) * (
            self.input_dims[1] // 2 ** len(self.conv_hidden_dims))]
        linear = linear_sequential(
            input_dims=lin_input_dims,
            hidden_dims=self.linear_hidden_dims,
            output_dim=self.output_dim,
            p_drop=self.p_drop)
        return convolutions, linear

    def forward(self, x: torch.FloatTensor, embedding=False):
        if embedding:
            emb = x
        else:
            batch_size = x.size(0)
            conv_out = self.convolutions(x)
            feature_extractor = torch.nn.Sequential(*list(self.linear.children())[:-1])
            emb = feature_extractor(conv_out.view(batch_size, -1))

        head = list(self.linear.children())[-1]
        output = head(emb)
        return output, emb
