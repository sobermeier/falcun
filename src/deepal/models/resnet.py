import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet18_Weights, ResNet101_Weights

from .utils import build_mlp


def get_resnet_weights(name):
    if name == "resnet18":
        return ResNet18_Weights.IMAGENET1K_V1
    elif name == "resnet50":
        return ResNet50_Weights.IMAGENET1K_V1
    elif name == "resnet101":
        return ResNet101_Weights.IMAGENET1K_V1
    else:
        return None


# Credits to https://github.com/aminparvaneh/alpha_mix_active_learning
class ResNetClassifier(nn.Module):
    def __init__(self,  output_dim=10, input_dim=3, arch_name='resnet50', pretrained=True, dropout=0.2,
                 fine_tune_layers=-1, emb_size=0):
        super(ResNetClassifier, self).__init__()

        self.n_label = output_dim

        model = getattr(models, arch_name)
        if pretrained:
            resnet = model(weights=get_resnet_weights(arch_name))
        else:
            resnet = model()

        # Remove linear layers
        modules = list(resnet.children())[:-1]
        if modules[0].in_channels != input_dim:
            conv = modules[0]
            modules[0] = nn.Conv2d(in_channels=input_dim, out_channels=conv.out_channels,
                                   kernel_size=conv.kernel_size, stride=conv.stride,
                                   padding=conv.padding, bias=conv.bias)
            pretrained = False

        self.resnet = nn.Sequential(*modules)

        if pretrained:
            self.fine_tune(fine_tune_layers)

        input_size = resnet.fc.in_features
        if emb_size <= 0 or emb_size == input_size:
            self.embedding_size = input_size
            print(self.embedding_size)
            self.hidden_layers = None
        else:
            self.embedding_size = emb_size
            self.hidden_layers = build_mlp(input_size, (), emb_size, dropout=dropout, use_batchnorm=False,
                                     add_dropout_after=False)

        self.classifier = build_mlp(self.embedding_size, (), output_dim,
                                    dropout=dropout,
                                    use_batchnorm=False,
                                    add_dropout_after=False)

    def forward(self, x, embedding=False):
        if embedding:
            embd = x
        else:
            #print(x.size())
            embd = self.resnet(x)
            #print(embd.size())

            batch_size, feature_size, x, y = embd.size()
            embd = embd.view(batch_size, feature_size)

            if self.hidden_layers:
                embd = self.hidden_layers(embd)

        out = self.classifier(embd)

        return out, embd

    def get_embedding_dim(self):
        return self.embedding_size

    def fine_tune(self, fine_tune_layers):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune_layers: How many convolutional layers to be fine-tuned (negative value means all)
        """
        for p in self.resnet.parameters():
            p.requires_grad = False

        # Last convolution layers to be fine-tuned
        for c in list(self.resnet.children())[
                 0 if fine_tune_layers < 0 else len(list(self.resnet.children())) - (1 + fine_tune_layers):]:
            for p in c.parameters():
                p.requires_grad = True

    def get_classifier(self):
        return self.classifier[-1]