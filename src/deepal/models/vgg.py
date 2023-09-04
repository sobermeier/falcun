import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights, VGG11_Weights, VGG13_Weights, VGG19_Weights

from .utils import build_mlp


def get_vgg_weights(name):
    if name == "vgg11":
        return VGG11_Weights.IMAGENET1K_V1
    elif name == "vgg13":
        return VGG13_Weights.IMAGENET1K_V1
    elif name == "vgg16":
        return VGG16_Weights.IMAGENET1K_V1
    elif name == "vgg16":
        return VGG19_Weights.IMAGENET1K_V1
    else:
        return None


# Credits to https://github.com/aminparvaneh/alpha_mix_active_learning
class VGGClassifier(nn.Module):
    def __init__(self,  output_dim=10, input_dim=3, arch_name='vgg11', pretrained=True, dropout=0.,
                 fine_tune_layers=-1, emb_size=0):
        super(VGGClassifier, self).__init__()

        self.n_label = output_dim

        model = getattr(models, arch_name)

        if pretrained:
            self.vgg = model(weights=get_vgg_weights(arch_name))
        else:
            self.vgg = model()

        # Remove linear layer
        modules = list(self.vgg.features.children())
        if modules[0].in_channels != input_dim:
            conv = modules[0]
            modules[0] = nn.Conv2d(in_channels=input_dim, out_channels=conv.out_channels,
                                   kernel_size=conv.kernel_size, stride=conv.stride,
                                   padding=conv.padding)
            pretrained = False

        self.features = nn.Sequential(*modules)

        if pretrained:
            self.fine_tune(fine_tune_layers)

        input_size = list(self.vgg.classifier.children())[-1].in_features
        if emb_size <= 0 or emb_size == input_size:
            self.embedding_size = input_size
            self.hidden_layers = None
        else:
            self.embedding_size = emb_size
            self.hidden_layers = build_mlp(input_size, (), emb_size, dropout=0, use_batchnorm=False,
                                           add_dropout_after=False)

        modules = list(self.vgg.classifier.children())[:-1]
        self.vgg.classifier = nn.Sequential(*modules)

        self.classifier = build_mlp(self.embedding_size, (), output_dim,
                                    dropout=dropout,
                                    use_batchnorm=False,
                                    add_dropout_after=False)

    def forward(self, x, embedding=False):
        if embedding:
            embd = x
        else:
            embd = self.vgg(x)
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
        for p in self.features.parameters():
            p.requires_grad = False

        # Last convolution layers to be fine-tuned
        for c in list(self.features.children())[
                 0 if fine_tune_layers < 0 else len(list(self.features.children())) - (1 + fine_tune_layers):]:
            for p in c.parameters():
                p.requires_grad = True