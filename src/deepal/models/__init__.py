from src.deepal.models.base import ConvLinSeq, LinearSeq
from src.deepal.models.vgg import VGGClassifier
from src.deepal.models.lenet import LeNet5
from src.deepal.models.resnet import ResNetClassifier
from src.deepal.models.densenet import DenseNetClassifier


def get_net(
    model_architecture: str,
):
    """Create a model from a configuration."""
    if model_architecture == "conv":
        return ConvLinSeq
    elif model_architecture == "linear":
        return LinearSeq
    elif model_architecture == "densenet":
        return DenseNetClassifier
    elif model_architecture == "vgg":
        return VGGClassifier
    elif model_architecture == "LeNet":
        return LeNet5
    elif model_architecture == "resnet":
        return ResNetClassifier
