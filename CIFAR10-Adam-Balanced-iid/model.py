import collections

import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
from fedn.utils.helpers.helpers import get_helper

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


def compile_model():
    """Compile the pytorch model.

    :return: The compiled model.
    :rtype: torch.nn.Module
    """
    class CifarResNet(ResNet):
        def __init__(
            self,
            layers=[1, 1, 1, 1],
            num_classes=10,
            channels=[64, 128, 256, 512]
        ):
            super().__init__(
                block=BasicBlock,
                layers=layers,
                num_classes=num_classes
            )
            self.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                    )
            self.maxpool = nn.Identity()
            if channels != [64, 128, 256, 512]:
                self.inplanes = channels[0]
                self.layer1 = self._make_layer(
                        BasicBlock, channels[0], layers[0]
                        )
                self.layer2 = self._make_layer(
                        BasicBlock, channels[1], layers[1], stride=2
                        )
                self.layer3 = self._make_layer(
                        BasicBlock, channels[2], layers[2], stride=2
                        )
                self.layer4 = self._make_layer(
                        BasicBlock, channels[3], layers[3], stride=2
                        )
                self.fc = nn.Linear(
                        channels[3] * BasicBlock.expansion, num_classes
                        )
    model = CifarResNet()
    return model

def save_parameters(model, out_path):
    """Save model paramters to file.

    :param model: The model to serialize.
    :type model: torch.nn.Module
    :param out_path: The path to save to.
    :type out_path: str
    """
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    helper.save(parameters_np, out_path)


def load_parameters(model_path):
    """Load model parameters from file and populate model.

    param model_path: The path to load from.
    :type model_path: str
    :return: The loaded model.
    :rtype: torch.nn.Module
    """
    model = compile_model()
    parameters_np = helper.load(model_path)
    
    if parameters_np is None:
        raise ValueError(f"Failed to load parameters from {model_path}. File might be corrupted or missing.")
    params_dict = zip(model.state_dict().keys(), parameters_np)
    state_dict = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model


def init_seed(out_path="seed.npz"):
    """Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    # Init and save
    model = compile_model()
    save_parameters(model, out_path)


if __name__ == "__main__":
    init_seed("../seed.npz")