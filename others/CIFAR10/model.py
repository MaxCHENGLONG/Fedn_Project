import collections

import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from fedn.utils.helpers.helpers import get_helper

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


def compile_model():
    """Compile the pytorch model.

    :return: The compiled model.
    :rtype: torch.nn.Module
    """
    class CifarResNet(nn.Module):
        def __init__(
            self,
            layers=[1, 1, 1],
            num_classes=10,
            channels=[64, 128, 256]
        ):
            super(CifarResNet, self).__init__()
            self.inplanes = channels[0]

            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels[0])
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Identity()

            self.layer1 = self._make_layer(resnet.BasicBlock, channels[0], layers[0])
            self.layer2 = self._make_layer(resnet.BasicBlock, channels[1], layers[1], stride=2)
            self.layer3 = self._make_layer(resnet.BasicBlock, channels[2], layers[2], stride=2)

            # 动态计算全连接层输入大小
            dummy_input = torch.randn(1, 3, 32, 32)
            dummy_output = self.layer3(self.layer2(self.layer1(self.maxpool(self.relu(self.bn1(self.conv1(dummy_input)))))))
            num_features = dummy_output.view(1, -1).size(1)

            self.fc = nn.Linear(num_features, num_classes)

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))
            
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x


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