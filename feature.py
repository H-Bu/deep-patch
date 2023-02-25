"""obtain the features of inputs"""
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from vgg import VGG
from densenet import DenseNet121
from resnet import ResNet101
from mobilenetv2 import MobileNetV2
from loader import GN
import numpy as np
import pathlib


def feature(noise_name, net_name):
    path = pathlib.Path(noise_name+net_name+'feature_T12.npy')
    if path.is_file():
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset11 = GN(data_path=noise_name+net_name+'T11.npy', label_path=noise_name+net_name+'T11_labels.npy',
                   transform=transform_test)
    testloader11 = torch.utils.data.DataLoader(testset11, batch_size=100, shuffle=False)
    testset12 = GN(data_path=noise_name+net_name+'T12.npy', label_path=noise_name+net_name+'T12_labels.npy',
                   transform=transform_test)
    testloader12 = torch.utils.data.DataLoader(testset12, batch_size=100, shuffle=False)

    # Model
    if net_name == 'vgg16':
        net = VGG('VGG16')
    elif net_name == 'densenet121':
        net = DenseNet121()
    elif net_name == 'resnet101':
        net = ResNet101()
    elif net_name == 'mobilenetv2':
        net = MobileNetV2()
    else:
        print("In function 'feature', net_name doesn't exist.")
        assert False
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    weight = torch.load('net_weight/' + net_name + '_ckpt.pth')
    net.load_state_dict(weight['net'])

    T11_feature = []
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(testloader11):
            inputs = inputs.to(device)
            outputs = net.module.new_forward(inputs)
            T11_feature.append(outputs.cpu().numpy())

    m = np.concatenate(T11_feature)
    np.save(noise_name+net_name+'feature_T11.npy', m)

    T12_feature = []
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(testloader12):
            inputs = inputs.to(device)
            outputs = net.module.new_forward(inputs)
            T12_feature.append(outputs.cpu().numpy())

    n = np.concatenate(T12_feature)
    np.save(noise_name+net_name+'feature_T12.npy', n)
