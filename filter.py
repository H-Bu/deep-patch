"""run network on T1 and get the index of T11 and T12"""
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


def filter(noise_name, net_name):
    path = pathlib.Path(noise_name+net_name+'T11_index.npy')
    if path.is_file():
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = GN(data_path='data/'+noise_name+'_train.npy', label_path='data/labels_train.npy',
                 transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

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
        print("In function 'filter', net_name doesn't exist.")
        assert False
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    weight = torch.load('net_weight/' + net_name + '_ckpt.pth')
    net.load_state_dict(weight['net'])

    T11_index = []
    T12_index = []

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            if predicted != targets:
                T11_index.append(batch_idx)
            else:
                T12_index.append(batch_idx)

    m = np.array(T11_index)
    np.save(noise_name+net_name+'T11_index.npy', m)
    n = np.array(T12_index)
    np.save(noise_name+net_name+'T12_index.npy', n)
