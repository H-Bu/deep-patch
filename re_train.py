"""re-train the network using T1 and save the model"""
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from vgg import VGG
from densenet import DenseNet121
from resnet import ResNet101
from mobilenetv2 import MobileNetV2
from loader import GN


def re_train(noise_name, net_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = GN(data_path='data/' + noise_name + '_train.npy', label_path='data/labels_train.npy',
                  transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    # Model
    lr = 1e-4
    if net_name == 'vgg16':
        net = VGG('VGG16')
    elif net_name == 'densenet121':
        net = DenseNet121()
    elif net_name == 'resnet101':
        net = ResNet101()
    elif net_name == 'mobilenetv2':
        net = MobileNetV2()
        lr = 5e-4
    else:
        print("In function 're_train', net_name doesn't exist.")
        assert False
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    weight = torch.load('net_weight/' + net_name + '_ckpt.pth')
    net.load_state_dict(weight['net'])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Training
    def train():
        net.train()
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    for epoch in range(10):
        train()

    torch.save(net.state_dict(), 'new_weight.pth')
