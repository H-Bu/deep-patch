"""final evaluate (0-noise 1-clear)"""
import joblib
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from vgg import VGG
from densenet import DenseNet121
from resnet import ResNet101
from mobilenetv2 import MobileNetV2
from loader import GN


def evaluate(noise_name, net_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    C = joblib.load('C.model')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testsetT2 = GN(data_path='data/' + noise_name + '_test.npy', label_path='data/labels_test.npy',
                   transform=transform_test)
    testloaderT2 = torch.utils.data.DataLoader(testsetT2, batch_size=1, shuffle=False)
    testsetD2 = GN(data_path='data/cifar-10-final-test.npy', label_path='data/cifar-10-final-test-label.npy',
                   transform=transform_test)
    testloaderD2 = torch.utils.data.DataLoader(testsetD2, batch_size=1, shuffle=False)

    # Model
    if net_name == 'vgg16':
        netC = VGG('VGG16')
    elif net_name == 'densenet121':
        netC = DenseNet121()
    elif net_name == 'resnet101':
        netC = ResNet101()
    elif net_name == 'mobilenetv2':
        netC = MobileNetV2()
    else:
        print("In function 'evaluate', net_name doesn't exist.")
        assert False
    netC = netC.to(device)
    if device == 'cuda':
        netC = torch.nn.DataParallel(netC)
        cudnn.benchmark = True

    netC.load_state_dict(torch.load('new_weight.pth'))

    if net_name == 'vgg16':
        net = VGG('VGG16')
    elif net_name == 'densenet121':
        net = DenseNet121()
    elif net_name == 'resnet101':
        net = ResNet101()
    elif net_name == 'mobilenetv2':
        net = MobileNetV2()
    else:
        print("In function 're_train', net_name doesn't exist.")
        assert False
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    weight = torch.load('net_weight/' + net_name + '_ckpt.pth')
    net.load_state_dict(weight['net'])

    net.eval()
    netC.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloaderT2):
            inputs, targets = inputs.to(device), targets.to(device)
            feature = net.module.new_forward(inputs)
            if C.predict(feature.cpu())[0] == 1:
                outputs = net(inputs)
            else:
                outputs = netC(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('T2_Acc_After: %.3f%% (%d/%d)'
              % (100. * correct / total, correct, total))

    net.eval()
    netC.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloaderD2):
            inputs, targets = inputs.to(device), targets.to(device)
            feature = net.module.new_forward(inputs)
            if C.predict(feature.cpu())[0] == 1:
                outputs = net(inputs)
            else:
                outputs = netC(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('D2_Acc_After: %.3f%% (%d/%d)'
              % (100. * correct / total, correct, total))

