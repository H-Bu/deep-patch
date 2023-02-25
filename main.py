from filter import filter
from T11T12generator import generator
from re_train import re_train
from feature import feature
from merge import merge
from C_classifier import C_train
from evaluate import evaluate

noise_dic = ['gaussian_noise', 'snow', 'glass_blur', 'pixelate', 'contrast', 'impulse_noise', 'frost', 'motion_blur',
             'shot_noise', 'defocus_blur', 'zoom_blur', 'fog', 'elastic_transform', 'jpeg_compression', 'brightness']
net_dic = ['vgg16', 'densenet121', 'resnet101', 'mobilenetv2']

for i in range(15):
    for j in range(4):
        for k in range(5):
            noise_name = noise_dic[i]
            net_name = net_dic[j]
            print(noise_name, net_name)
            filter(noise_name, net_name)
            generator(noise_name, net_name)
            re_train(noise_name, net_name)
            feature(noise_name, net_name)
            merge(noise_name, net_name)
            C_train(noise_name, net_name)
            evaluate(noise_name, net_name)

