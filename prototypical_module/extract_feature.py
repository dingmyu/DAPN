import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
from imagenet import ImageNet
from resnet import *
import numpy as np
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric

def main():
    set_gpu('0')
    save_path = 'features/test_new_domain_miniimagenet/'
    test_set = ImageNet(root='../cross-domain-fsl/dataset/mini-imagenet/test_new_domain')
    val_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=8,
                            pin_memory=True)
    model = resnet50()
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('save/proto-5/max-acc.pth'))

    # model_dict = model.state_dict()
    # pretrained_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
    # # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # 2. overwrite entries in the existing state dict
    # print pretrained_dict
    # model_dict.update(pretrained_dict)
    # # 3. load the new state dict
    # model.load_state_dict(model_dict)
    model = model.cuda()
    model.eval()
    # model = torch.nn.DataParallel(model).cuda()
    features = [[] for i in range(359)]
    for (image, label) in val_loader:
        image = image.cuda()
        label = label.numpy()
        feature = model(image)
        feature = feature.data.cpu().numpy()
        # print feature.shape[0]
        for j in range(feature.shape[0]):
            features[int(label[j])].append(feature[j])
    for i in range(359):
        save_file = os.path.join(save_path, str(i)+'.txt')
        feature_np = np.asarray(features[i])
        np.savetxt(save_file, feature_np)

if __name__ == '__main__':
    main()
