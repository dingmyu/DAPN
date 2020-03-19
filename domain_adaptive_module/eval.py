import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math


from torchvision import models

class ResNetFc(nn.Module):
  def __init__(self, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
    super(ResNetFc, self).__init__()
    model_resnet = models.resnet50(pretrained=True)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
            self.fc.apply(init_weights)
            self.__in_features = model_resnet.fc.in_features
    else:
        self.fc = model_resnet.fc
        self.__in_features = model_resnet.fc.in_features

  def forward(self, x):
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)
    y = self.fc(x)
    return x, y



def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
#         if test_10crop:
#             iter_test = [iter(loader['test'][i]) for i in range(10)]
#             for i in range(len(loader['test'][0])):
#                 data = [iter_test[j].next() for j in range(10)]
#                 inputs = [data[j][0] for j in range(10)]
#                 labels = data[0][1]
#                 for j in range(10):
#                     inputs[j] = inputs[j].cuda()
#                 labels = labels
#                 outputs = []
#                 for j in range(10):
#                     _, predict_out = model(inputs[j])
#                     outputs.append(nn.Softmax(dim=1)(predict_out))
#                 outputs = sum(outputs)
#                 if start_test:
#                     all_output = outputs.float().cpu()
#                     all_label = labels.float()
#                     start_test = False
#                 else:
#                     all_output = torch.cat((all_output, outputs.float().cpu()), 0)
#                     all_label = torch.cat((all_label, labels.float()), 0)
#         else:
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            print(inputs.size(), labels.size())
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            print(outputs.size())
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = all_output.squeeze(2).squeeze(2).numpy()
    all_label = all_label.cpu().numpy()
    length = len(all_label)
    f = open('feature15_all.txt','w')
    for i in range(length):
        line = all_output[i]
        for item in line:
            print (item, end =' ', file=f)
        print(int(all_label[i]), file=f)
    f.close()
    return 


def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    print(config, data_config)
#     dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
#                                 transform=prep_dict["source"])
#     dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
#             shuffle=True, num_workers=4, drop_last=True)
#     dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
#                                 transform=prep_dict["target"])
#     dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
#             shuffle=True, num_workers=4, drop_last=True)

#     if prep_config["test_10crop"]:
#         for i in range(10):
#             dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
#                                 transform=prep_dict["test"][i]) for i in range(10)]
#             dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
#                                 shuffle=False, num_workers=4) for dset in dsets['test']]
#     else:


    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                            transform=prep_dict["test"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                            shuffle=False, num_workers=4)

    print ('load data finished')
    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    #base_network = net_config["name"](**net_config["params"])
    #base_network = base_network.cuda()
    #print(base_network)
    
    base_network = torch.load('snapshot/iter_90000_model.pth.tar')
    base_network= list(base_network.children())[0]
    base_network = base_network.module
#     for key, value in base_network.state_dict().items():
#         print(key)
    base_network = list(base_network.children())[9].cuda()
    # base_network= torch.nn.Sequential(*list(base_network.children())[:-1]).cuda()
    base_network = nn.DataParallel(base_network)
    print(base_network)
    #base_network.load_state_dict(checkpoint, strict=False)
#     ## add additional network for some methods
#     if config["loss"]["random"]:
#         random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
#         ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
#     else:
#         random_layer = None
#         ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
#     if config["loss"]["random"]:
#         random_layer.cuda()
#     ad_net = ad_net.cuda()
#     parameter_list = base_network.get_parameters()# + ad_net.get_parameters()
 
#     ## set optimizer
#     optimizer_config = config["optimizer"]
#     optimizer = optimizer_config["type"](parameter_list, \
#                     **(optimizer_config["optim_params"]))
#     param_lr = []
#     for param_group in optimizer.param_groups:
#         param_lr.append(param_group["lr"])
#     schedule_param = optimizer_config["lr_param"]
#     lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

#     gpus = config['gpu'].split(',')
#     if len(gpus) > 1:
# #         ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
#         base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])
        

    ## train   
#     len_train_source = len(dset_loaders["source"])
#     len_train_target = len(dset_loaders["target"])
#     transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
#     best_acc = 0.0

    
    base_network.train(False)
    image_classification_test(dset_loaders, \
        base_network, test_10crop=prep_config["test_10crop"])


    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='CDAN+E', choices=['CDAN', 'CDAN+E', 'DANN'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home', 'imagenet'], help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='dataset/mini-imagenet/list/train_list.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='dataset/mini-imagenet/list/test_transfer_20.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=5000000000, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    # train config
    config = {}
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["num_iterations"] = 100004
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = "snapshot/" + args.output_dir
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop":False, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    config["loss"] = {"trade_off":1.0}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name":network.AlexNetFc, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "VGG" in args.net:
        config["network"] = {"name":network.VGGFc, \
            "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    config["dataset"] = args.dset
    config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":200}, \
                      "target":{"list_path":args.t_dset_path, "batch_size":8}, \
                      "test":{"list_path":args.t_dset_path, "batch_size":256}}

    if config["dataset"] == "office":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
           ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
             ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters       
        config["network"]["params"]["class_num"] = 31 
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 65
    elif config["dataset"] == "imagenet":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 1000
        config['loss']["trade_off"] = 1.0
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config))
    config["out_file"].flush()
    train(config)
