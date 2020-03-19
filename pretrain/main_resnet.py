import shutil
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torchvision import transforms
from torch.utils.data import DataLoader

from dataloader import *
import resnet
from resnet import ResNetFc

parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', default='DATASET_DIR')
parser.add_argument('--arch', default='resnet50',
                    choices=['resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=4096, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--iter-size', default=4, type=int,
                    metavar='I', help='iter size as in Caffe to reduce memory usage (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--new_length', default=400, type=int)
parser.add_argument('--new_width', default=400, type=int)

parser.add_argument('--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--save-freq', default=5, type=int,
                    metavar='N', help='save frequency (default: 200)')
parser.add_argument('--resume', default='output', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec = 0


def main():
    global args, best_prec
    args = parser.parse_args()
    print ("Build model ...")
    model = build_model()
    if not os.path.exists(args.resume):
        os.makedirs(args.resume)
    print("Saving everything to directory %s." % (args.resume))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    cudnn.benchmark = True

    # data transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
                        transforms.RandomRotation(20),
                        transforms.RandomResizedCrop(84, scale=(0.8, 1.2), ratio=(0.75, 1.3333333333333333), interpolation=2),
                        #transforms.RandomCrop(84),
                        transforms.RandomHorizontalFlip(),
                       # transforms.RandomVerticalFlip(),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
    val_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])

    train_data = MyDataset(os.path.join(args.dir_path, 'trainval_list.txt'), args.dir_path, args.new_width, args.new_length,train_transform)
#    val_data = MyDataset(os.path.join(args.dir_path, 'val.txt'), args.dir_path, args.new_width, args.new_length, val_transform)
    
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True)
#    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            #pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        print ('epoch: ' + str(epoch + 1))
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        #prec = validate(val_loader, model, criterion)

        # remember best prec and save checkpoint
        #is_best = prec > best_prec
        #best_prec = max(prec, best_prec)

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, checkpoint_name, args.resume)


def build_model():
    model = ResNetFc(class_num=448)
    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()
    return model


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.float().cuda(async=True)
        #print(input.size(),target.size())
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec = accuracy(output.data, target)
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, _) in enumerate(val_loader):
        input = input.float().cuda(async=True)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec = accuracy(output.data, target)
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec {top1.avg:.3f} '
          .format(top1=top1))

    return top1.avg


def save_checkpoint(state, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    torch.save(state, cur_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        # param_group['lr'] = param_group['lr']/2


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    maxk = 1
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[0].view(-1).float().sum(0)
    res = correct_k.mul_(100.0 / batch_size)
    return res

if __name__ == '__main__':
    main()
