import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os, sys
import time
import numpy as np
import random

def get_feature(root='./features/test/', k=5):
    Xtr = []
    Xte = []
    Ytr = []
    Yte = []
    train = []
    for i in range(359):
        fw = open(os.path.join(root, str(i)+'.txt'), 'r')
        lines = fw.readlines()
        for j in range(len(lines)):
            line = lines[j].strip()
            features = map(float, line.split())
            if j < k:
                train.append((features, i))
            else:
                Xte.append(features)
                Yte.append(i)
    random.shuffle(train)
    for i in range(len(train)):
        Xtr.append(train[i][0])
        Ytr.append(train[i][1])
    print Ytr
    Xtr = np.array(Xtr, np.float)
    Xte = np.array(Xte, np.float)
    Ytr = np.array(Ytr, np.int)
    Yte = np.array(Yte, np.int)
    return Xtr, Xte, Ytr, Yte


def train(model, X, labels, opts):
    optimizer = optim.SGD(model.parameters(),
                          lr = opts.lr,
                          momentum = opts.mom,
                          weight_decay=opts.wd)
    N = X.shape[0]
    er = 0
    for it in range(0, opts.maxiter):
        dt = opts.lr
        model.train()
        optimizer.zero_grad()

        idx = np.random.randint(0,N,opts.batchsize)
        x = Variable(torch.Tensor(X[idx]))
        y = Variable(torch.from_numpy(labels[idx]).long())
        yhat = model(x)
        print(x.size(),y.size(),yhat.size())
        loss = F.nll_loss(yhat, y)
        er = er + loss.data.item()
        loss.backward()
        optimizer.step()

        if it % opts.verbose == 1:
            print(er/opts.verbose)
            er = 0
    return er/opts.verbose



def train_balanced(model, X, labels, opts, freq):
    optimizer = optim.SGD(model.parameters(),
                          lr = opts.lr,
                          momentum = opts.mom,
                          weight_decay=opts.wd)
    unq, inv, cnt = np.unique(labels,
                              return_inverse=True,
                              return_counts=True)
    lid = np.split(np.argsort(inv), np.cumsum(cnt[:-1]))
    N = X.shape[0]
    er = 0
    nlabels = len(lid)
    llid = np.zeros(nlabels).astype('int')
    for i in range(nlabels):
        llid[i] = len(lid[i])
    t0 = time.time()
    model = model.cuda()
    for it in range(opts.maxiter):
        dt = opts.lr
        model.train()
        optimizer.zero_grad()
        idx = np.random.randint(0,nlabels,opts.batchsize)
        for t in range(opts.batchsize):
            i = idx[t]
            idx[t] = lid[i][np.random.randint(0,llid[i])]
        x = Variable(torch.Tensor(X[idx]))
        y = Variable(torch.from_numpy(labels[idx]).long())
        x = x.cuda()
        y = y.cuda()
        yhat = model(x)
 #       print(x.size(),y.size(),yhat.size())
        loss = F.nll_loss(yhat, y)
        er = er + loss.data.item()
        loss.backward()
        optimizer.step()

        if it % opts.verbose == 1:
            print(er/opts.verbose)
            er = er/opts.verbose
        if (it+1) % freq == 0:
            print('[%.3fs] iteration %d' % (time.time() - t0, it + 1))
            train_acc = validate(Xtr, Ytr, model)
            test_acc = validate(Xte, Yte, model)
    return er


def validate(tX, tlabels, model):
        """ compute top-1 and top-5 error of the model """
        model.eval()
        N = tX.shape[0]
        batch = tX
        batchlabels = tlabels
        x = Variable(torch.Tensor(batch), volatile=True)
        x = x.cuda()
        yhat = model(x)
        values, indices = torch.max(yhat, 1)
        indices = indices.data.cpu().numpy()
        print(indices.shape, tlabels.shape)
        print('###################  acc is : ', float(sum(indices==tlabels))/float(len(tlabels)))
        return float(sum(indices==tlabels))/float(len(tlabels))

class Net(nn.Module):
    def __init__(self, d, nclasses):
        super(Net, self).__init__()
        self.l1 = nn.Linear(d,nclasses)
    def forward(self, x):
        return F.log_softmax(self.l1(x))




##########################################################3
#



parser = argparse.ArgumentParser()
parser.add_argument('--nlabeled', type=int, default = 2)
parser.add_argument('--maxiter', type=int, default = 1500)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--verbose', type=int, default=500)
parser.add_argument('--pcadim', type=int, default=2048)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--lr', type=float, default=.01)
parser.add_argument('--wd', type=float, default=0.0)
parser.add_argument('--mom', type=float, default=0.0)
parser.add_argument('--mode', default='val')
parser.add_argument('--dataset', default='test')
parser.add_argument('--storemodel', type=str, default='')
parser.add_argument('--storeL', type=str, default='')


opts = parser.parse_args()
mode = opts.mode

root = os.path.join('./features', opts.dataset)
Xtr, Xte, Ytr, Yte = get_feature(root=root, k=opts.nlabeled)


nclasses = 359
Xtr_orig = Xtr
Xte_orig = Xte

print("dataset sizes: Xtr %s (%d labeled), Xte %s, %d classes (eval on %s)" % (
    Xtr.shape, (Ytr >= 0).sum(),
    Xte.shape, nclasses, Yte))

net= Net(opts.pcadim, nclasses)


print('============== start logreg')


if opts.mode == 'val':
    eval_freq = 500
else:
    eval_freq = opts.maxiter

train_balanced(net, Xtr, Ytr, opts, eval_freq)

# if opts.storeL:
#     L = validate(Xte, Yte, net)
#     print('writing', opts.storeL)
#     np.save(opts.storeL, L)

if opts.storemodel:
    print('writing', opts.storemodel)
    torch.save(net.state_dict(), opts.storemodel)