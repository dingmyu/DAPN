import torch
import numpy as np
import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        print(max(label))
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

class MiniImageNet(Dataset):

    def __init__(self, root='dataset/mini-imagenet/train', dataset='mini-imagenet', mode='train'):
        self.root = root
        self.data = []
        self.label = []
        self.dataset = dataset
        self.mode = mode
        self._load_dataset()
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_dataset(self):
        path = self.root
        fw = open(os.path.join('dataset', self.dataset, self.mode+'.txt'))
        lines = fw.readlines()
        for line in lines:
            img_path = os.path.join(path, line.split()[0])
            labels = int(line.split()[1])
            self.data.append(img_path)
            self.label.append(labels)
        fw.close()

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

    def __len__(self):
        return len(self.data)
