import torch.utils.data as data
import os
from PIL import Image


class MyDataset(data.Dataset):
    def __init__(self, file, dir_path, new_width, new_height, transform=None):
        imgs = []
        fw = open(file, 'r')
        lines = fw.readlines()
        for line in lines:
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.dir_path = dir_path
        self.height = new_height
        self.width = new_width
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.imgs[index]
        idx = path.split('/')[1].split('.')[0]
        path = os.path.join(self.dir_path, path)
        img = Image.open(path).convert('RGB')
       # img = img.resize((self.width, self.height), Image.ANTIALIAS)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, idx

    def __len__(self):
        return len(self.imgs)
