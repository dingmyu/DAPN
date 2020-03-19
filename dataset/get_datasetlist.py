import os
import random
root = 'dataset/'
dataset_list = ['mini-imagenet', 'tiered-imagenet']
split_list = ['train', 'val', 'test', 'test_new_domain']
for dataset in dataset_list:
    label_dict = {}
    label_dict_reverse = {}
    idx = 0
    for split in split_list:
        if split != 'test_new_domain':
            sw = open(os.path.join('dataset', dataset, split+'.txt'), 'w+')
            path = os.path.join(root, dataset, split)
            subdirs = os.listdir(path)
            for subdir in subdirs:
                label_dict[subdir] = idx
                label_dict_reverse[idx] = subdir
                imgs = os.listdir(os.path.join(path, subdir))
                for img in imgs:
                    img_path = os.path.join(split, subdir, img)
                    sw.writelines(img_path+" "+str(idx)+"\n")
                idx += 1
            sw.close()
        else:
            sw = open(os.path.join('dataset', dataset, split+'.txt'), 'w+')
            sw2 = open(os.path.join('dataset', dataset, split+'_fsl.txt'), 'w+')
            path = os.path.join(root, dataset, split)
            subdirs = os.listdir(path)
            for subdir in subdirs:
                idx = label_dict[subdir]
                imgs = os.listdir(os.path.join(path, subdir))
                random.shuffle(imgs)
                for i, img in enumerate(imgs):
                    img_path = os.path.join(split, subdir, img)
                    if i < 5:
                        sw.writelines(img_path+" "+str(idx)+"\n")
                    else:
                        sw2.writelines(img_path+" "+str(idx)+"\n")
            sw.close()
            sw2.close()
    sw = open(os.path.join('dataset', dataset, 'label_dict.txt'), 'w+')
    for i in range(len(label_dict_reverse.keys())):
        sw.writelines(label_dict_reverse[i]+" "+str(i)+"\n")
    sw.close()
