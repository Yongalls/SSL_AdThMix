from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import torch

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return np.array(img)
    elif rot == 1: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1,0,2)))
    elif rot == 2: # 90 degrees rotation
        return np.fliplr(np.flipud(img))
    elif rot == 3: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1,0,2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

class SimpleImageLoader(torch.utils.data.Dataset):
    def __init__(self, rootdir, split, ids, transform, loader=default_image_loader):
        if split == 'test':
            self.impath = os.path.join(rootdir, 'test_data')
            meta_file = os.path.join(self.impath, 'test_meta.txt')
        else:
            self.impath = os.path.join(rootdir, 'train/train_data')
            meta_file = os.path.join(rootdir, 'train/train_label')

        imnames = []
        imclasses = []

        with open(meta_file, 'r') as rf:
            for i, line in enumerate(rf):
                if i == 0:
                    continue
                instance_id, label, file_name = line.strip().split()
                if (split == 'train' and i<50000) or (split == 'val' and i>=50000):
                    if os.path.exists(os.path.join(self.impath, file_name)):
                        imnames.append(file_name)

        self.transform = transform
        self.TransformTwice = TransformTwice(transform)
        self.rotation = rotate_img
        self.loader = loader
        self.split = split
        self.imnames = imnames
        self.imclasses = imclasses

    def __getitem__(self, index):
        filename = self.imnames[index]
        img = self.loader(os.path.join(self.impath, filename))

        rot = np.random.randint(4)
        transforms_toPIL = transforms.Compose([transforms.ToPILImage()])
        #rotated_imgs = [
        #    self.transform(transforms_toPIL(self.rotation(img,0))),
        #    self.transform(transforms_toPIL(self.rotation(img,1))),
        #    self.transform(transforms_toPIL(self.rotation(img,2))),
        #    self.transform(transforms_toPIL(self.rotation(img,3)))
        #]
        rot_img = self.transform(transforms_toPIL(self.rotation(img,rot)))
               
        return rot_img, rot

    def __len__(self):
        return len(self.imnames)
