
from __future__ import print_function
import Image
import os
import os.path
import errno
import numpy as np
import sys
import cv, cv2

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(class_file):
    with open(class_file) as r:
        classes = map(lambda s : s.strip(), r.readlines())
    
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx

def loadCVImage(path):
    img = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)
    trans_img = cv2.cvtColor(img, cv.CV_BGR2RGB)
    return trans_img.swapaxes(0, 2).swapaxes(1, 2)

def make_dataset(is_train, dir, class_to_idx):
    images = []

    if is_train:
        for fname in sorted(os.listdir(dir)):
            cls_fpath = os.path.join(dir, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, 'images')
                for imgname in sorted(os.listdir(cls_imgs_path)):
                    if is_image_file(imgname):
                        path = os.path.join(cls_imgs_path, imgname)
                        item = (loadCVImage(path), class_to_idx[fname])
                        images.append(item)
    else:
        imgs_path = os.path.join(dir, 'images')
        imgs_annotations = os.path.join(dir, 'val_annotations.txt')
        
        with open(imgs_annotations) as r:
            data_info = map(lambda s : s.split('\t'), r.readlines())
        
        cls_map = {line_data[0]: line_data[1] for line_data in data_info}
        
        for imgname in sorted(os.listdir(imgs_path)):
            if is_image_file(imgname):
                path = os.path.join(imgs_path, imgname)
                item = (loadCVImage(path), class_to_idx[cls_map[imgname]])
                images.append(item)

    return images

class TinyImagenet200(data.Dataset):
    """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``tiny-imagenet-200`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    base_folder = 'tiny-imagenet-200'
    download_fname = "tiny-imagenet-200.zip"
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.fpath = os.path.join(root, self.download_fname)

        if download:
            self.download()

        if not check_integrity(self.fpath, self.md5):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        classes, class_to_idx = find_classes(os.path.join(self.root, self.base_folder, 'wnids.txt'))
        
        dirname = ''
        if self.train:
            dirname = 'train'
        else:
            dirname = 'val'

        self.data = make_dataset(self.train, os.path.join(self.root, self.base_folder, dirname), class_to_idx)
        
        if len(self.data) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # if self.train:
        #     img, target = self.train_data[index], self.train_labels[index]
        # else:
        #     img, target = self.test_data[index], self.test_labels[index]

        img, target = self.data[index][0], self.data[index][1]

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        import zipfile

        if check_integrity(self.fpath, self.md5):
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.base_folder, self.md5)

        # extract file
        dataset_zip = zipfile.ZipFile(self.fpath)
        dataset_zip.extractall()
        dataset_zip.close



