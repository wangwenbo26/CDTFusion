import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt


NUM_CLASSES = 6
CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]

def extract_number(filename):
    return int(''.join(filter(str.isdigit, filename)))

class CustomDataset(Dataset):
    def __init__(self, root_dir_rgb, root_dir_ir, root_dir_seg, transform=None):
        self.root_dir_rgb = root_dir_rgb
        self.root_dir_ir = root_dir_ir
        self.root_dir_seg = root_dir_seg
        self.transform = transform
        self.rgb_images = sorted(os.listdir(root_dir_rgb), key=extract_number)
        self.ir_images = sorted(os.listdir(root_dir_ir), key=extract_number)
        self.seg_images = sorted(os.listdir(root_dir_seg), key=extract_number)

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_image_name = os.path.join(self.root_dir_rgb, self.rgb_images[idx])
        ir_image_name = os.path.join(self.root_dir_ir, self.ir_images[idx])
        seg_image_name = os.path.join(self.root_dir_seg, self.seg_images[idx])

        rgb_image = Image.open(rgb_image_name).convert("RGB")
        ir_image = Image.open(ir_image_name).convert("L")
        seg_image = Image.open(seg_image_name).convert("RGB")

        rgb_segimage = np.array(seg_image)

        segmentation_mask = np.zeros((rgb_segimage.shape[0], rgb_segimage.shape[1]), dtype=np.uint8)

        for class_idx in range(NUM_CLASSES):
            class_color = PALETTE[class_idx]
            class_mask = np.all(rgb_segimage == class_color, axis=-1)
            segmentation_mask[class_mask] = class_idx



        if self.transform:
            rgb_image = self.transform(rgb_image)
            ir_image = self.transform(ir_image)
            mask = torch.from_numpy(segmentation_mask).long()

        return rgb_image, ir_image, mask

def get_train_loader(dir_rgb_train, dir_ir_train, dir_seg_train, batch_size):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(root_dir_rgb=dir_rgb_train, root_dir_ir=dir_ir_train, root_dir_seg=dir_seg_train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader

def get_test_loader(dir_rgb_test, dir_ir_test, dir_seg_test, batch_size):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    test_dataset = CustomDataset(root_dir_rgb=dir_rgb_test, root_dir_ir=dir_ir_test, root_dir_seg=dir_seg_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
