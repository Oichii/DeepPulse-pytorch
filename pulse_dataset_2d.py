import json
import cv2
import glob
import os
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import resample


class PulseDataset(Dataset):
    """
    PURE pulse dataset. Containing video frames and corresponding to them pulse signal.
    Frames in shape [c x w x h] and label with corresponding PPG value.
    """

    def __init__(self, sequence_list, root_dir, img_w=128, img_h=192, transform=None):
        """
        Initialize dataset
        :param sequence_list: list of sequences in dataset
        :param root_dir: directory containing sequences folders
        :param img_h: height of frame
        :param img_w: width of frame
        :param transform: transforms to apply to data
        """
        seq_list = []
        with open(sequence_list, 'r') as seq_list_file:
            for line in seq_list_file:
                seq_list.append(line.rstrip('\n'))

        self.frames_list = pd.DataFrame()
        for s in seq_list:
            sequence_dir = os.path.join(root_dir, s)
            if os.path.exists(sequence_dir + '/cropped/'):
                fr_list = glob.glob(sequence_dir + '/cropped/*.png')
            else:
                fr_list = glob.glob(sequence_dir + '/*.png')

            path, s_name = os.path.split(sequence_dir)
            p = os.path.join(path,  s_name)
            reference = pd.read_csv(p + '.txt', sep='\t')

            ref = reference.loc[:, 'waveform']
            ref = np.array(ref)

            ref_resample = resample(ref, len(fr_list))
            ref_resample = (ref_resample-np.mean(ref_resample))/np.std(ref_resample)

            self.frames_list = self.frames_list.append(pd.DataFrame({'frames': fr_list, 'labels': ref_resample}))

        self.img_w = img_w
        self.img_h = img_h
        self.root_dir = root_dir
        self.transform = transform
        print('Found', self.__len__(), "frames belonging to:", len(seq_list), "sequences")

    def __len__(self):
        return self.frames_list.__len__()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mm = []

        img_name = os.path.join(self.frames_list.iloc[idx, 0])  # path to image
        image = Image.open(img_name)
        image = image.resize((self.img_w, self.img_h))

        _, b, _ = image.split()
        mean_img = np.mean(b)
        mm.append(mean_img)
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.frames_list.iloc[idx, 1], dtype=torch.float)

        image = torch.as_tensor(image)
        image = (image - torch.mean(image))/torch.std(image)*255
        sample = (image, label)
        return sample
