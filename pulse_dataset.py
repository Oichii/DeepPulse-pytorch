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
    PURE, VIPL-hr, optospare and pff pulse dataset. Containing video frames and corresponding to them pulse measurements
    """

    def __init__(self, sequence_list, root_dir, length, img_heiht=120, img_width=120, seq_len=1, transform=None):
        """
        Initialize dataset
        :param sequence_list: list of sequences in dataset
        :param root_dir: directory containing sequences folders
        :param length: number of possible sequences
        :param img_heiht: height of the image
        :param img_width: width of the image
        :param seq_len: length of generated sequence
        :param transform: transforms to apply to data
        """
        seq_list = []
        with open(sequence_list, 'r') as seq_list_file:
            for line in seq_list_file:
                seq_list.append(line.rstrip('\n'))
        self.frames_list = pd.DataFrame()
        for s in seq_list:
            sequence_dir = os.path.join(root_dir, s)
            if sequence_dir[-2:len(sequence_dir)] == '_1':
                fr_list = glob.glob(sequence_dir[0:-2] + '/cropped/*.png')
                fr_list = fr_list[0:len(fr_list) // 2]
            elif sequence_dir[-2:len(sequence_dir)] == '_2':
                fr_list = glob.glob(sequence_dir[0:-2] + '/cropped/*.png')
                fr_list = fr_list[len(fr_list) // 2: len(fr_list)]
            else:
                if os.path.exists(sequence_dir + '/cropped/'):
                    fr_list = glob.glob(sequence_dir + '/*.png')
                else:
                    fr_list = glob.glob(sequence_dir + '/*.png')
                # reference = json.load(open(sequence_dir + '.JSON', 'r'))
            # print(sequence_dir)
            directory, name = os.path.split(sequence_dir)

            new_dir = os.path.join(directory, 'reference_bvp', name)
            # print(new_dir)
            reference = pd.read_csv(sequence_dir+'.txt', sep='\t')  #sep='\t'
            # print(reference)
            # ref = []
            # pulse_rate = []
            # for sample in reference['/FullPackage']:
            #     pulse_rate.append(sample['Value']['waveform'])
            #     ref.append(sample['Value']['waveform'])
            ref = reference.loc[:, 'waveform']
            # ref = reference.loc[:, 'bvp_ica']
            ref = np.array(ref)
            # plt.plot(ref)
            # plt.show()
            # q = round((250-50)/quant)

            ref_resample = resample(ref, len(fr_list))
            ref_resample = (ref_resample - np.mean(ref_resample)) / (np.max(ref_resample)-np.min(ref_resample))
            # ref_resample = (ref_resample - np.mean(ref_resample))/np.std(ref_resample)
            # plt.plot(ref_resample)
            # plt.show()
            # ref_resample = np.round(ref_resample / q)
            # print(fr_list)
            # print(ref_resample)
            self.frames_list = self.frames_list.append(pd.DataFrame({'frames': fr_list, 'labels': ref_resample}))
            # plt.plot(ref_resample)
            # plt.show()

        self.length = length
        self.seq_len = seq_len
        self.img_height = img_heiht
        self.img_width = img_width
        self.root_dir = root_dir
        self.transform = transform
        # print('Found', self.__len__(), "sequences")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # print(idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        frames = []
        flip = np.random.randint(0, 2)
        # print(idx, idx+self.seq_len)
        for fr in range(idx, idx+self.seq_len):  # frames from idx to idx+seq_len
            img_name = os.path.join(self.frames_list.iloc[fr, 0])  # path to image
            image = Image.open(img_name)

            image = image.resize((self.img_width, self.img_height))
            # print(flip, fr)
            # if flip:
            #     image = image.transpose(Image.FLIP_LEFT_RIGHT)
            # plt.imshow(image)
            # plt.show()
            if self.transform:
                image = self.transform(image)

            frames.append(image)

        frames = torch.stack(frames)
        frames = frames.permute(1, 0, 2, 3)
        frames = torch.squeeze(frames, dim=1)
        frames = (frames-torch.mean(frames))/torch.std(frames)*255
        # frames = (frames-torch.min(frames))/(torch.max(frames)-torch.min(frames))
        # print(torch.max(frames), torch.min(frames))
        # labels = torch.tensor(np.mean(self.frames_list.iloc[idx:idx+self.seq_len, 1]), dtype=torch.float)

        lab = np.array(self.frames_list.iloc[idx:idx + self.seq_len, 1])  # <--------
        # lab = resample(lab, self.seq_len-14)
        # lab = (ref_resample - np.mean(ref_resample)) / np.std(ref_resample)
        # print(lab.size, self.seq_len)
        # print(self.frames_list.iloc[idx:idx + self.seq_len])
        labels = torch.tensor(lab, dtype=torch.float)
        # print(torch.mean(frames), torch.min(frames), torch.max(frames))
        # print(labels, self.frames_list.iloc[idx:idx+self.seq_len, 1])
        # plt.plot(self.frames_list.iloc[idx:idx + self.seq_len-14, 1])
        # plt.show()

        sample = (frames, labels)
        return sample
