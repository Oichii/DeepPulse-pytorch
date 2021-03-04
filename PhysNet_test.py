"""
PhysNet based models testing, power spectrum, correlation and errors calculation
"""
import os
import glob
import json
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from pulse_sampler import PulseSampler
from pulse_dataset_3d import PulseDataset
from PhysNet import NegPearson
from PhysNet import PhysNet
from scipy.stats import pearsonr
import heartpy as hp
from utils import butter_bandpass_filter
import torch.nn as nn
import time


resume = 'save_temp/checkpoint.tar'
print("initialize model...")

seq_len = 32
model = PhysNet(seq_len)
model = torch.nn.DataParallel(model)
model.cuda()
ss = sum(p.numel() for p in model.parameters())
print('num params: ', ss)
if os.path.isfile(resume):
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume)
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(resume))

sequence_list = "sequence_test.txt"
root_dir = 'E:/Datasets_PULSE/set_all/'
seq_list = []
end_indexes_test = []
with open(sequence_list, 'r') as seq_list_file:
    for line in seq_list_file:
        seq_list.append(line.rstrip('\n'))

# seq_list = ['test_static']
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
            fr_list = glob.glob(sequence_dir + '/cropped/*.png')
        else:
            fr_list = glob.glob(sequence_dir + '/*.png')
    # print(fr_list)
    end_indexes_test.append(len(fr_list))

end_indexes_test = [0, *end_indexes_test]
# print(end_indexes_test)

sampler_test = PulseSampler(end_indexes_test, seq_len, False)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

pulse_test = PulseDataset(sequence_list, root_dir, seq_len=seq_len,
                                        length=len(sampler_test), transform=transforms.Compose([
                                                                                            transforms.ToTensor(),
                                                                                            normalize]))
val_loader = torch.utils.data.DataLoader(pulse_test, batch_size=1, shuffle=False, sampler=sampler_test, pin_memory=True)

model.eval()
criterion = NegPearson()
criterion = criterion.cuda()

outputs = []
reference_ = []
loss_avg = []
loss_avg2 = []

start = time.time()
for k, (net_input, target) in enumerate(val_loader):
    net_input = net_input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    with torch.no_grad():
        output, x_visual, x, t = model(net_input)

        outputs.append(output[0])
        reference_.append(target[0])
end = time.time()
print(end-start, len(val_loader))
outputs = torch.cat(outputs)

outputs = (outputs - torch.mean(outputs)) / torch.std(outputs)
outputs = outputs.tolist()

reference_ = torch.cat(reference_)
reference_ = (reference_-torch.mean(reference_))/torch.std(reference_)
reference_ = reference_.tolist()

fs = 30
lowcut = 1
highcut = 3

yr = butter_bandpass_filter(outputs, lowcut, highcut, fs, order=4)
yr = (yr - np.mean(yr)) / np.std(yr)

plt.subplots_adjust(right=0.7)
plt.plot(outputs, alpha=0.7, label='wyjście\n sieci')
plt.plot(yr, label='wyjście\n sieci')
plt.plot(reference_, '--', label='referencja\n PPG')

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='large')
plt.ylabel('Amplituda', fontsize='large', fontweight='semibold')
plt.xlabel('Czas [próbka]', fontsize='large', fontweight='semibold')
plt.grid()
plt.xlim([350, 550])
plt.ylim([-2, 3])

plt.savefig('3d.svg', bbox_inches='tight')
plt.show()
reference_ = np.array(reference_)
outputs = np.array(outputs)

bpm_ref = []
bpm_out = []
bmmp_filt = []
bpm_out2 = []
hrv_ref = []
hrv_out = []

win = 255
for i in range(2*win, len(reference_), win):
    peaks, _ = find_peaks(reference_[i:i+win], distance=20, height=0.9)
    peaks_out, _ = find_peaks(yr[i:i + win], height=0.95)
    _, measures2 = hp.process(reference_[i:i+win], 30.0)
    bpm_ref.append(30/(win/len(peaks))*win)
    bmmp_filt.append(measures2['bpm'])
    _, mmm = hp.process(yr[i:i + win], 30.0)
    bpm_out.append(mmm['bpm'])
    bpm_out2.append(30/(win/len(peaks_out))*win)

corr, _ = pearsonr(bmmp_filt, bpm_out)
c = np.corrcoef(bmmp_filt, bpm_out)
cc = np.corrcoef(bpm_ref, bpm_out2)
ccc = np.corrcoef(bmmp_filt, bpm_out2)
print('Correlation:', c)

plt.subplots_adjust(right=0.7)
time = np.arange(0, 3, 1 / fs)
fourier_transform = np.fft.rfft(outputs)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)
frequency = np.linspace(0, fs / 2, len(power_spectrum))
plt.semilogy(frequency, power_spectrum, label='wyjście\n sieci')

fourier_transform = np.fft.rfft(reference_)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)
plt.xlim(-0.1, 10)
plt.ylim(10e-6, 10e6)
plt.semilogy(frequency, power_spectrum, label='referencja\n PPG')
plt.ylabel('|A(f)|', fontsize='large', fontweight='semibold')
plt.xlabel('Częstotliwość f [Hz]', fontsize='large', fontweight='semibold')
plt.title('Częstitliwościowe widmo mocy')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()

reference_ = torch.tensor(reference_)
outputs = torch.tensor(outputs)

criterionMSE = nn.MSELoss()
criterionMAE = nn.L1Loss()
mse = criterionMSE(reference_, outputs)
rmse = torch.sqrt(mse)
mae = criterionMAE(reference_, outputs)
se = torch.std(outputs-reference_)/np.sqrt(outputs.shape[0])
print("MAE: ", mae, "MSE: ", mse, "RMSE: ", rmse, "SE:", se)
