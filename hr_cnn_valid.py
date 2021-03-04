"""
HR-CNN validation script, power spectrum, correlation and errors calculation
"""
import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import heartpy as hp
from hr_cnn import HrCNN
import pulse_dataset_2d
from utils import butter_bandpass_filter
import time

resume = 'save_temp/model_path.tar'

print("initialize model...")
model = HrCNN(3)
model = torch.nn.DataParallel(model)

model.cuda()

ss = sum(p.numel() for p in model.parameters())
print('num params: ', ss)

print("loading model...")
if os.path.isfile(resume):
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume)
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(resume))

pulse_test = pulse_dataset_2d.PulseDataset("sequence_test.txt", 'E:/Datasets_PULSE/set_all/',
                                           transform=transforms.ToTensor())

val_loader = torch.utils.data.DataLoader(
    pulse_test,
    batch_size=128, shuffle=False, pin_memory=True, drop_last=True)

model.eval()
criterion = nn.MSELoss()

criterion = criterion.cuda()
outputs = []
reference_ = []


start = time.time()
for i, (net_input, target) in enumerate(val_loader):
    net_input = net_input.cuda(non_blocking=True)

    target = target.cuda(non_blocking=True)

    # compute output
    with torch.no_grad():
        output = model(net_input)
        outputs.append(output.squeeze())

        reference_.append(target)
end = time.time()
print("processing time: ", end - start)

reference_ = torch.cat(reference_)
reference_ = reference_.tolist()
outputs = torch.cat(outputs)
outputs = (outputs - torch.mean(outputs)) / torch.std(outputs)
outputs = outputs.tolist()

fs = 30
lowcut = 0.8
highcut = 6

plt.subplots_adjust(right=0.7)
yr = butter_bandpass_filter(outputs, lowcut, highcut, fs, order=4)
yr = (yr - np.mean(yr)) / np.std(yr)
plt.plot(outputs, label='wyjście sieci')
plt.plot(reference_, '--', label='referencja\n PPG')
plt.plot(yr, label='wyjście sieci\n po filtracji')

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='large')
# plt.legend(bbox_to_anchor=(0.5, -0.20), loc='upper center', borderaxespad=0, fontsize='x-large', ncol=3)

plt.ylabel('Amplituda', fontsize='large', fontweight='semibold')
plt.xlabel('Czas [próbka]', fontsize='large', fontweight='semibold')
plt.xlim([500, 750])
plt.ylim([-2, 3])
plt.grid()

plt.show()

reference_ = (reference_ - np.min(reference_)) / (np.max(reference_) - np.min(reference_))

bpm_ref = []
bpm_out = []

win = 255
for i in range(win, len(yr), win):
    _, measures2 = hp.process(reference_[i:i + win], 30.0, bpmmax=200)
    bpm_ref.append(measures2['bpm'])
    _, mm = hp.process(yr[i:i + win], 30.0, bpmmax=200)
    bpm_out.append(mm['bpm'])

c = np.corrcoef(bpm_ref, bpm_out)
print('Correlation:', c)

#  Calculate metrics
reference_ = torch.tensor(reference_)
outputs = torch.tensor(outputs)
yr = torch.tensor(yr)

criterionMSE = nn.MSELoss()
criterionMAE = nn.L1Loss()
mse = criterionMSE(reference_, outputs)
rmse = torch.sqrt(mse)
mae = criterionMAE(reference_, outputs)
se = torch.std(outputs - reference_) / np.sqrt(outputs.shape[0])
print("MAE: ", mae, "MSE: ", mse, "RMSE: ", rmse, "SE:", se)

# Calculate and plot power spectrum
time = np.arange(0, 3, 1 / fs)
fourier_transform = np.fft.rfft(outputs)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)
frequency = np.linspace(0, fs / 2, len(power_spectrum))
plt.subplots_adjust(right=0.7)
plt.semilogy(frequency, power_spectrum, label='wyjście\n sieci')

fourier_transform = np.fft.rfft(reference_)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)

plt.semilogy(frequency, power_spectrum, label='referencja\n PPG')
plt.ylabel('|A(f)|', fontsize='large', fontweight='semibold')
plt.xlabel('Częstotliwość f [Hz]', fontsize='large', fontweight='semibold')
plt.title('Częstitliwościowe widmo mocy')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()
