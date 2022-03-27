# DeepPulse
## PyTorch implementation of  pulse measurement neural networks. 

Python scripts to train and test different PPG calculating neural networks. 
Network architectures used in research: 
* PhysNet 3D (https://github.com/ZitongYu/PhysNet)
* SpatioTemporal PhysNet (https://arxiv.org/pdf/1905.02419v1.pdf)
* HR-CNN (https://github.com/radimspetlik/hr-cnn)
* Global Context Block (https://arxiv.org/pdf/1904.11492.pdf)
* Dilated convolution 

### Usage 
To train or test network use appropriate script from listed below. 
PyTorch Dataloader and Dataset are prepared to work with sets containing `waveform` information from pulseoximeter. 
Scripts require lists of sequences to train and test the network. Sequences should be placed in `<dataset>\<sequence_num>`
directory, where `<dataset>` should be set as `root_dir = <dataset>` in training and testing scripts. 

Each image of the sequence should be cropped to only subjects face using `utils\face_crop.py`

#### HR-CNN 
training script `hr_cnn_train.py` \
validation script `hr_cnn_valid.py`\
network implementation and global context block `hr_cnn.py`\
network implementation using dilated convolution `hr_cnn_dil.py`

#### PhysNet 3D 
training script `PhysNet_train.py` \
validation script `PhysNet_train.py`\
base network implementation `PhysNet.py`\
network with GCBlock `PhysNetGlobal.py`\
network implementation using dilated convolution `PhysNetDil.py`

#### Spatio-temporal PhysNet 
training script `PhysNet_train.py` \
validation script `PhysNet_test.py`\
network implementation `PhysNet_SpaTemp.py`

#### GradCam 
Script `grad_cam.py` contains function to visualise activation maps using GradCam method adapted to 3D convolution. 

### Data 
Datasets used are acquired from:
* PURE (https://www.tu-ilmenau.de/en/neurob/data-sets-code/pulse/)
* VIPL-HR (https://vipl.ict.ac.cn/view_database.php?id=15)
* PFF (https://ieeexplore.ieee.org/document/8272721)

#### Requirements 
* scipy
* torch
* opencv-python
* torchvision
* numpy
* pandas


