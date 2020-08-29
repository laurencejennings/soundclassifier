# code based on the paper:
# 
# End-to-end environmental sound classification using a 1D convolutional neural network
# =========================
# by
# 
# Sajjad Abdoli ∗ , Patrick Cardinal, Alessandro Lameiras Koerich
# 
# reference paper: https://doi.org/10.1016/j.eswa.2019.06.040
# 
# 
# 

import os
import sys

import pandas as pd
import numpy as np
import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torchaudio
from random import seed
from random import randint
from sklearn.utils import shuffle 


base_dir = "/home/laurence/Documents/machine-learning/deepfake/soundwaves/data/"


def get_chunk_labels(n_chunk):
# numbers of dataset chunks to use:
    chunk_num = n_chunk
    data_chunk = f'dfdc_train_part_{chunk_num}'
    output_dir = os.path.join(base_dir, data_chunk, "output", data_chunk)
    audio_labels = os.path.join(output_dir, "audio_fake_real_pairs.pkl")
    unpickled_df = pd.read_pickle(audio_labels)
    return unpickled_df


# randomize the pickle and split the data:
def randomize_pickle(pickle, seed):
    np_unpickled_df = np.asarray(pickle)
    shuffled_df = shuffle(np_unpickled_df, random_state=seed)
    train, test = np.split(shuffled_df, [int(.8*len(shuffled_df))])
    return train, test


def append_to_dict(elements_dict, pair, chunk):
    elements_dict.update({pair[0].strip():{'chunk':chunk, 'label': 0}}) # 0 = fake
    elements_dict.update({pair[1].strip():{'chunk':chunk, 'label': 1}}) # 1 = real

def append_to_dict_v2(elements_dict, name, label, chunk):
    elements_dict.update({name.strip():{'chunk':chunk, 'label': label}}) 


def get_chunks(chunk_list):
    seed = randint(1, 1000)
    train_lables = {}
    test_lables = {}
    for chunk in chunk_list:
        pickle = get_chunk_labels(chunk)
        print(f'adding chunk #{chunk} of length: {len(pickle)}')
        randomized_train, randomized_test = randomize_pickle(pickle, seed)
        for pair in randomized_train:
            append_to_dict(train_lables, pair, chunk)
        for pair in randomized_test:
            append_to_dict(test_lables, pair, chunk)
    return train_lables, test_lables

def get_chunks_v2(chunk_list):
    seed = randint(1, 1000)
    train_lables = {}
    test_lables = {}
    for chunk in chunk_list:
        pickle = get_chunk_labels(chunk)
        print(f'adding chunk #{chunk} of length: {len(pickle)}')
        randomized_train, randomized_test = randomize_pickle(pickle, seed)
        for video in randomized_train:
            append_to_dict_v2(train_lables, video[0], video[1], chunk)
        for pair in randomized_test:
            append_to_dict_v2(test_lables, video[0], video[1], chunk)
    return train_lables, test_lables
    
def get_audio_chunk_dir(chunk_num):
    data_chunk = f'audio_dfdc_train_part_{chunk_num}'
    chunk_dir = os.path.join(base_dir, data_chunk)
    return chunk_dir

def get_torchaudio_file(filename):
    audio_file = os.path.join(get_chunk_dir(randomized_train[filename].get('chunk')), filename + ".wav")
    waveform, sample_rate = torchaudio.load(audio_file)
    return audio_file, waveform, sample_rate

def get_chunk_dir(chunk_num):
    data_chunk = f'audio_dfdc_train_part_{chunk_num}'
    chunk_dir = os.path.join(base_dir, data_chunk)
    return chunk_dir

def get_file(dataset, filename):
    audio_file = os.path.join(get_chunk_dir(dataset[filename].get('chunk')), filename + ".wav")
    waveform, sample_rate = torchaudio.load(audio_file)
    return audio_file, waveform, sample_rate

# Let’s check if a CUDA GPU is available and select our device. Running
# the network on a GPU will greatly decrease the training/testing runtime.
# 
# 
# 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('the device being used is: ', device)

# Creating the data variable
# --------------------------
# the dataset variables randomized_train and randomized_test contain pre randomized split datasets from the total number of files used in the dataset. The function get_chunks gets a list of numbers which are the zip file numbers found in the original datasets of dfdc. The function get_chunks outputs the number of audio pairs in the chunk. 
# 
# ### TODO, should use also the videos with real audio files to train on more data. 

# randomized_train, randomized_test = get_chunks([0, 2, 23, 40, 41, 42, 43, 45, 46, 47, 48, 49])
randomized_train, randomized_test = get_chunks_v2([49])


# for video in randomized_train:
#     print(video, randomized_train[video])
#     break

# ### the labels:
# 
# 0 fake
# 
# 1 real

# for video in randomized_train:
#     print('videoname: ', video) 
#     print('chunk: ', randomized_train[video].get('chunk'))
#     print('label: ', randomized_train[video].get('label'))
#     break

# Play the audio files using ipd:
# -------------------------------
# 
# the path to the file is given by: 
# for filename in randomized_train:
#     audio_file, waveform, sample_rate = get_file(filename)

# import IPython.display as ipd
# for filename in randomized_train:
#     audio_file, waveform, sample_rate = get_file(randomized_train, filename)
#     ipd.Audio(audio_file)
#     break

# sample rate of the dataset
# --------------------------
# 
# find the number of occurences of a specific sample rate in the dataset

# sample_rates = {}
# known_sr = []
# for filename in randomized_train:
#     audio_file, waveform, sample_rate = get_file(randomized_train, filename)
#     if sample_rate in known_sr:
#         sample_rates[sample_rate] = sample_rates[sample_rate] + 1
#     else:
#         known_sr.append(sample_rate)
#         sample_rates[sample_rate] = 1
# print(known_sr)
# print(sample_rates)

# print(known_sr)
# print(max(known_sr))

# Channels of the dataset:
# ------------------------
# 
# same as above but for the channels of the audio (are there stereo waves?)
# and for the number of samples in the file with a verification for the audio file's length

channels = {}
known_chans = []
samples_l = {}
known_sl = []
durations_n = {}
known_dn = []
sample_rates = {}
known_sr = []
for filename in randomized_train:
    audio_file, waveform, sample_rate = get_file(randomized_train, filename)
    
    if sample_rate in known_sr:
        sample_rates[sample_rate] = sample_rates[sample_rate] + 1
    else:
        known_sr.append(sample_rate)
        sample_rates[sample_rate] = 1

    if waveform.shape[0] in known_chans:
        channels[waveform.shape[0]] = channels[waveform.shape[0]] + 1
    else:
        known_chans.append(waveform.shape[0])
        channels[waveform.shape[0]] = 1

    if waveform.shape[1] in known_sl:
        samples_l[waveform.shape[1]] = samples_l[waveform.shape[1]] + 1
    else:
        known_sl.append(waveform.shape[1])
        samples_l[waveform.shape[1]] = 1

    if waveform.shape[1]/sample_rate in known_dn:
        durations_n[waveform.shape[1]/sample_rate] = durations_n[waveform.shape[1]/sample_rate] + 1
    else:
        known_dn.append(waveform.shape[1]/sample_rate)
        durations_n[waveform.shape[1]/sample_rate] = 1

print("present sample rates in the dataset:")
print(known_sr)
print("sample rates occurrence:")
print(sample_rates)
print("max sample rate found:")
print(max(known_sr))

print("present sample numbers in the dataset:")
print(known_sl)
print("sample numbers occurrence:")
print(samples_l, '\n')

print('time durations present in the dataset:')
print(known_dn)
print("time durations occurrence:")
print(durations_n, '\n')

print('present channels in the dataset:')
print(known_chans)
print('channels occurrence:')
print(channels, '\n')

# Windowing function
# -------------------
# This function is supposed to take as input a tensor (the audio file) and output a list of tensors each representing a window of the original input and each overlapping of a certain degree. The overlapping percentage of window width and the number of data points per window are parameters of this function 

def create_windowed_tensor(input, window_size, labels):
    # input: the input tensor 
    windows_num = math.ceil(input.shape[2]/window_size)
    stacked_partials = []
    stretched_labels = []
    label_index = 0
    for audio in input:
        audio_windows = []
        for window_n in range(windows_num):
            window_tensor = audio[:,window_n*window_size:(window_n+1)*window_size]
            if window_tensor.shape[1] < window_size:
                temp = torch.zeros(audio.shape[0], window_size)
                temp[:,:window_tensor.shape[1]] = window_tensor
                window_tensor = temp

            audio_windows.append(window_tensor)
        stacked_partials.append(torch.stack(audio_windows))
        # print(labels, len(labels))
        for i in range(len(audio_windows)):
            # print(labels[label_index])
            stretched_labels.append(labels[label_index])
        label_index += 1
    stacked_total = torch.cat(stacked_partials)
    stretched_labels = torch.stack(stretched_labels)
    # print(stacked_total.shape, stretched_labels.shape)
    return stacked_total, stretched_labels

# -------------------
# 
# Now that we know the format of the dict file entries, we can construct
# our dataset. We will create a rapper class for our dataset using
# ``torch.utils.data.Dataset`` that will handle loading the files and
# performing some formatting steps.
# 
# The rapper class will store the file names, labels, and folder numbers of the audio
# files in the inputted folder list when initialized. The actual loading
# and formatting steps will happen in the access function ``__getitem__``.
# 
# In ``__getitem__``, we use ``torchaudio.load()`` to convert the wav
# Formatting the Data
# files to tensors. ``torchaudio.load()`` returns a tuple containing the
# newly created tensor along with the sampling frequency of the audio file
# (there are many different sampling frequencies in the dataset therefore they must be variables in the getitem function where the audio is manipulated). 
# If the audio file is stereo a torch.mean operation is necessary to reduce the number of channels to 1. 
# 
# Next, we need to format the audio data. The network
# we will make takes an input size of 32,000, while most of the audio
# files have well over 400,000 samples. 
# 
# There are two possible options: split the audio in chunks of a second each (each file becomes ten different subfiles), downsample the audio sufficiently to make it fit in a reasonable input size. By
# downsampling the audio to aproximately 8kHz, we can represent 4 seconds
# with the 32,000 samples. This downsampling is achieved by taking every
# n-th sample of the original audio tensor. Not every audio tensor is
# long enough to handle the downsampling so these tensors will need to be
# padded with zeros. The minimum length that won’t require padding is
# 160,000 samples.
# 
# 
# 

class DeepfakeVoiceDataset(Dataset):
#rapper for the DeepfakeVoice dataset
    # Argument List
    #  dictionary dataset
    #  window size int
    #  window_overlap float
    
    def __init__(self, dataset):
        self.file_names = []
        self.labels = []
        self.folders = []
        for video in dataset:
            self.file_names.append(video + ".wav")
            self.labels.append(dataset[video].get('label'))
            self.folders.append(dataset[video].get('chunk'))
                
        self.mixer = torch.mean #UrbanSound8K uses two channels, this will convert them to one
    def __getitem__(self, index):
        #format the file path and load the file
        path = os.path.join(get_chunk_dir(self.folders[index]), self.file_names[index])

        # normalization True is equal to normalization = 32 and it scales each datapoint from a 2**32
        # order of magnitude number to a 2**0 order of magnitude one. This can be changed to
        # normalization = 16 or normalization = Function
        waveform, sr = torchaudio.load(path, out = None, normalization = True, channels_first=False)
        waveform = waveform.permute(1,0)
        # make all inputs the same size based on the longest sample length 
        tempData = torch.zeros([1, max(known_sl)]) 
        tempData[:,:waveform.numel()] = waveform[:]

        # if waveform.numel() < max(known_sl):
        #     tempData[:,:waveform.numel()] = waveform[:]
        # else:
        #     tempData[:] = waveform[:max(known_sl)]
        
        waveform = tempData
        return waveform, self.labels[index]
 
    
    def __len__(self):
        return len(self.file_names)
     
    
train_set = DeepfakeVoiceDataset(randomized_train) 
test_set = DeepfakeVoiceDataset(randomized_test)
print("Train set size: " + str(len(train_set)))
print("Test set size: " + str(len(test_set)))

kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {} #needed for using datasets on gpu

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 4, shuffle = True, **kwargs) #changed from 64
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 4, shuffle = True, **kwargs) #changed from 64

# Define the Network
# ------------------
# 
# For this tutorial we will use a convolutional neural network to process
# the raw audio data. Usually more advanced transforms are applied to the
# audio data, however CNNs can be used to accurately process the raw data.
# The specific architecture is modeled after the M5 network architecture
# described in https://arxiv.org/pdf/1610.00087.pdf. An important aspect
# of models processing raw audio data is the receptive field of their
# first layer’s filters. Our model’s first filter is length 80 so when
# processing audio sampled at 8kHz the receptive field is around 10ms.
# This size is similar to speech processing applications that often use
# receptive fields ranging from 20ms to 40ms.
# 
# 
# 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 64, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(8, stride=8)
        self.conv2 = nn.Conv1d(16, 32, 32, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(8, stride=8)
        self.conv3 = nn.Conv1d(32, 64, 16, stride=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 8, stride=2)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 256, 4, stride=2)
        self.bn5 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4, stride=4)
        self.dropout = nn.Dropout(p=0.25)
        self.input_linear = 256*2
        self.fc1 = nn.Linear(self.input_linear, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        
    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = F.relu(self.bn1(x))
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = F.relu(self.bn2(x))
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = F.relu(self.bn3(x))
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = F.relu(self.bn4(x))
        # print(x.shape)
        x = self.conv5(x)
        # print(x.shape)
        x = F.relu(self.bn5(x))
        # print(x.shape)
        x = self.pool3(x)
        # print(x.shape)
        x = x.view(-1, self.input_linear)
        x = self.dropout(self.fc1(x)) #apply dropout on the fc layer
        x = self.dropout(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

model = Net()
params = model.parameters()
model.to(device)
print(model)

# We will use the same optimization technique used in the paper, an Adam
# optimizer with weight decay set to 0.0001. At first, we will train with
# a learning rate of 0.01, but we will use a ``scheduler`` to decrease it
# to 0.001 during training.
# 
# 
# 
lr=0.003
#The Adadelta ( Zeiler, 2012 ) optimizer with the default learning rate of 1.0 was used. Adadelta has been chosen because this method dynamically adapts the learning rate during the optimization process.
optimizer = torch.optim.Adadelta(params, lr=lr, rho=0.9, eps=1e-06, weight_decay=0) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)

# Training and Testing the Network
# --------------------------------
# 
# Now let’s define a training function that will feed our training data
# into the model and perform the backward pass and optimization steps.
# 
# 
# 
# model = Net()

# define loss function
criterion = nn.CrossEntropyLoss()

# define parameters of training:
epochs = 80
steps = 0
running_loss = 0
print_every = 30
train_losses, test_losses = [], []
trained_on = 0

# loop over epoch range
for epoch in range(epochs):
    # perform scheduler step to reduce learning rate
    scheduler.step()
    print('Epoch {}, lr {}'.format(epoch, optimizer.param_groups[0]['lr']))
    trained_on = 0

    # loop over all inputs in train loader
    for inputs, labels in train_loader:
        # print inputs size from the train loader
        # print(f'inputs from train_loader: {inputs.shape}')
        # convert train loader inputs into window of inputs
        inputs, labels = create_windowed_tensor(inputs, 32000, labels)
        # print inputs size from the windows
        # print(f'inputs from windows: {inputs.shape}')
        # adjust trained on to consider all windows for data augmentation
        trained_on += inputs.shape[0]
        # using the longest sample length, divided by the window size, determine the num of windows
        total_windows = math.ceil(max(known_sl)/32000)
        # calculate percentage of training samples used
        trained_pct = math.floor(trained_on*100/(len(train_set)*total_windows))
        
        # increase step counter used in printing calculation
        steps += 1
        # pass inputs and labels (windowed) to device
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the gradients before backprop
        optimizer.zero_grad()
        # calculate predictions using model and windowed inputs
        predictions = model.forward(inputs)
        # calculate loss using predictions
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        # add the loss of this batch to the running loss
        running_loss += loss.item()
        # print(f'running loss: {running_loss}')
        
        # if the step is one in which printing is necessary load the test data and run a test round.
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            # switch the model to eval mode: batchnorm or dropout layers will work in eval mode instead of training mode
            model.eval()
            # use no grad as the test items shouldn't have impact on the training and gradient
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = create_windowed_tensor(inputs, 32000, labels)
                    inputs, labels = inputs.to(device),labels.to(device)
                    predictions = model.forward(inputs)
                    batch_loss = criterion(predictions, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(predictions)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/print_every)
            test_losses.append(test_loss/len(test_loader))  
            print(f"Epoch {epoch+1}/{epochs}.. tested {trained_pct}%/100% "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(test_loader):.3f}.. "
                  f"Test accuracy: {accuracy/len(test_loader):.3f}")
            running_loss = 0
            model.train()
    print(f'train losses: {train_losses}')
    print(f'epoch: {epoch} finished. Saving')
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f'./state_lr{lr}.tar')




