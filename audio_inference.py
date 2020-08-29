import os
import sys
import subprocess

import pandas as pd
import numpy as np
import math
import functools
import glob 
import re 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

import pickle

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print('the device being used is: ', device)

saved_model_path = './state_lr-0.01_2020-03-31-17-54-18.tar'
test_audio_dir = '/home/laurence/Documents/machine-learning/deepfake/soundwaves/data/audio_dfdc_train_part_49/'
output_dir = './output/'
input_dir = './input/'


test_labels = ['/media/laurence/black/deepfake_challenge/data/dfdc_train_part_42/output/uniques_pickle.pkl', '/media/laurence/black/deepfake_challenge/data/dfdc_train_part_43/output/uniques_pickle.pkl', '/media/laurence/black/deepfake_challenge/data/dfdc_train_part_44/output/uniques_pickle.pkl']


def append_to_dict_v2(elements_dict, name, label, chunk):
    elements_dict.update({name.strip():{'chunk':chunk, 'label': label}}) 

def get_chunks_v2(chunk_list):
    test_lables = {}
    for chunk in chunk_list:
        print(chunk)
        pickle = pd.read_pickle(chunk)
        print(f'adding chunk #{chunk} of length: {len(pickle)}')
        for video in pickle:
            append_to_dict_v2(test_lables, video[0], int(video[1]), chunk)
    return test_lables


range_zips = range(42,45)
original_test_set = get_chunks_v2(test_labels)
for video in original_test_set:
    print(video)



def create_windowed_tensor(input, window_size):
    input = input.view([1,input.shape[0],input.shape[1]])
    windows_num = math.ceil(input.shape[2]/window_size)
    stacked_partials = []
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
    stacked_total = torch.cat(stacked_partials)
    return stacked_total

def feed_input(video):
    audio_path = video
    waveform, sr = torchaudio.load(audio_path, out = None, normalization = True, channels_first=False)
    waveform = waveform.permute(1,0)
    windowed_waveform = create_windowed_tensor(waveform, 32000)
    return windowed_waveform

def extract_audio(video_name):
    video_path =  os.path.join(input_dir, video_name)
    output_wav_path = os.path.join(output_dir, video_name[:-4] + ".wav")
    probe_real = f"ffprobe -show_streams -print_format json {video_path} | grep -o 'Audio'"
    cmd_real = subprocess.run(probe_real,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)

    if "Audio" in cmd_real.stdout.decode('utf-8'):
        if not os.path.exists(output_wav_path):
            print("extracting audio from video")
            subprocess.call(['ffmpeg', '-i', video_path, '-max_muxing_queue_size', '9999',output_wav_path])
    return output_wav_path

def windows_prediction_aggregator(window_predictions):
    final_prediction = 0
    # print('raw',window_predictions)
    # print('mean',torch.mean(window_predictions))
    # print('max ', torch.max(window_predictions))
    # print(window_predictions.shape)
    # window_predictions = window_predictions.view(window_predictions.shape[0])
    # print(window_predictions.shape)
    # topvals = torch.topk(window_predictions, 4)
    # print('mean ', torch.mean(topvals[0]))
    return torch.max(window_predictions)
    for window_pred in window_predictions:
        window_pred = torch.argmax(window_pred)
        if window_pred == 1:
            final_prediction = 1
    return final_prediction


def predict(video_name):
    # create the audio file and assign the path to a var
    audio_path = extract_audio(video_name)
    # feed the audio file as a tensor and split it in windows
    data = feed_input(audio_path)
    data = data.to(device)
    # determine the predictions over each window
    window_output = model(data)
    # aggregate the predictions on the windows: if any window is predicted to be fake the whole video is considered fake
    prediction = windows_prediction_aggregator(window_output)
    print(f'The video {video_name} is predicted to be {"real" if prediction == 0 else "fake"}')

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
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.pool3(x)
        x = x.view(-1, self.input_linear)
        x = self.fc1(x) #apply dropout on the fc layer
        x = self.fc2(x)
        x = self.fc3(x)
#         print(f'sigmoid {torch.sigmoid(x)}')
        return torch.sigmoid(x)

model = Net()
params = model.parameters()
model.to(device)
print(model)

checkpoint = torch.load(saved_model_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

def predict_test(audio_file):
    # create the audio file and assign the path to a var
#     audio_path = extract_audio(video_name)
    # feed the audio file as a tensor and split it in windows
    sr = re.compile('.+\/(\w+)')

    filename = sr.match(audio_file).group(1)
    ground_truth = original_test_set[filename].get('label')
    data = feed_input(audio_file)
    data = data.to(device)
    # determine the predictions over each window
    window_output = model(data)
    # aggregate the predictions on the windows: if any window is predicted to be fake the whole video is considered fake
    prediction = windows_prediction_aggregator(window_output)
    if ground_truth == 1:
        print(f'the ground truth for file {filename} is {ground_truth} and the prediction is {prediction.item()}')
    return ground_truth, prediction

test_data_dirs = ['/home/laurence/Documents/machine-learning/deepfake/soundwaves/data/audio_dfdc_train_part_42/', '/home/laurence/Documents/machine-learning/deepfake/soundwaves/data/audio_dfdc_train_part_43/', '/home/laurence/Documents/machine-learning/deepfake/soundwaves/data/audio_dfdc_train_part_44/']
ground_truths = []
predictions = []
for test_dir in test_data_dirs:
    list_of_files =  glob.glob(f'{test_dir}/*.wav')
    # print(test_dir, len(list_of_files))
    for file in list_of_files:
        # print(file)
        file_gt, file_pred = predict_test(file)
        ground_truths.append(file_gt)
        predictions.append(file_pred.item())

with open('./groundtruths.pkl', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(ground_truths, filehandle)
with open('./predictions.pkl', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(predictions, filehandle)
# # to run the function from command line
# if __name__ == '__main__':
#     globals()[sys.argv[1]](sys.argv[2])