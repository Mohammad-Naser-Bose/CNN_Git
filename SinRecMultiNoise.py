################################### Libraries
import librosa
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import DataLoader, TensorDataset, Dataset
import IPython.display as ipd
import sounddevice as sd
import scipy.signal as signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math

################################### Inputs
recordings_dir = r"C:\Users\mn1059928\OneDrive - Bose Corporation\Desktop\audio_files_short"
noise_dir = r"C:\Users\mn1059928\OneDrive - Bose Corporation\Desktop\noise_files"
window_size_sec = 4  # in [s]
sampling_freq = 44100  # in [Hz]  
num_epochs=150
train_ratio = .9
val_ratio = .05
downsampling_new_sr = 690   #Ratio=64,128 = 690,344
batch_size = 1
use_filter=False
filter_num_coeff = [1]
filter_dem_coeff = [1, 1]
normalization_flag = True
noise_gains = [0] # dB
ML_type = "CNN"
norm_feature =True
sought_ratio = [2, 5, 10]         # data to noise          #audio_gains = [-10,-50,-90] #[i for i in np.arange(-100,-210,-10)]  
window_len_sample = window_size_sec * sampling_freq
window_len_sample_downsampled = window_size_sec * downsampling_new_sr
noise_files = os.listdir(noise_dir); num_noise_combinations=sum(os.path.isfile(os.path.join(noise_dir,f )) for f in noise_files)
################################### Main
def loading_data(dir,label):
    if label=="noise":
        files = os.listdir(dir) 
        full_recordings = {}
        for i,file in enumerate(files):
            audio_path=os.path.join(dir,file)
            audio_data, sample_rate = librosa.load(audio_path,sr=None)
            full_recordings[i] = audio_data[:window_len_sample]
    else:
        # will need to limit it to a set number or recrdings
        files = os.listdir(dir) 
        full_recordings = {}
        for i,file in enumerate(files):
            audio_path=os.path.join(dir,file)
            audio_data, sample_rate = librosa.load(audio_path,sr=None)
            full_recordings[i] = audio_data     
    return full_recordings
def resampling(orig_data,label):
    full_resampled_data={}
    for i in range(0, len(orig_data)):
        downsampled_data_temp = librosa.resample(orig_data[i],orig_sr=sampling_freq,target_sr=downsampling_new_sr)
        full_resampled_data[i] = downsampled_data_temp
        
        # sd.play(downsampled_data_temp, samplerate=44100)
        # sd.wait()
    return full_resampled_data
def modifying_noise(Data):
    modified_data={}
    for i in range(0, len(Data)):
        if i==0:
            modified_data[i] = Data[i] * 1
        else:
            modified_data[i] = Data[i]
    return modified_data
def adding_gain_audio(data,audio_gains):
    desired_gains_linear = []
    for gain in audio_gains:
        desired_gains_linear.append(10 ** (gain / 20))
    
    adjusted_rec = {}
    master_c = 0
    for gain in desired_gains_linear:
        for i in range (0, len(data)):
            adjusted_single_rec = data[i] * gain
            adjusted_rec[master_c] = adjusted_single_rec
            master_c+=1
    return adjusted_rec
def adding_gain_noise(data,gain):
    desired_gains_linear = []
    for gain in noise_gains:
        desired_gains_linear.append(10 ** (gain / 20))
    
    adjusted_rec = {}
    master_c = 0
    for gain in desired_gains_linear:
        for i in range (0, len(data)):
            adjusted_single_rec = data[i] * gain 
            adjusted_rec[master_c] = adjusted_single_rec
            master_c+=1
    return adjusted_rec
def transfer_fun(Data):
    if use_filter == False:
        return Data
    else:
        audio_transformed = {}
        system = signal.TransferFunction(filter_num_coeff, filter_dem_coeff).to_discrete(1/downsampling_new_sr)
        for i in range (0, len(Data)):
            _,audio_transformed_single_rec = signal.dlsim(system,Data[i])
            audio_transformed[i] = audio_transformed_single_rec
        return audio_transformed
def duplicating_recordings(audio, noise):
    duplicated_audio = {}
    duplicated_noise = {}
    master_c = 0
    for ii in range (0, num_noise_combinations):
        for i in range (0, len(audio)):
            duplicated_audio[master_c]=audio[i]
            duplicated_noise[master_c]=noise[ii]
            master_c+=1
    return duplicated_audio, duplicated_noise
def duplicating_recordings_for_gain(audio,audio_gains):
    duplicated_audio = {}
    master_c = 0
    for i in range (0, len(audio)):
        for gain in audio_gains:
            duplicated_audio[master_c]=audio[i]
            master_c+=1
    return duplicated_audio
def concatenating_noise(audio, noise):
    noise_data_full_length_dic = {}
    for i in range (0, len(audio)):
        repeat_factor = int(np.ceil(len(audio[i]) / len(noise[i])))
        noise_data_full_length = np.tile(noise[i], repeat_factor)[:len(audio[i])]
        noise_data_full_length_dic [i] = noise_data_full_length
    return noise_data_full_length_dic
def mixing(audio, noise):
    mixed_signal_sin_rec = {}
    for i in range (0, len(audio)):
        mixed_signal_sin_rec[i] = audio[i] + noise[i]
    return mixed_signal_sin_rec
def windowing(signal):
    master_c = 0
    windowed_data = {}
    for i, rec in enumerate (signal.items()):
        my_rec = rec[1]
        num_windows = len(my_rec) // window_len_sample_downsampled
        extra_samples = len(my_rec) % (num_windows*window_len_sample_downsampled)
        truncated_rec = my_rec[extra_samples:]
        rec_windows = np.array_split(truncated_rec, num_windows)
        rec_windows_truncated = [window[:window_len_sample_downsampled] for window in rec_windows]
        for arr in rec_windows_truncated:
            windowed_data[master_c] = arr.tolist()
            master_c+=1
    return windowed_data
def normalization (train_no_norm, val_no_norm, test_no_norm):

    all_windows_training = np.concatenate ([np.array(value) for value in train_no_norm.values()])
    scaler = StandardScaler()
    scaler.fit(all_windows_training.reshape(-1,1))
    normalized_training_windows = {key:scaler.transform(np.array(value).reshape(-1,1)).flatten().tolist() for key, value in train_no_norm.items()}
    normalized_validation_windows = {key:scaler.transform(np.array(value).reshape(-1,1)).flatten().tolist() for key, value in val_no_norm.items()}
    normalized_testing_windows = {key:scaler.transform(np.array(value).reshape(-1,1)).flatten().tolist() for key, value in test_no_norm.items()}

    return normalized_training_windows, normalized_validation_windows, normalized_testing_windows
def find_RMS_data(data):
    RMS_values = {}
    for i, recording in enumerate (data.items()):
        my_data = recording[1]
        RMS_values [i] =  librosa.feature.rms(y=np.array(my_data)).mean()
    
    RMS_values_new = {}
    for key, value in RMS_values.items():
        RMS_values_new[key] = np.array(value, dtype= np.float32)
    return RMS_values_new
def find_RMS_noise(data):
    RMS_values = {}
    for i, recording in enumerate (data.items()):
        my_data = recording[1]
        RMS_values [i] =  librosa.feature.rms(y=np.array(my_data)).mean()
    
    RMS_values_new = {}
    for key, value in RMS_values.items():
        RMS_values_new[key] = np.array(value, dtype= np.float32)
    return RMS_values_new
def find_RMS_noise_with_norm(data_1, data_2, data_3):
    if norm_feature==True:
        RMS_values_1 = {}
        for i, recording in enumerate (data_1.items()):
            my_data = recording[1]
            RMS_values_1 [i] = np.sqrt(np.mean((np.array(my_data)**2)))
        temp_arr = np.hstack(list(RMS_values_1.values())).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler.fit(temp_arr)
        temp_arr_new = scaler.transform(temp_arr)
        RMS_values_new_1 = {}
        for i in range(0,len(temp_arr_new)):
            RMS_values_new_1[i] = np.array(temp_arr_new[i], dtype= np.float32)

        RMS_values_2 = {}
        for i, recording in enumerate (data_2.items()):
            my_data = recording[1]
            RMS_values_2 [i] = np.sqrt(np.mean((np.array(my_data)**2)))
        temp_arr = np.hstack(list(RMS_values_2.values())).reshape(-1, 1)
        temp_arr_new =scaler.transform(temp_arr)
        RMS_values_new_2 = {}
        for i in range(0,len(temp_arr_new)):
            RMS_values_new_2[i] = np.array(temp_arr_new[i], dtype= np.float32)

        RMS_values_3 = {}
        for i, recording in enumerate (data_3.items()):
            my_data = recording[1]
            RMS_values_3 [i] = np.sqrt(np.mean((np.array(my_data)**2)))
        temp_arr = np.hstack(list(RMS_values_3.values())).reshape(-1, 1)
        temp_arr_new =scaler.transform(temp_arr)
        RMS_values_new_3 = {}
        for i in range(0,len(temp_arr_new)):
            RMS_values_new_3[i] = np.array(temp_arr_new[i], dtype= np.float32)
    else:
        RMS_values_1 = {}
        for i, recording in enumerate (data_1.items()):
            my_data = recording[1]
            RMS_values_1 [i] = np.sqrt(np.mean((np.array(my_data)**2)))
        RMS_values_new_1 = {}
        for key, value in RMS_values_1.items():
            RMS_values_new_1[key] = np.array(value, dtype= np.float32)

        RMS_values_2 = {}
        for i, recording in enumerate (data_2.items()):
            my_data = recording[1]
            RMS_values_2 [i] = np.sqrt(np.mean((np.array(my_data)**2)))
        RMS_values_new_2 = {}
        for key, value in RMS_values_2.items():
            RMS_values_new_2[key] = np.array(value, dtype= np.float32)

        RMS_values_3 = {}
        for i, recording in enumerate (data_3.items()):
            my_data = recording[1]
            RMS_values_3 [i] = np.sqrt(np.mean((np.array(my_data)**2)))
        RMS_values_new_3 = {}
        for key, value in RMS_values_3.items():
            RMS_values_new_3[key] = np.array(value, dtype= np.float32)

    return RMS_values_new_1, RMS_values_new_2, RMS_values_new_3
def data_prep_for_ML(channel1, channel2):
    keys = sorted(channel1.keys())
    data1_tensors = [torch.tensor(channel1[key]) for key in keys]
    data2_tensors = [torch.tensor(channel2[key]) for key in keys]
    
    data1_batch = torch.stack(data1_tensors)
    data2_batch = torch.stack(data2_tensors)

    data1_batch = data1_batch.unsqueeze(1)
    data2_batch = data2_batch.unsqueeze(1)

    combined_data = torch.cat((data1_batch,data2_batch), dim=1)

    return combined_data
def data_splitting(x, y, z):
    keys = list(x.keys())
    random.shuffle(keys)
    train_end = int(train_ratio * len(keys))
    val_end = train_end + int(val_ratio * len(keys))

    train_keys = keys[:train_end]
    val_keys = keys[train_end:val_end]
    test_keys = keys[val_end:]

    train_x = {key: x[key] for key in train_keys}
    val_x = {key: x[key] for key in val_keys}
    test_x = {key: x[key] for key in test_keys}

    train_y = {key: y[key] for key in train_keys}
    val_y = {key: y[key] for key in val_keys}
    test_y = {key: y[key] for key in test_keys}

    train_z = {key: z[key] for key in train_keys}
    val_z= {key: z[key] for key in val_keys}
    test_z= {key: z[key] for key in test_keys}

    return train_x, val_x, test_x, train_y, val_y, test_y, train_z, val_z, test_z, train_keys, val_keys, test_keys

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, stride=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=3, padding=1, dilation=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=3, padding=1, dilation=1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.flattened_size= self._get_flattened_size()
        self.fc1 = nn.Linear(self.flattened_size,128)
        # self.fc2 = nn.Linear(4096,1024)
        # self.fc3= nn.Linear(1024,256)
        self.fc4= nn.Linear(128,32)
        self.fc5= nn.Linear(32,1)

    def _get_flattened_size(self):
        x = torch.zeros(1,2,window_len_sample_downsampled) # one sample regardless the batch size, num channels, num timepoints
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return x.numel()

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x_dim = x.dim()
        if x_dim==3:
            x=x.view(x.size(0), -1)
        else:
            x=x.view(-1)
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(13800,10240)
        self.fc2 = nn.Linear(10240,7168)
        self.fc3 = nn.Linear(7168,3072)
        self.fc4 = nn.Linear(3072,1024)
        self.fc5 = nn.Linear(1024,256)
        self.fc6 = nn.Linear(256,64)
        self.fc7 = nn.Linear(64,1)



    def forward(self,x):
        batch_size, dim1, dim2 = x.size()
        x = x.view(batch_size,dim1 * dim2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)


        return x
class CNN_LSTM_b(nn.Module):
    def __init__(self):
        super(CNN_LSTM_b, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, stride=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=3, padding=1, dilation=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=3, padding=1, dilation=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.flattened_size= self._get_flattened_size()
        self.lstm = nn.LSTM(input_size=self.flattened_size, hidden_size=512, num_layers=1)
        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,64)        
        self.fc3 = nn.Linear(64,1)
        
    def _get_flattened_size(self):
        x = torch.zeros(1,2,window_len_sample_downsampled) # one sample regardless the batch size, num channels, num timepoints
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return x.numel()

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x_dim = x.dim()
        if x_dim==3:
            x=x.view(x.size(0), -1)
        else:
            x=x.view(-1)

        x,_ = self.lstm(x)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))  
        x = self.fc3(x)
        return x
class CNN_LSTM_a(nn.Module):
    def __init__(self):
        super(CNN_LSTM_a, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, stride=5, padding=1, dilation=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU(negative_slope=0.01)
        self.flattened_size= self._get_flattened_size()
        self.fc1 = nn.Linear(self.flattened_size,4096)
        self.lstm = nn.LSTM(input_size=4096, hidden_size=4096, num_layers=2)
        self.fc2= nn.Linear(4096,1024)
        self.fc3= nn.Linear(1024,128)
        self.fc4= nn.Linear(128,1)
        
    def _get_flattened_size(self):
        x = torch.zeros(1,2,window_len_sample_downsampled) # one sample regardless the batch size, num channels, num timepoints
        x = self.pool(self.relu(self.conv1(x)))
        return x.numel()

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x_dim = x.dim()
        if x_dim==3:
            x=x.view(x.size(0), -1)
        else:
            x=x.view(-1)

        x = self.relu(self.fc1(x))
        x,_ = self.lstm(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
class CNN_LSTM_c(nn.Module):
    def __init__(self):
        super(CNN_LSTM_c, self).__init__()
        self.lstm = nn.LSTM(input_size=window_len_sample_downsampled, hidden_size=500, num_layers=5)

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=2, stride=2, padding=1, dilation=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU(negative_slope=0.01)
        self.flattened_size= self._get_flattened_size()
        self.fc1 = nn.Linear(self.flattened_size,256)
        self.fc2 = nn.Linear(256,64)
        self.fc3= nn.Linear(64,1)
        
    def _get_flattened_size(self):
        x = torch.zeros(1,2,window_len_sample_downsampled) # one sample regardless the batch size, num channels, num timepoints
        x,_  = self.lstm(x)
        x = self.pool(self.relu(self.conv1(x)))
        return x.numel()

    def forward(self,x):
        x,_  = self.lstm(x)
        x = self.pool(self.relu(self.conv1(x)))

        x_dim = x.dim()
        if x_dim==3:
            x=x.view(x.size(0), -1)
        else:
            x=x.view(-1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.relu = nn.ReLU(negative_slope=0.01)
        self.fc1 = nn.Linear(74304,4096)
        self.fc2 = nn.Linear(4096,2048)
        self.fc3 = nn.Linear(2048,1024)
        self.fc4 = nn.Linear(1024,512)
        self.fc5= nn.Linear(512,256)
        self.fc6= nn.Linear(256,128)
        self.fc7= nn.Linear(128,64)
        self.fc8= nn.Linear(64,32)
        self.fc9= nn.Linear(32,16)
        self.fc10= nn.Linear(16,8)
        self.fc11= nn.Linear(8,4)
        self.fc12= nn.Linear(4,2)
        self.fc13= nn.Linear(2, 1)
        
    def forward(self,x):
        x_dim = x.dim()
        if x_dim==3:
            x=x.view(x.size(0), -1)
        else:
            x=x.view(-1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))
        x = self.relu(self.fc9(x))
        x = self.relu(self.fc10(x))
        x = self.relu(self.fc11(x))
        x = self.relu(self.fc12(x))
        x = self.fc13(x)
        return x
class CustomDataset(Dataset):
    def __init__(self,inputs,labels):
        self.inputs = inputs
        self.labels = labels
    def __len__(self):
        return(len(self.inputs))
    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        label = self.labels[idx]
        return input_data, label

def ML_training(train_inputs,train_labels, train_keys):
    dataset = CustomDataset(train_inputs,train_labels)
    dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=True)
    reg_criterion = nn.MSELoss()
    model = my_ML_model 
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",factor=0.1, patience=10)

    train_loss_values = []
    error=[]
    predictions=[]
    gt=[]
    for epoch in range(num_epochs):
        print("-----------------------------")
        print("epoch:", epoch)
        model.train()
        running_train_loss = 0
        num_train_batches = len(train_inputs)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            inputs = inputs.to(torch.float32)
            outputs = model(inputs.squeeze(1))

            loss_value = reg_criterion(outputs, targets.unsqueeze(1))
            loss_value.backward()
            optimizer.step()
            running_train_loss += loss_value.item()

            print("real:", targets.detach().cpu().numpy().flatten())
            print("pred:", outputs.detach().cpu().numpy().flatten())
            print("-------")

            

            if epoch == num_epochs-1:
                ground_truth_values = targets.detach().cpu().numpy().flatten()
                predicted_values = outputs.detach().cpu().numpy().flatten()
                error.append(((abs(ground_truth_values-predicted_values))/(ground_truth_values))*100)

                predictions.append(predicted_values)
                gt.append(ground_truth_values)

        

        avg_train_loss = running_train_loss / num_train_batches
        train_loss_values.append(avg_train_loss)
        #scheduler.step(avg_train_loss)

        #print(scheduler.get_last_lr())

    plotting_performance(train_loss_values,"Training")
    plotting_results_general_training(error,predictions,gt,"Training")
    

    return model
def ML_validating(model, val_inputs,val_labels):
    reg_criterion = nn.MSELoss()

    val_loss_values = []
    model.eval()
    running_val_loss = 0
    num_val_batches = len(val_inputs)
    errors_val=[]
    all_pred=[]
    all_gt=[]
    for i in range(0, len(val_inputs)):
        my_input = val_inputs[i].requires_grad_(True)
        ground_truth_value = torch.tensor(val_labels[i])        

        predicted_value = model(my_input)
        loss_value = reg_criterion(predicted_value,ground_truth_value)

        running_val_loss += loss_value.item()

        avg_val_loss = running_val_loss / num_val_batches
        val_loss_values.append(avg_val_loss)    

        errors_val.append(((abs(ground_truth_value-predicted_value))/(ground_truth_value))*100)
        all_pred.append(predicted_value)
        all_gt.append(ground_truth_value)
    plotting_results_general_other(errors_val,all_pred,all_gt,"Validation")

    return model
def ML_testing(model, test_inputs, test_labels):
    model.eval()
    test_loss = 0
    test_errors=[]
    num_test_batches = len(test_inputs)
    reg_criterion = nn.MSELoss()
    errors_test = []
    all_gt=[]
    all_pred=[]
    for i in range(0, len(test_inputs)):
        my_input = test_inputs[i].requires_grad_(True)
        ground_truth_value = torch.tensor(test_labels[i])        

        predicted_value = model(my_input)
        loss_value = reg_criterion(predicted_value,ground_truth_value)

        test_loss += loss_value.item()
        error = ((abs(ground_truth_value-predicted_value))/(ground_truth_value))*100
        test_errors.append(error)

        #print(f"Orig:{ground_truth_value}, Predicted: {predicted_value}")

        errors_test.append(((abs(ground_truth_value-predicted_value))/(ground_truth_value))*100)
        all_pred.append(predicted_value)
        all_gt.append(ground_truth_value)

    avg_test_loss = test_loss / num_test_batches
    #print(f"Test loss: {avg_test_loss}")

    printing_label = "Testing"
    plotting_results_general_other(errors_test,all_pred,all_gt,printing_label)
   

    return 

def plotting_performance(loss_values,title):
    plt.figure(figsize=(10,5))
    plt.plot(range(1,num_epochs+1), loss_values, marker = "o", label = "Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    #plt.show()
    plt.savefig("Training error_per_Epoch.png")
def plotting_results_general_training(error,predictions,gt,printing_label):
    error_ready = [element for array in error for element in array.tolist()]
    # plt.figure(figsize=(10,5))
    # plt.hist(error_ready,bins=100)
    # plt.xlabel("Error [%]")
    # plt.ylabel("Num of datapoints")
    # #plt.title(title)
    # #plt.show()
    # plt.savefig(f"{printing_label} histogram performance.png")

    real_ready = [element for array in gt for element in array.tolist()]
    pred_ready = [element for array in predictions for element in array.tolist()]
    diff = [a-b for a,b in zip(real_ready,pred_ready)]
    plt.figure(figsize=(10,5))
    plt.plot(real_ready,label="orig")
    plt.plot(pred_ready,label="pred")
    plt.legend()
    plt.xlabel("datapoint")
    plt.ylabel("Noise RMS")
    #plt.title(title)
    #plt.show()
    plt.savefig(f"{printing_label} raw performance.png")
def plotting_results_general_other(error,predictions,gt,printing_label):
    error_ready = [element for array in error for element in array.tolist()]
    # plt.figure(figsize=(10,5))
    # plt.hist(error_ready,bins=30)
    # plt.xlabel("Error [%]")
    # plt.ylabel("Num of datapoints")
    # #plt.title(title)
    # #plt.show()
    # plt.savefig(f"{printing_label} histogram performance.png")

    real_ready = [value.item() for value in gt]
    pred_ready = [value.item() for value in predictions]
    plt.figure(figsize=(10,5))
    plt.plot(real_ready,label="orig")
    plt.plot(pred_ready,label="pred")
    plt.legend()
    plt.xlabel("datapoint")
    plt.ylabel("Noise RMS")
    #plt.title(title)
    #plt.show()
    plt.savefig(f"{printing_label} raw performance.png")

def determine_data_gain(noise, audio):
    noise_rms_full = find_RMS_noise(noise)
    noise_rms = np.hstack(list(noise_rms_full.values()))
    mean_noise_rms = np.mean(noise_rms)
    needed_audio_rms = mean_noise_rms / sought_ratio
    current_data_rms = find_RMS_data(audio)
    current_data_rms_list = list(current_data_rms.values())
    gains_linear = needed_audio_rms/current_data_rms_list
    gains_db = []
    for gain in gains_linear:
        a = math.log10(gain)
        gains_db.append(20 * a)
    return gains_db

def run_ML():

    Data_A = loading_data(noise_dir,"noise")
    Data_AB = resampling(Data_A,"noise")
    Data_B = modifying_noise(Data_AB)
    Data_C = adding_gain_noise(Data_B, noise_gains)
    Data_D = loading_data(recordings_dir,"audio")
    Data_E = resampling(Data_D,"data")
    Data_F = transfer_fun(Data_E)
    audio_gains = determine_data_gain(Data_A, Data_D)
    Data_G = adding_gain_audio(Data_F, audio_gains)

    Data_I, Data_H = duplicating_recordings(Data_G, Data_C)
    Data_J = concatenating_noise(Data_I, Data_H)

    Data_K = mixing(Data_I, Data_J)
    Data_X= windowing(Data_K)

    Data_LD = duplicating_recordings_for_gain(Data_E,audio_gains)
    Data_L, _ = duplicating_recordings(Data_LD, Data_C)
    Data_Y = windowing(Data_L)

    _ , Data_M = duplicating_recordings(Data_G, Data_B)
    Data_N = concatenating_noise(Data_I, Data_M)
    Data_O = windowing(Data_N) 




    # Splitting and normalization
    if normalization_flag == False:
        Data_Z = find_RMS_noise(Data_O); 
        
        train_x_norm, val_x_norm, test_x_norm, train_y_norm, val_y_norm, test_y_norm, train_z_norm, val_z_norm, test_z_norm, train_keys, val_keys, test_keys = data_splitting (Data_X, Data_Y, Data_Z)
    
    else:
        train_x_no_norm, val_x_no_norm, test_x_no_norm, train_y_no_norm, val_y_no_norm, test_y_no_norm, train_o_no_norm, val_o_no_norm, test_o_no_norm, train_keys, val_keys, test_keys = data_splitting (Data_X, Data_Y, Data_O)
   
        train_x_norm, val_x_norm, test_x_norm = normalization (train_x_no_norm, val_x_no_norm, test_x_no_norm)
        train_y_norm, val_y_norm, test_y_norm = normalization (train_y_no_norm, val_y_no_norm, test_y_no_norm)
        train_o_norm, val_o_norm, test_o_norm = normalization (train_o_no_norm, val_o_no_norm, test_o_no_norm)
        
        train_z_norm, val_z_norm, test_z_norm  = find_RMS_noise_with_norm(train_o_norm, val_o_norm, test_o_norm )  
   
    data_train_xy = data_prep_for_ML(train_x_norm, train_y_norm); data_val_xy = data_prep_for_ML(val_x_norm, val_y_norm); data_test_xy = data_prep_for_ML(test_x_norm, test_y_norm)

    train_z_norm_l= list(train_z_norm.values()); val_z_norm_l = list(val_z_norm.values()); test_z_norm_l= list(test_z_norm.values())




    # ML work
    model = ML_training(data_train_xy,train_z_norm_l, train_keys)
    ML_validating(model, data_val_xy, val_z_norm_l)
    ML_testing(model, data_test_xy, test_z_norm_l)
    
    




if __name__ == "__main__":
    if ML_type == "CNN":
        my_ML_model = CNN()
    elif ML_type == "NN":
        my_ML_model = NN()
    elif ML_type == "CNN_LSTM_a":
        my_ML_model = CNN_LSTM_a()      
    elif ML_type == "CNN_LSTM_b":
        my_ML_model = CNN_LSTM_b()            
    elif ML_type == "CNN_LSTM_c":
        my_ML_model = CNN_LSTM_c()    
    elif ML_type == "FC":
        my_ML_model = FC()     
    run_ML()
