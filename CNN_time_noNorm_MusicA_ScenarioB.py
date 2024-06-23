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
from sklearn.preprocessing import StandardScaler

################################### Inputs
recordings_dir = r"C:\Users\mn1059928\OneDrive - Bose Corporation\Desktop\Audio_short"
noise_dir = r"C:\Users\mn1059928\OneDrive - Bose Corporation\Desktop\Noise_to_use_two"
window_size_sec = 1  # in [s]
sampling_freq = 44100  # in [Hz]  
window_len_sample = window_size_sec * sampling_freq
num_noise_combinations = 3
num_epochs=200
train_ratio = 0.6
val_ratio = 0.2
downsampling_new_sr = 344    #6890   # Ratio=64
window_len_sample_downsampled = window_size_sec * downsampling_new_sr
batch_size = 2
use_filter=False
filter_num_coeff = [1]
filter_dem_coeff = [1, 1]
audio_gain = 10 # dB
noise_gain = 5 # dB
normalization_flag = False
################################### Main
def loading_data(dir):
    files = os.listdir(dir) 
    full_recordings = {}
    for i,file in enumerate(files):
        audio_path=os.path.join(dir,file)
        audio_data, sample_rate = librosa.load(audio_path,sr=None)
        full_recordings[i] = audio_data
    return full_recordings
def resampling(orig_data):
    full_resampled_data={}
    for i in range(0, len(orig_data)):
        downsampled_data_temp = librosa.resample(orig_data[i],orig_sr=sampling_freq,target_sr=downsampling_new_sr)
        full_resampled_data[i] = downsampled_data_temp
        # sd.play(audio_data_downsampled, samplerate=downsampling_new_sr)
        # sd.wait()
    return full_resampled_data
def adding_gain(data,gain):
    desired_gain_linear = 10 ** (gain / 20)
    adjusted_rec = {}
    for i in range (0, len(data)):
        adjusted_single_rec = data[i] * desired_gain_linear
        adjusted_rec[i] = adjusted_single_rec
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
    for i in range (0, len(audio)):
        for ii in range (0, num_noise_combinations):
            duplicated_audio[master_c]=audio[i]
            duplicated_noise[master_c]=noise[ii]
            master_c+=1
    return duplicated_audio, duplicated_noise
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
def normalization (train_x_no_norm, val_x_no_norm, test_x_no_norm):

    all_windows_training = np.concatenate ([np.array(value) for value in train_x_no_norm.values()])
    scaler = StandardScaler()
    scaler.fit(all_windows_training.reshape(-1,1))
    normalized_training_windows = {key:scaler.transform(np.array(value).reshape(-1,1)).flatten().tolist() for key, value in train_x_no_norm.items()}

    normalized_validation_windows = {key:scaler.transform(np.array(value).reshape(-1,1)).flatten().tolist() for key, value in val_x_no_norm.items()}

    normalized_testing_windows = {key:scaler.transform(np.array(value).reshape(-1,1)).flatten().tolist() for key, value in test_x_no_norm.items()}

    return normalized_training_windows, normalized_validation_windows, normalized_testing_windows



    return 
def find_RMS_noise(data):
    RMS_values = {}
    for i, recording in enumerate (data.items()):
        my_data = recording[1]
        RMS_values [i] = np.sqrt(np.mean((np.array(my_data)**2)))
    
    RMS_values_new = {}
    for key, value in RMS_values.items():
        RMS_values_new[key] = np.array(value, dtype= np.float32)
    return RMS_values_new
def find_RMS_noise_with_norm(data_1, data_2, data_3):
    # Update to Numpy
    RMS_values_1 = {}
    for i, recording in enumerate (data_1.items()):
        my_data = recording[1]
        RMS_values_1[i] = librosa.feature.rms(y=np.array(my_data)).mean()
    RMS_values_2 = {}
    for i, recording in enumerate (data_2.items()):
        my_data = recording[1]
        RMS_values_2[i] = librosa.feature.rms(y=np.array(my_data)).mean()
    RMS_values_3 = {}
    for i, recording in enumerate (data_3.items()):
        my_data = recording[1]
        RMS_values_3[i] = librosa.feature.rms(y=np.array(my_data)).mean()
    return RMS_values_1, RMS_values_2, RMS_values_3
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

    return train_x, val_x, test_x, train_y, val_y, test_y, train_z, val_z, test_z

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2,stride=2,padding=0)
        self.flattened_size= self._get_flattened_size()
        self.fc1 = nn.Linear(self.flattened_size,512)
        self.fc2= nn.Linear(512,128)
        self.fc3 = nn.Linear(128, 1)
        
    def _get_flattened_size(self):
        x = torch.zeros(1,2,window_len_sample_downsampled) # one sample regardless the batch size, num channels, num timepoints
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        return x.numel()

    def forward(self,x):
        x=self.pool(self.relu(self.conv1(x)))
        x=self.pool(self.relu(self.conv2(x)))
        x=self.pool3(self.relu(self.conv3(x)))

        x_dim = x.dim()
        if x_dim==3:
            x=x.view(x.size(0), -1)
        else:
            x=x.view(-1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
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

def ML_training(train_inputs,train_labels):
    dataset = CustomDataset(train_inputs,train_labels)
    dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=True)

    reg_criterion = nn.MSELoss()
    model = CNN()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    train_loss_values = []
    error=[]
    predictions=[]
    gt=[]
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0
        num_train_batches = len(train_inputs)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs.squeeze(1))
            loss_value = reg_criterion(outputs, targets)
            loss_value.backward()
            optimizer.step()
            running_train_loss += loss_value.item()

            if epoch == num_epochs-1:
                ground_truth_values = targets.detach().cpu().numpy().flatten()
                predicted_values = outputs.detach().cpu().numpy().flatten()
                error.append(((abs(ground_truth_values-predicted_values))/(ground_truth_values))*100)

                predictions.append(predicted_values)
                gt.append(ground_truth_values)

        avg_train_loss = running_train_loss / num_train_batches
        train_loss_values.append(avg_train_loss)

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

        print(f"Orig:{ground_truth_value}, Predicted: {predicted_value}")

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
    plt.savefig("Training Error_per_Epoch.png")
def plotting_results(errors,title):
    errors_ready = [error.item() for error in errors]
    plt.figure(figsize=(10,5))
    plt.hist(errors_ready,bins=10)
    plt.xlabel("Error [%]")
    plt.ylabel("Num of datapoints")
    plt.title(title)
    #plt.show()
    plt.savefig(f"{title}.png")
def plotting_results_general_training(error,predictions,gt,printing_label):
    error_ready = [element for array in error for element in array.tolist()]
    plt.figure(figsize=(10,5))
    plt.hist(error_ready,bins=100)
    plt.xlabel("Error [%]")
    plt.ylabel("Num of datapoints")
    #plt.title(title)
    #plt.show()
    plt.savefig(f"{printing_label} histogram performance.png")

    real_ready = [element for array in gt for element in array.tolist()]
    pred_ready = [element for array in predictions for element in array.tolist()]
    diff = [a-b for a,b in zip(real_ready,pred_ready)]
    plt.figure(figsize=(10,5))
    plt.plot(real_ready[1:1000])
    plt.plot(pred_ready[1:1000])
    #plt.title(title)
    #plt.show()
    plt.savefig(f"{printing_label} raw performance.png")
def plotting_results_general_other(error,predictions,gt,printing_label):
    error_ready = [element for array in error for element in array.tolist()]
    plt.figure(figsize=(10,5))
    plt.hist(error_ready,bins=30)
    plt.xlabel("Error [%]")
    plt.ylabel("Num of datapoints")
    #plt.title(title)
    #plt.show()
    plt.savefig(f"{printing_label} histogram performance.png")

    real_ready = [value.item() for value in gt]
    pred_ready = [value.item() for value in predictions]
    plt.figure(figsize=(10,5))
    plt.plot(real_ready)
    plt.plot(pred_ready)
    #plt.title(title)
    #plt.show()
    plt.savefig(f"{printing_label} raw performance.png")










def run_CNN():
    # Preprocessing
    Data_A = loading_data(noise_dir)
    Data_B = resampling(Data_A)
    Data_C = adding_gain(Data_B, noise_gain)
    Data_D = loading_data(recordings_dir)
    Data_E = resampling(Data_D)
    Data_F = transfer_fun(Data_E)
    data_G = adding_gain(Data_F, audio_gain)

    Data_I, Data_H = duplicating_recordings(data_G, Data_C)
    Data_J = concatenating_noise(Data_I, Data_H)

    Data_K = mixing(Data_I, Data_J)
    Data_X= windowing(Data_K)
    

    Data_L, _ = duplicating_recordings(Data_E, Data_C)
    Data_Y = windowing(Data_L)

    _ , Data_M = duplicating_recordings(data_G, Data_B)
    Data_N = concatenating_noise(Data_I, Data_M)
    Data_O = windowing(Data_N) 

    
           

    # Data prep for ML work
    
    if normalization_flag == True:
        train_x_no_norm, val_x_no_norm, test_x_no_norm, train_y_no_norm, val_y_no_norm, test_y_no_norm, train_z_no_norm, val_z_no_norm, test_z_no_norm = data_splitting (Data_X, Data_Y, Data_O)

        train_x, val_x, test_x = normalization (train_x_no_norm, val_x_no_norm, test_x_no_norm)
        train_y, val_y, test_y = normalization (train_y_no_norm, val_y_no_norm, test_y_no_norm)

        train_z_full_data, val_z_full_data, test_z_full_data = normalization (train_z_no_norm, val_z_no_norm, test_z_no_norm)
        train_z, val_z, test_z = find_RMS_noise_with_norm(train_z_full_data, val_z_full_data, test_z_full_data)  
    else:
        Data_Z = find_RMS_noise(Data_O) 
        train_x, val_x, test_x, train_y, val_y, test_y, train_z, val_z, test_z = data_splitting (Data_X, Data_Y, Data_Z)

    data_train_xy = data_prep_for_ML(train_x, train_y); data_val_xy = data_prep_for_ML(val_x, val_y); data_test_xy = data_prep_for_ML(test_x, test_y)

    train_z_l= list(train_z.values()); val_z_l = list(val_z.values()); test_z_l = list(test_z.values())

    # ML work
    model = ML_training(data_train_xy,train_z_l)
    ML_validating(model, data_val_xy, val_z_l)
    ML_testing(model, data_test_xy, test_z_l)
    
    stop=1
    
    # Different architecture
    # Other features than RMS or actual data
    # Return back parameters

if __name__ == "__main__":
    run_CNN()