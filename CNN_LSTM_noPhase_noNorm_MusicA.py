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

################################### Inputs
recordings_path = r"C:\Users\mn1059928\OneDrive - Bose Corporation\Desktop\Generated_noisy_recordings_short"
noise_path = r"\\solid\oemcom\MN_2024\Noise\to_use"
window_size_sec = 4    # in [s]
sampling_freq = 44100  # in [Hz]  
window_len_sample = window_size_sec * sampling_freq
num_noise_combinations = 27
num_epochs=50
train_ratio = 0.6
val_ratio = 0.2
downsampling_new_sr = 200
window_len_sample_downsampled = window_size_sec * downsampling_new_sr
batch_size = 2
################################### Main
def audio_windowing(audio):
    num_windows = len(audio) // window_len_sample_downsampled
    extra_samples = len(audio) % (num_windows*window_len_sample_downsampled)
    truncated_audio = audio[extra_samples:]
    audio_windows = np.array_split(truncated_audio, num_windows)
    min_length = min(len(array) for array in audio_windows)
    audio_windows_truncated = [window[:min_length] for window in audio_windows]
    audio_windows_array = np.vstack(audio_windows_truncated)
    return audio_windows_array
def generate_IO():
    # Importing noise file names
    with open(r"C:\Users\mn1059928\OneDrive - Bose Corporation\Desktop\recordings_dict.pkl","rb") as file:
        orig_noise = pickle.load(file)["noise"]

    # Initialization
    audio_files = os.listdir(recordings_path) 
    input_data = np.empty(shape=(0,window_len_sample_downsampled))
    output_data = np.empty(shape=(0,window_len_sample_downsampled))

    # Iterate over files and split into windows
    noise_index=1
    for file in audio_files:
        audio_path=os.path.join(recordings_path,file)
        audio_data, sample_rate = librosa.load(audio_path,sr=None)
        audio_data_downsampled = librosa.resample(audio_data,orig_sr=sampling_freq,target_sr=downsampling_new_sr)
        audio_windows = audio_windowing(audio_data_downsampled)
        input_data=np.append(input_data,audio_windows,axis=0)

        current_noise_file_name = orig_noise[noise_index-1]
        current_noise_file = os.path.join(noise_path,current_noise_file_name)
        noise_data, sample_rate = librosa.load(current_noise_file,sr=None)
        repeat_factor = int(np.ceil(len(audio_data) / len(noise_data)))
        noise_data_full_length = np.tile(noise_data, repeat_factor)[:len(audio_data)]
        noise_data_full_length_downsampled = librosa.resample(noise_data_full_length,orig_sr=sampling_freq,target_sr=downsampling_new_sr)
        noise_windows = audio_windowing(noise_data_full_length_downsampled)
        output_data=np.append(output_data,noise_windows,axis=0)

        if noise_index != num_noise_combinations:
            noise_index+=1  
        else:
            noise_index = 1

    return input_data, output_data
def generate_spectrogram(input_data):
    tf_inputs = {}

    for i, datapoint in enumerate (input_data):
        tf_inputs[i+1] = librosa.stft(datapoint, n_fft=256,hop_length=128)

    num_freq_comp = tf_inputs[1].shape[0]
    num_time_comp = tf_inputs[1].shape[1]

    inputs_tensor = {k:torch.tensor(v,dtype=torch.float32).unsqueeze(0).unsqueeze(0) for k, v in tf_inputs.items()}

    return inputs_tensor, num_freq_comp, num_time_comp
def find_RMS_noise(output_data):
    RMS_values = {}

    for i, datapoint in enumerate (output_data):
        RMS_values[i+1] = librosa.feature.rms(y=datapoint).mean()

    return RMS_values
class CNN(nn.Module):
    def __init__(self,num_freq_comp,num_time_comp):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * (num_freq_comp//4)*(num_time_comp//4),128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128,num_layers=2, batch_first=True)
        self.fc2 = nn.Linear(128, 1)
    def forward(self,x,num_freq_comp,num_time_comp):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=x.view(-1,32 * (num_freq_comp//4)*(num_time_comp//4))
        x=torch.relu(self.fc1(x))
        x = x.unsqueeze(1)
        x,_ = self.lstm(x)
        x = x[:,-1,:]
        x=self.fc2(x)
        return x
    
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
    plt.hist(error_ready,bins=10)
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
class CustomDataset(Dataset):
    def __init__(self,inputs_dict,labels_dict):
        self.inputs = inputs_dict
        self.labels = labels_dict
        self.keys = list(inputs_dict.keys())
    def __len__(self):
        return(len(self.keys))
    def __getitem__(self, idx):
        key = self.keys[idx]
        input_data = self.inputs[key]
        label = self.labels[key]
        return input_data, label
def ML_train_model(num_freq_comp,num_time_comp,train_inputs,train_labels, val_inputs, val_labels):
    reg_criterion = nn.MSELoss()
    
    
    dataset = CustomDataset(train_inputs,train_labels)
    dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=True)


    model = CNN(num_freq_comp,num_time_comp)
    reg_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    # Training
    train_loss_values = []
    error_training_arr={}
    predictions_training_arr={}
    ground_truth_training_arr={}
    error=[]
    predictions=[]
    gt=[]
    for epoch in range(num_epochs):

        model.train()
        running_train_loss = 0
        num_train_batches = len(train_inputs)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs.squeeze(1),num_freq_comp,num_time_comp)
            loss_value = reg_criterion(outputs, targets)
            loss_value.backward()
            optimizer.step()
            running_train_loss += loss_value.item()

            ground_truth_values = targets.detach().cpu().numpy().flatten()
            predicted_values = outputs.detach().cpu().numpy().flatten()
            error.append(((abs(ground_truth_values-predicted_values))/(ground_truth_values))*100)

            predictions.append(predicted_values)
            gt.append(ground_truth_values)

        avg_train_loss = running_train_loss / num_train_batches
        train_loss_values.append(avg_train_loss)

    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)
    with open("train_loss_values.pkl", "wb") as file:
        pickle.dump(train_loss_values, file)
    with open("error.pkl", "wb") as file:
        pickle.dump(error, file)
    with open("predictions.pkl", "wb") as file:
        pickle.dump(predictions, file)
    with open("gt.pkl", "wb") as file:
        pickle.dump(gt, file)


    # with open("model.pkl","rb") as file:
    #     model = pickle.load(file)
    # with open("train_loss_values.pkl","rb") as file:
    #     train_loss_values = pickle.load(file)
    # with open("error.pkl","rb") as file:
    #     error = pickle.load(file)
    # with open("predictions.pkl","rb") as file:
    #     predictions = pickle.load(file)
    # with open("gt.pkl","rb") as file:
    #     gt = pickle.load(file)


    plotting_performance(train_loss_values,"Training")
    plotting_results_general_training(error,predictions,gt,"Training")
    ###################################################################
    # Validation
    val_loss_values = []

    model.eval()
    running_val_loss = 0
    num_val_batches = len(val_inputs)
    errors_val=[]
    all_pred=[]
    all_gt=[]
    for key in val_inputs:
        my_input = val_inputs[key].requires_grad_(True)
        ground_truth_value = torch.tensor(val_labels[key])        

        predicted_value = model(my_input,num_freq_comp,num_time_comp)
        loss_value = reg_criterion(predicted_value,ground_truth_value)

        running_val_loss += loss_value.item()

        avg_val_loss = running_val_loss / num_val_batches
        val_loss_values.append(avg_val_loss)    

        errors_val.append(((abs(ground_truth_value-predicted_value))/(ground_truth_value))*100)
        all_pred.append(predicted_value)
        all_gt.append(ground_truth_value)
    printing_label = "Validation"
    plotting_results_general_other(errors_val,all_pred,all_gt,printing_label)

    return model
def ML_test_model(model, num_freq_comp, num_time_comp, test_inputs, test_labels):
    model.eval()
    test_loss = 0
    test_errors=[]
    num_test_batches = len(test_inputs)
    reg_criterion = nn.MSELoss()
    errors_test = []
    all_gt=[]
    all_pred=[]
    for key in test_inputs:
        my_input = test_inputs[key].requires_grad_(True)
        ground_truth_value = torch.tensor(test_labels[key])        

        predicted_value = model(my_input,num_freq_comp,num_time_comp)
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
   

    return test_errors
def data_splitting(data,labels):
    keys = list(data.keys())
    random.shuffle(keys)

    train_end = int(train_ratio * len(keys))
    val_end = train_end + int(val_ratio * len(keys))

    train_keys = keys[:train_end]
    val_keys = keys[train_end:val_end]
    test_keys = keys[val_end:]

    train_inputs = {key: data[key] for key in train_keys}
    val_inputs = {key: data[key] for key in val_keys}
    test_inputs = {key: data[key] for key in test_keys}

    train_labels = {key: labels[key] for key in train_keys}
    val_labels = {key: labels[key] for key in val_keys}
    test_labels= {key: labels[key] for key in test_keys}

    return train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels



def run_CNN():
    #input_data, output_data = generate_IO()

    # with open("input_data.pkl", "wb") as file:
    #     pickle.dump(input_data, file)
    # with open("output_data.pkl", "wb") as file:
    #     pickle.dump(output_data, file)


    with open("input_data.pkl","rb") as file:
        input_data = pickle.load(file)
    with open("output_data.pkl","rb") as file:
        output_data = pickle.load(file)

    inputs_tensor, num_freq_comp, num_time_comp = generate_spectrogram(input_data)
    labels = find_RMS_noise(output_data)

    train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels = data_splitting (inputs_tensor, labels)
    
    model = ML_train_model(num_freq_comp,num_time_comp,train_inputs,train_labels, val_inputs, val_labels)
    test_errors = ML_test_model(model, num_freq_comp, num_time_comp, test_inputs, test_labels)
    
if __name__ == "__main__":
    run_CNN()