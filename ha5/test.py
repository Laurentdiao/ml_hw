"""
Student Name: Diao Weili
Student ID: 21127071
Student Email: wdiaoaa@connect.ust.hk
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torch.nn.functional as F
import random
import time
import math

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
os.chdir(current_directory)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 10
hidden_size = 256
output_size = 2

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        hidden = self.dropout(hidden)
        output = self.h2o(hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)

def predict(model, number):
    with torch.no_grad():
        number_tensor = number_to_tensor(number).to(device)
        hidden = model.init_hidden(1)
        for digit in number_tensor:
            output, hidden = model(digit.view(1, -1), hidden)
        prediction = F.softmax(output, dim=1)
        return torch.argmax(prediction).item()

test_data_file = 'test.txt'
test_numbers = []
test_labels = []
with open(test_data_file, 'r') as f:
    for line in f:
        number, label = line.strip().split('\t')
        test_numbers.append(int(number))
        test_labels.append(int(label))

def number_to_tensor(number, max_length=6):
    number_str = str(number).zfill(max_length)
    tensor = torch.zeros(len(number_str), 10) 
    for li, digit in enumerate(number_str):
        tensor[li][int(digit)] = 1
    return tensor.view(len(number_str), 1, -1)

def load_model(model_path):
    model = RNN(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

loaded_rnn = load_model("rnnmodel_20000.pth")

correct = 0
total = len(test_numbers)
for i, number in enumerate(test_numbers):
    prediction = predict(loaded_rnn, number)
    correct += (prediction == test_labels[i])

accuracy = correct / total * 100
print(f'Accuracy: {accuracy:.2f}%')