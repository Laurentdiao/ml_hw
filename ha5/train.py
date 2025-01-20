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

data_file = 'samples_200.txt'
numbers = []
labels = []
with open(data_file, 'r') as f:
    next(f)
    for line in f:
        number, label = line.strip().split('\t')
        numbers.append(int(number))
        labels.append(int(label))

def number_to_tensor(number, max_length=6):
    number_str = str(number).zfill(max_length)
    tensor = torch.zeros(len(number_str), 10) 
    for li, digit in enumerate(number_str):
        tensor[li][int(digit)] = 1
    return tensor.view(len(number_str), 1, -1)

def batch_to_tensor(batch_numbers, max_length=6):
    batch_size = len(batch_numbers)
    batch_tensor = torch.zeros(max_length, batch_size, 10)
    for i, number in enumerate(batch_numbers):
        number_str = str(number).zfill(max_length)
        for li, digit in enumerate(number_str):
            batch_tensor[li][i][int(digit)] = 1
    return batch_tensor.to(device)

input_size = 10
hidden_size = 256
output_size = 2
learning_rate = 0.001
batch_size = 64
n_iters = 10000
print_every = 1000
plot_every = 1000

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

rnn = RNN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

def random_training_batch(batch_size):
    indices = random.sample(range(len(numbers)), batch_size)
    batch_numbers = [numbers[i] for i in indices]
    label_tensors = torch.tensor([labels[i] for i in indices], dtype=torch.long).to(device)
    number_tensors = batch_to_tensor(batch_numbers)
    return number_tensors, label_tensors

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {s:.0f}s'

current_loss = 0
all_losses = []
start = time.time()

for iter in range(1, n_iters + 1):
    number_tensors, label_tensors = random_training_batch(batch_size)
    hidden = rnn.init_hidden(batch_size)

    for digit in range(number_tensors.size(0)):
        output, hidden = rnn(number_tensors[digit], hidden)

    loss = criterion(output, label_tensors)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    current_loss += loss.item()

    if iter % print_every == 0:
        print(f'{iter} {iter / n_iters * 100:.2f}% ({timeSince(start)}) Loss: {loss.item():.4f}')

    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

model_path = "rnnmodel_200.pth"
torch.save(rnn.state_dict(), model_path)