import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class NeuralNetwork(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.pop()
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class ProductDataset(Dataset):
    def __init__(self, targets_file, inputs_file, transform=None, target_transform=None):
        self.targets = pd.read_csv(targets_file)
        self.inputs = pd.read_csv(inputs_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        input_features = self.inputs.iloc[idx, :]
        target = self.targets.iloc[idx, 0]
        if self.transform:
            input_features = self.transform(input_features)
        if self.target_transform:
            target = self.target_transform(target)
        return input_features, target

def load_datasets(data_file, batch_size: int):
    m_dataset = ProductDataset(data_file[0], data_file[1])
    m_dataloader = DataLoader(m_dataset, batch_size=batch_size, shuffle=True)
    return m_dataloader

def training_loop(dataloader, model, loss_fn, optimizer):
    model.training()
    for batch, (x, y) in enumerate(dataloader):
        ypred = model(x)
        loss = loss_fn(ypred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def main(training_data_files, testing_data_files, batch_size, learning_rate, epochs, model_save_path, loss_fn=nn.MSELoss(), optimizer_fn = torch.optim.SGD, dims=[], model_load_path=''):
    training_dataloader = load_datasets(training_data_files, batch_size)
    testing_dataloader = load_datasets(testing_data_files, batch_size)
    if model_load_path:
        model = torch.load(model_load_path)
    else:
        model = NeuralNetwork(dims)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)

    optimizer = optimizer_fn(model.parameters(), lr=learning_rate)

    for i in range(epochs):
        training_loop(training_dataloader, model, loss_fn, optimizer)
    torch.save(model, model_save_path)


