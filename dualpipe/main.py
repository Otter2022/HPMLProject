#!/usr/bin/env python
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from fairscale.nn import Pipe
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

##################################
# Data Loader Helper Functions   #
##################################
def get_train_valid_loader(data_dir, batch_size, augment, random_seed, valid_size=0.1, shuffle=True):
    # Normalization for CIFAR-10 (for training and validation)
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    
    # For validation, resize to 227x227 directly.
    valid_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])
    
    # For training when augmentation is enabled, first resize to 256x256 then randomly crop to 227x227.
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize,
        ])
    
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )
    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)
    return train_loader, valid_loader

def get_test_loader(data_dir, batch_size, shuffle=True):
    # For testing, we also resize the image to 227x227.
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    transform_pipeline = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform_pipeline,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return test_loader

##################################
# AlexNet Model Definitions      #
##################################
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        # Flatten the output and use clone() to avoid view issues.
        out = out.view(out.size(0), -1).clone()
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

####################################
# Partitioned AlexNet for Pipelining #
####################################
class AlexNetStage1(nn.Module):
    def __init__(self, original_model: AlexNet):
        super(AlexNetStage1, self).__init__()
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.layer5 = original_model.layer5

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1).clone()
        return x

class AlexNetStage2(nn.Module):
    def __init__(self, original_model: AlexNet):
        super(AlexNetStage2, self).__init__()
        self.fc = original_model.fc
        self.fc1 = original_model.fc1
        self.fc2 = original_model.fc2

    def forward(self, x):
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

##########################################
# Pipeline Training Main Function        #
##########################################
def main():
    # Hyper-parameters and settings.
    class Args:
        data_dir = "./data"
        batch_size = 128       # Global batch size.
        num_classes = 10
        num_epochs = 20
        lr = 0.005
        num_chunks = 4         # Number of micro-batches.
    args = Args()

    # Specify the devices for the pipeline (2 GPUs in this case).
    devices = ["cuda:0", "cuda:1"]

    # Prepare the data loaders.
    train_loader, _ = get_train_valid_loader(args.data_dir, args.batch_size, augment=True, random_seed=42)
    test_loader = get_test_loader(args.data_dir, args.batch_size)

    # Build the original AlexNet and partition it into two stages.
    original_model = AlexNet(num_classes=args.num_classes)
    stage1 = AlexNetStage1(original_model).to(devices[0])
    stage2 = AlexNetStage2(original_model).to(devices[1])
    model_seq = nn.Sequential(stage1, stage2)

    # Wrap the sequential model in FairScale's Pipe with an explicit balance.
    # Here, balance=[1, 1] assigns one module per device.
    pipeline_model = Pipe(model_seq, balance=[1, 1], devices=devices, chunks=args.num_chunks)

    # Setup the optimizer.
    optimizer = optim.SGD(pipeline_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.005)

    # Set the model to training mode.
    pipeline_model.train()

    # Training loop.
    for epoch in range(args.num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move the input images to the first device.
            images = images.to(devices[0])
            # Since the final output is on the last device, move labels there.
            labels = labels.to(devices[-1])
            optimizer.zero_grad()
            # FairScale's Pipe handles splitting the batch into micro-batches.
            outputs = pipeline_model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        print(f"Epoch {epoch+1} complete.")

if __name__ == "__main__":
    main()
