import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_train_valid_loader(data_dir, batch_size, augment, random_seed, valid_size=0.1, shuffle=True):
    """
    Returns training and validation data loaders for the CIFAR-10 dataset.
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # Define validation transforms
    valid_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
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
    """
    Returns the test data loader for the CIFAR-10 dataset.
    """
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
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def train_and_validate(model, train_loader, valid_loader, criterion, optimizer, num_epochs):
    """
    Trains the model and validates it on the validation set after each epoch.
    """
    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f"Validation Accuracy: {100 * correct / total:.2f}% on {total} images")
    return model


def test_model(model, test_loader):
    """
    Evaluates the trained model on the test dataset.
    """
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Test Accuracy: {100 * correct / total:.2f}% on {total} images")


def measure_inference_time(model, dummy_input, num_runs=100, warmup_runs=10):
    """
    Measures and prints the average inference time of the model using dummy input.
    """
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            if dummy_input.device.type == "cuda":
                torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(num_runs):
            _ = model(dummy_input)
            if dummy_input.device.type == "cuda":
                torch.cuda.synchronize()
        total_time = time.time() - start_time
    avg_time = total_time / num_runs
    print(f"Average inference time over {num_runs} runs: {avg_time:.6f} seconds per run")
    return avg_time


def main():
    data_dir = "./data"
    batch_size = 64
    random_seed = 1
    num_classes = 10
    num_epochs = 20
    learning_rate = 0.005

    train_loader, valid_loader = get_train_valid_loader(
        data_dir, batch_size, augment=False, random_seed=random_seed
    )
    test_loader = get_test_loader(data_dir, batch_size)

    model = AlexNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

    model = train_and_validate(model, train_loader, valid_loader, criterion, optimizer, num_epochs)

    test_model(model, test_loader)

    data_iter = iter(train_loader)
    images, _ = next(data_iter)
    images = images.to(device)
    measure_inference_time(model, images, num_runs=100, warmup_runs=10)


if __name__ == "__main__":
    main()
