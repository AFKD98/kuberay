"""
Reference: https://docs.ray.io/en/master/train/examples/pytorch/torch_fashion_mnist_example.html

This script is a modified version of a PyTorch CIFAR-10 example.
It uses only CPU resources to train a simple CNN model on the CIFAR-10 dataset.
See `ScalingConfig` for more details.
"""

import os
from typing import Dict

import torch
from filelock import FileLock
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import ray.train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer


def get_dataloaders(batch_size: int):
    # Define the transformation: Convert images to tensors and normalize.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    with FileLock(os.path.expanduser("~/data.lock")):
        # Download the CIFAR-10 training and test datasets.
        train_data = datasets.CIFAR10(
            root="~/data",
            train=True,
            download=True,
            transform=transform,
        )
        test_data = datasets.CIFAR10(
            root="~/data",
            train=False,
            download=True,
            transform=transform,
        )

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader


# Define a simple CNN model for CIFAR-10.
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # CIFAR-10 images are RGB.
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample from 32x32 to 16x16.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample from 16x16 to 8x8.
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def train_func_per_worker(config: Dict):
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]

    # Get dataloaders for CIFAR-10.
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # Prepare the DataLoaders for distributed training.
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    test_loader = ray.train.torch.prepare_data_loader(test_loader)

    model = SimpleCNN()

    # Wrap the model to enable distributed training.
    model = ray.train.torch.prepare_model(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop.
    for epoch in range(epochs):
        if ray.train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)

        model.train()
        for X, y in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation loop.
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in tqdm(test_loader, desc=f"Test Epoch {epoch}"):
                pred = model(X)
                loss = loss_fn(pred, y)
                test_loss += loss.item()

                total += y.size(0)
                correct += (pred.argmax(1) == y).sum().item()

        test_loss /= len(test_loader)
        accuracy = correct / total

        # Report metrics to Ray Train.
        ray.train.report(metrics={"loss": test_loss, "accuracy": accuracy})


def train_cifar10(num_workers: int = 4, cpus_per_worker: int = 2, use_gpu: bool = False):
    global_batch_size = 64

    train_config = {
        "lr": 1e-3,
        "epochs": 5,
        "batch_size_per_worker": global_batch_size // num_workers,
    }

    # Configure computation resources.
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker={"CPU": cpus_per_worker},
    )

    # Initialize the Ray TorchTrainer.
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
    )

    # Start distributed training.
    result = trainer.fit()
    print(f"Training result: {result}")
    print('Benchmark Completed Successfully.')


if __name__ == "__main__":
    num_workers = int(os.getenv("NUM_WORKERS", "4"))
    cpus_per_worker = int(os.getenv("CPUS_PER_WORKER", "2"))
    train_cifar10(num_workers=num_workers, cpus_per_worker=cpus_per_worker)
