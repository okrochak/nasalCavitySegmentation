from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import ray
from ray import tune
from ray.tune import CLIReporter

def load_data(data_dir=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=transform)

    return trainset
    

def train_cifar(config, data_dir=None):
        
    net = models.resnet18()
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)


    trainset = load_data(data_dir)


    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0)

        
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        running_correct = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            pred = outputs.argmax(dim=1, keepdim=True)
            loss.backward()
            optimizer.step()
            
            running_correct += pred.eq(labels.view_as(pred)).sum().item()


        tune.report(loss = loss.item(), accuracy=running_correct / len(trainset))
        
    print("Finished Training")
    

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    ray.init(address='auto')
    
    
    config = {
        "batch_size": tune.choice([64, 128, 256, 512]),
        "lr": tune.loguniform(10e-5, 1)
    }
    
    result = tune.run(
        partial(train_cifar, data_dir='/p/project/raise-ctp2/cifar10/data'),
        local_dir=os.path.join(os.path.abspath(os.getcwd()), "ray_results"),
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=None)

    
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=1)