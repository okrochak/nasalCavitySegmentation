# general imports
import numpy as np
import os

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Ray Tune imports
import ray
from ray import tune
from ray.tune import CLIReporter


# method to average the parameters over all GPUs

# mean of field over GPUs
def par_mean(field):
    res = torch.tensor(field).float()
    res = res.cuda()
    dist.all_reduce(res,op=dist.ReduceOp.SUM,group=None,async_op=True).wait()
    res/=dist.get_world_size()
    return res


# dataloading method
def load_data(data_dir=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=transform)

    return trainset
    

# cifar training method
def train_cifar(config):
    
    # get model
    net = models.resnet18()
    
    # perpare model for RayTune
    net = ray.train.torch.prepare_model(net)    

    # loss and optimizer definition
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)


    # get the training set
    trainset = load_data('/p/project/raise-ctp2/cifar10/data')

    # define dataloader with hyperparameters set by RayTune
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    
    # prepare dataloader for RayTune
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

        
    for epoch in range(20):  # loop over the dataset multiple times
        
        loss = 0 
        
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        loss = par_mean(loss)
            
        # report metric of interest back to RayTune    
        ray.train.report(loss = loss.item())
        
    print("Finished Training")
    

def main(num_samples, max_num_epochs, gpus_per_trial):
    ray.init(address='auto')
    
    
    # prepare RayTune with PyTorch DDP backend, num_workers specifies the number of GPUs to use per trial
    from ray.train import Trainer
    trainer = Trainer(backend="torch", num_workers=gpus_per_trial, use_gpu=True)
    
    # convert the train function to a Ray trainable
    trainable = trainer.to_tune_trainable(train_cifar)
    
    # set search space
    config = {
        "batch_size": tune.choice([64, 128, 256, 512]),
        "lr": tune.loguniform(10e-5, 1)
    }
    
    
    reporter = CLIReporter(
        max_report_frequency=60)
    
    # run hyperparameter optimization
    result = tune.run(
        trainable,
        local_dir=os.path.join(os.path.abspath(os.getcwd()), "ray_results"),
        config=config,
        num_samples=num_samples,
        progress_reporter=reporter,
        verbose=1,
        scheduler=None)


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=30, gpus_per_trial=4)