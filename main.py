'''Train Models using PyTorch.'''
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

def train(model, device, train_loader, optimizer,scheduler):
    """Model Training Loop
    Args:
        model : torch model 
        device : "cpu" or "cuda" gpu 
        train_loader : Torch Dataloader for trainingset
        optimizer : optimizer to be used
    Returns:
        float: accuracy and loss values
    """
    model.train()
    pbar = tqdm(train_loader)
    lr_trend = []
    correct = 0
    processed = 0
    num_loops = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(pbar):
            # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch 
        # accumulates the gradients on subsequent backward passes. Because of this, when you start your training loop, 
        # ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)
        # Calculate loss
        loss = F.nll_loss(y_pred, target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        scheduler.step()
        lr_trend.append(scheduler.get_last_lr()[0])

        train_loss += loss.item()
        
        # Update pbar-tqdm
        
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        num_loops += 1
        pbar.set_description(desc= f'Batch_id={batch_idx} Loss={train_loss/num_loops:.5f} Accuracy={100*correct/processed:0.2f}')

    return 100*correct/processed, train_loss/num_loops, lr_trend

def test(model, device, test_loader):
    """Model Testing Loop
    Args:
        model : torch model 
        device : "cpu" or "cuda" gpu 
        test_loader : Torch Dataloader for testset
    Returns:
        float: accuracy and loss values
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    
    return 100. * correct / len(test_loader.dataset), test_loss

def fit_model(net, device, train_loader, test_loader, optimizer, scheduler, NUM_EPOCHS=20):
    """Train+Test Model using train and test functions
    Args:
        net : torch model 
        NUM_EPOCHS : No. of Epochs
        device : "cpu" or "cuda" gpu 
        train_loader: Train set Dataloader with Augmentations
        test_loader: Test set Dataloader with Normalised images

    Returns:
        model, Tran Test Accuracy, Loss
    """
    training_acc, training_loss, testing_acc, testing_loss = list(), list(), list(), list()

    lr_trend = []
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1,NUM_EPOCHS+1):
        print("EPOCH: {} (LR: {})".format(epoch, optimizer.param_groups[0]['lr']))
        train_acc, train_loss, lr_hist = train(net, device, train_loader, optimizer,scheduler)
        test_acc, test_loss = test(net, device, test_loader)

        training_acc.append(train_acc)
        training_loss.append(train_loss)
        testing_acc.append(test_acc)
        testing_loss.append(test_loss)
        lr_trend.extend(lr_hist)
        
    return net, (training_acc, training_loss, testing_acc, testing_loss, lr_trend)