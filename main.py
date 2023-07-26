'''Train Models using PyTorch.'''
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import SGD

torch.manual_seed(1)


def get_correct_count(prediction, labels):
    return prediction.argmax(dim=1).eq(labels).sum().item()


def get_incorrect_preds(prediction, labels):
    prediction = prediction.argmax(dim=1)
    indices = prediction.ne(labels).nonzero().reshape(-1).tolist()
    return indices, prediction[indices].tolist(), labels[indices].tolist()

def train(model, device, lr_scheduler, criterion, train_loader, optimizer):

    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0
    lr_trend=[]
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        # if l1 > 0:
        #     loss += l1 * sum(p.abs().sum() for p in model.parameters())

        train_loss += loss.item() * len(data)

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += get_correct_count(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Batch_id={batch_idx}')
        lr_scheduler.step()
        lr_trend.append(lr_scheduler.get_last_lr()[0])

    train_acc = 100 * correct / processed
    train_loss /= processed
    
    print(f"\nTrain Accuracy: {train_acc:0.4f}%")
    print(f"Train Average Loss: {train_loss:0.2f}")
    return train_acc, train_loss



def test(model, device, criterion, test_loader):
    model.eval()

    test_loss = 0
    test_loss1 = 0
    correct = 0
    processed = 0
    test_acc=0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            pred = model(data)

            test_loss += criterion(pred, target).item() * len(data)

            correct += get_correct_count(pred, target)
            processed += len(data)


    test_acc = 100 * correct / processed
    test_loss /= processed
    
    print(f"Test Average loss: {test_loss:0.4f}")
    print(f"Test Accuracy: {test_acc:0.2f}%")
    print("\n")
    return test_acc, test_loss

def fit_model(net, device, train_loader, test_loader, criterion, optimizer, lr_scheduler, NUM_EPOCHS=20):
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
        #lr_hist=optimizer.param_groups[0]['lr']
        train_acc, train_loss = train(net, device, lr_scheduler, criterion, train_loader, optimizer)
        test_acc, test_loss = test(net, device, criterion, test_loader)

        training_acc.append(train_acc)
        training_loss.append(train_loss)
        testing_acc.append(test_acc)
        testing_loss.append(test_loss)
        #lr_trend.append(lr_hist)
        
    return (training_acc, training_loss, testing_acc, testing_loss)