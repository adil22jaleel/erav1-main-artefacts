import torch.nn.functional as F
from albumentations import Compose, PadIfNeeded, RandomCrop, Normalize, HorizontalFlip, ShiftScaleRotate, CoarseDropout, Cutout, PadIfNeeded
from albumentations.pytorch.transforms import ToTensorV2
import torch              
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from albumentations import *
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from torch_lr_finder import LRFinder
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget



class album_Compose_train():
    def __init__(self):
        self.albumentations_transform = Compose([
            Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
            PadIfNeeded(min_height=36, min_width=36, border_mode=cv2.BORDER_REFLECT),
            RandomCrop(height=32, width=32),
            Cutout(num_holes=1, max_h_size=16, max_w_size=16, p=1.0),
            ToTensorV2()
        ])
    def __call__(self,img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img


class album_Compose_test():
    def __init__(self):
        self.albumentations_transform = Compose([
            Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615]),
            ToTensorV2()
        ])

    def __call__(self,img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img

class dataset_cifar10:
    """
    Class to load the data and define the data loader
    """

    def __init__(self, batch_size=128):

        # Defining CUDA
        cuda = torch.cuda.is_available()
        print("CUDA availability ?",cuda)

        # Defining data loaders with setting
        self.dataloaders_args = dict(shuffle=True, batch_size = batch_size, num_workers = 2, pin_memory = True) if cuda else dict(shuffle=True,batch_size = batch_size)
        self.sample_dataloaders_args = self.dataloaders_args.copy()

        self.classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def data(self, train_flag):

        # Transformations data augmentation (only for training)
        if train_flag :
            return datasets.CIFAR10('./Data',
                            train=train_flag,
                            transform=album_Compose_train(),
                            download=True)

        # Testing transformation - normalization adder
        else:
            return datasets.CIFAR10('./Data',
                                train=train_flag,
                                transform=album_Compose_test(),
                                download=True)

    # Dataloader function
    def loader(self, train_flag=True):
        return(torch.utils.data.DataLoader(self.data(train_flag), **self.dataloaders_args))


    def data_summary_stats(self):
        # Load train data as numpy array
        train_data = self.data(train_flag=True).data
        test_data = self.data(train_flag=False).data

        total_data = np.concatenate((train_data, test_data), axis=0)
        print(total_data.shape)
        print(total_data.mean(axis=(0,1,2))/255)
        print(total_data.std(axis=(0,1,2))/255)

    def sample_pictures(self, train_flag=True, return_flag = False):

        # get some random training images
        images,labels = next(iter(self.loader(train_flag)))

        sample_size=25 if train_flag else 5

        images = images[0:sample_size]
        labels = labels[0:sample_size]

        fig = plt.figure(figsize=(10, 10))

        # Show images
        for idx in np.arange(len(labels.numpy())):
            ax = fig.add_subplot(5, 5, idx+1, xticks=[], yticks=[])
            npimg = unnormalize(images[idx])
            ax.imshow(npimg, cmap='gray')
            ax.set_title("Label={}".format(str(self.classes[labels[idx]])))

        fig.tight_layout()  
        plt.show()

        if return_flag:
            return images,labels
# CUDA?
def get_device():
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)
  device = torch.device("cuda" if cuda else "cpu")
  return device

def unnormalize(img):
    channel_means = (0.4914, 0.4822, 0.4471)
    channel_stdevs = (0.2469, 0.2433, 0.2615)
    img = img.numpy().astype(dtype=np.float32)
  
    for i in range(img.shape[0]):
         img[i] = (img[i]*channel_stdevs[i])+channel_means[i]
  
    return np.transpose(img, (1,2,0))

#

def custom_lr_finder(model,criterion,optimizer,device,trainloader):
  lr_finder = LRFinder(model, optimizer, criterion, device=device)
  lr_finder.range_test(trainloader, end_lr=10, num_iter=100,step_mode="exp")
  _, best_lr = lr_finder.plot()
  lr_finder.reset()
  return best_lr

# define a function to plot misclassified images
def plot_misclassified(model, test_loader, classes, device, no_misclf=20, plot_size=(4,5), return_misclf=False):
    """Plot the images are wrongly clossified by model

    Args:
        model (instance): torch instance of defined model (pre trained)
        test_loader (instace): torch data loader of testing set
        classes (dict or list): classes in the dataset
                if dict:
                    key - class id
                    value - as class name
                elif list:
                    index of list correspond to class id and name
        device (str): 'cpu' or 'cuda' device to be used
        dataset_mean (tensor or np array): mean of dataset
        dataset_std (tensor or np array): std of dataset
        no_misclf (int, optional): number of misclassified images to plot. Defaults to 20.
        plot_size (tuple): tuple containing size of plot as rows, columns. Defaults to (4,5)
        return_misclf (bool, optional): True to return the misclassified images. Defaults to False.

    Returns:
        list: list containing misclassified images as np array if return_misclf True
    """
    def convert_image_np(inp, mean, std):
      inp = inp.numpy().transpose((1, 2, 0))
      inp = std * inp + mean
      inp = np.clip(inp, 0, 1)
      return inp

    count = 0
    k = 0
    misclf = list()
    dataset_mean, dataset_std = np.array([0.49139968, 0.48215841, 0.44653091]), np.array([0.24703223, 0.24348513, 0.26158784])
  
    while count<no_misclf:
        img_model, label = test_loader.dataset[k]
        pred = model(img_model.unsqueeze(0).to(device)) # Prediction
        # pred = model(img.unsqueeze(0).to(device)) # Prediction
        pred = pred.argmax().item()

        k += 1
        if pred!=label:
            img = convert_image_np(
                img_model, dataset_mean, dataset_std)
            misclf.append((img_model, img, label, pred))
            count += 1
    
    rows, cols = plot_size
    figure = plt.figure(figsize=(cols*3,rows*3))

    for i in range(1, cols * rows + 1):
        _, img, label, pred = misclf[i-1]

        figure.add_subplot(rows, cols, i) # adding sub plot
        plt.title(f"Pred label: {classes[pred]}\n True label: {classes[label]}") # title of plot
        plt.axis("off") # hiding the axis
        plt.imshow(img, cmap="gray") # showing the plot

    plt.tight_layout()
    plt.show()
    
    if return_misclf:
        return misclf


def train_test_plots(history):
    fig, axs = plt.subplots(1,2,figsize=(16,7))
    axs[0].set_title('Loss')
    axs[0].plot(history[1], label='Train')
    axs[0].plot(history[3], label='Test')
    axs[0].legend()
    axs[0].grid()

    axs[1].set_title('Accuracy')
    axs[1].plot(history[0], label='Train')
    axs[1].plot(history[2], label='Test')
    axs[1].legend()
    axs[1].grid()
    plt.show()

def gradcam_vis(model,target_layers, misclf, classes,plot_size=(4,5)):
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    rows, cols = plot_size
    figure = plt.figure(figsize=(cols*3,rows*3))

    for i in range(1, cols * rows + 1):
        img_model, img, label, pred = misclf[i-1]
        targets = [ClassifierOutputTarget(label)]
        grayscale_cam = cam(input_tensor=img_model.unsqueeze(0), targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        figure.add_subplot(rows, cols, i) # adding sub plot
        plt.title(f"Pred label: {classes[pred]}\n True label: {classes[label]}") # title of plot
        plt.axis("off") # hiding the axis
        plt.imshow(img) # showing the plot
        plt.imshow(cam_image)

    plt.tight_layout()
    plt.show()