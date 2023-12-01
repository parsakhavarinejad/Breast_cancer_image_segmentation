import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import random

import glob
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")

from utils import plot_train_label, CustomImageMaskDataset, bce_dice_loss, plot_metrics, plot_subplots
from model import Unet
from utils import Trainer

masks = glob.glob("data/Dataset_BUSI_with_GT/*/*_mask.png")
images = [mask_images.replace("_mask", "") for mask_images in masks]
series = list(zip(images, masks))

random_image = random.sample(range(200), 7)
for image in random_image:
    plot_train_label(series[image])

dataset = pd.DataFrame(series, columns=['image_path', 'mask_path'])
train, test= train_test_split(dataset, test_size=0.2)
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")


# Define transformations
train_transforms = transforms.Compose([
    transforms.Resize([240, 240]),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.Resize([240, 240]),
    transforms.ToTensor()
])

# Create datasets
train_dataset = CustomImageMaskDataset(train, train_transforms)
test_dataset = CustomImageMaskDataset(test, val_transforms)

# Create DataLoaders
batch_size = 4
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for batch in train_dataloader:
    # Assuming your dataset returns a tuple (inputs, targets)
    inputs, targets = batch

    # Print the shapes
    print("Input shape:", inputs.shape)
    print("Target shape:", targets.shape)

    print(targets[0].shape)
    # Break the loop after printing the shapes of the first batch
    break

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

unet = Unet().to(device)
learning_rate = 0.0001
weight_decay = 1e-4

optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate, weight_decay=weight_decay)

trainer = Trainer(model=unet, num_epochs=80, optimizer=optimizer, criterion=bce_dice_loss, device=device)

trainer.train(train_dataloader, test_dataloader)
metrics = trainer.get_metrics()

plot_metrics(metrics)

for i in [2, 3, 10, 20, 55, 66, 87, 98]:
    image = train_dataset[i][0]
    mask = train_dataset[i][1]
    im = image.to(device)
    pred = unet(im.unsqueeze(0))
    pred = pred.squeeze()

    plot_subplots(im, mask, pred)