
import cv2
import random
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms import v2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vgg16

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
torch.cuda.empty_cache()

print(f"Using {device} device")
learning_rate = 0.01
n_epochs = 4


nb_classes = 8
batch_size=32
nb_filters = 32       # number of convolutional filters to use
kernel_size = (3, 3)  # convolution kernel size
pool_size = (2, 2)    # size of pooling area for max pooling




class CustomImageDataset2(Dataset):
    def __init__(self, annotations_file, img_paths, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, usecols=['ID','CLASS'])
        self.img_paths = img_paths
        self.transform = transform


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels.iloc[idx, 1]-1
        

        id = self.img_labels.iloc[idx, 0]
        img_path = self.img_paths+id+'.jpg'
        image = read_image(img_path)
        if image.shape[0] == 4:
            image = image[:3]
        
        if self.transform:
            image = self.transform(image)
            
       

        sample = {'image': image, 'label': label}
        return image,label

transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((120, 120)),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(30),




])


csv_file = 'finalDataUpSampled.csv'
dataset_train = CustomImageDataset2(csv_file, 'Train/Train/', transform=transform)

print('dataset initialized')
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
print(len(dataloader_train))

class CustomImageDatasetTest(Dataset):
    def __init__(self, annotations_file, img_paths, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, usecols=['ID'])
        self.img_paths = img_paths
        self.transform = transform



    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):    
        id = self.img_labels.iloc[idx, 0]
        img_path = self.img_paths+id+'.jpg'
        image = image = read_image(img_path)

        if image.shape[0] == 4:
            image = image[:3]
    
        if self.transform:
            #print('Augmenting')
 
            image = self.transform(image)
        return image,id

test_transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((120, 120)),



])

#Dataloader for Test 
csv_file = 'finalData.csv'
dataset_test = CustomImageDatasetTest(csv_file, 'Test/Test/', transform=test_transform)
print('dataset initialized')
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
print(len(dataloader_test))



#Custom CNN implementation
class MelanomaCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(MelanomaCNN, self).__init__()

        self.conv_layers = torch.nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Dropout2d(0.25),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(0.25),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(0.25),
        nn.Flatten(),
        nn.Linear(25088, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.5),
        nn.Linear(1024, 8),
        nn.Softmax(dim=1)
    )








    def forward(self, x):
        x = self.conv_layers(x)
        return x


# Using VGG model
model= MelanomaCNN()

model=model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def cnn_accuracy(predict,labels):
  predict = np.array(predict)
  accuracy = np.mean(predict == labels)
  return accuracy

def vector_to_class(x):
    y = torch.argmax(nn.Softmax(dim=1)(x),axis=1)
    return y

train_losses=[]


for epoch in range(0,n_epochs):
    print(epoch)
    train_loss=0.0
    all_labels = []
    all_predicted = []
    all_epoch_labels=[]
    all_epoch_predicted=[]
    for batch_idx, (imgs, labels) in enumerate(dataloader_train):
        print(batch_idx)
        imgs =imgs.to(device)
        print(imgs.shape)
        labels=labels.to(device)
        # pass the samples through the network
        predict = model(imgs) # FILL IN STUDENT
        # apply loss function
        loss = criterion(predict, labels) # FILL IN STUDENT
        # set the gradients back to 0
        optimizer.zero_grad() # FILL IN STUDENT
        # backpropagation
        loss.backward() # FILL IN STUDENT
        # parameter update
        optimizer.step() # FILL IN STUDENT
        # compute the train loss
        train_loss += loss.item()
        # store labels and class predictions
        all_labels.extend(labels.tolist())
        all_predicted.extend(vector_to_class(predict).tolist())
        print(vector_to_class(predict).tolist())

    print('Epoch:{} Train Loss:{:.4f}'.format(epoch,train_loss/len(dataloader_train.dataset)))
    print('Accuracy:{:.4f}'.format(cnn_accuracy(np.array(all_predicted),np.array(all_labels))))



# Calculate accuracy on the training set and the test set

test_prediction = []
test_id=[]
for batch_idx, (imgs, id) in enumerate(dataloader_test):
    imgs=imgs.to(device)
    id=id.to(device)
    print(batch_idx)
    predict = model(imgs)

    test_id.extend(id)
    test_prediction.extend(vector_to_class(predict).tolist())


test_results = pd.DataFrame({
    'ID': test_id,
    'Prediction': test_prediction
})
test_results.to_csv('test_results.csv', index=False)







    


