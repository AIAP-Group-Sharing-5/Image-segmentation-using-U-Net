
<div style="text-align: center;">
<h1>Image-Segmentation-using-U-Net</h1>
</div>







## 2. Overview of U-Net Architecture

### 2.1. Review of key concepts used in U-net’s architecture


<br />Before going into the network’s architecture, let’s have a quick review on a few critical concepts.

1. Convolutional Layer

    The Convolutional layer is a key building block in a Convolutional Neural Network (ConvNet). The convolution process takes in an input data of dimensions (C * H * W) and outputs a “feature map” of dimensions (F * H’ * W’), whereby F is the number of filters (or kernels) used in the convolution and H’ and W’ are the corresponding Height and Width of the output “feature map”.

    H’ and W’ can be calculated by the following formula:


    <div style="text-align: center;">
    
    ![Formula for Height of output feature map](/markdown%20images/Convolution1.png)


    ![Formula for Width of output feature map](/markdown%20images/Convolution2.png)
    </div>

    assuming padding of P, stride of s and filter size of k * k * C.

    For every stride in a convolution, the dot product of the elements within the filter and the corresponding elements in the cropped portion of the input data is calculated. A visual representation is as shown below:

    <div style="text-align: center;">


    ![Visual representation of Convolution step](/markdown%20images/Convolution3.png)
    
    [Image Source](https://anhreynolds.com/blogs/cnn.html)
    </div>

    Within the Convolutional Layer, it is common to have Batch normalization (not used in the U-Net) and an activation function before the feature map (Output) is produced.



2. Pooling Layers

    There are two common Pooling Layers found in ConvNet, Maximum Pooling and Average Pooling.

    Pooling Layers work in the same way as Convolutional Layers in terms of the Input and Output dimensions, with the difference being in the output elements.

    A visual representation of these Pooling layers of padding = 0, stride = 2 and filter size of 2 * 2  is as such:

    <div style="text-align: center;">


   ![Visual representation of Pooling Layer](/markdown%20images/Pooling1.png)
    
    [Image Source](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
    </div>

    In Max Pooling, the element with the maximum value of the cropped portion from the input data is taken, while in Average Pooling, the average of the cropped portion is taken.

3. Upsampling/Transpose Convolution

    Transpose convolution is a method used to increase the spatial dimensions of the output feature map compared to the input. 

    A visual representation of a transpose convolution with a 2 * 2 input and a 2 * 2 filter with padding = 0 and stride = 1 is as such:

    <div style="text-align: center;">


   ![Visual representation of Upsampling](/markdown%20images/TC1.png)
    
    [Image Source](https://mriquestions.com/up-convolution.html)
    </div>

    This method is commonly used in ConvNet with an encoder-decoder architecture to re-expand the resolution of the “image” after the initial downsampling done in the encoder portion of the network.

    The output spatial dimensions can be calculated by the following formula:
    <div style="text-align: center;">


    ![Formula for Height and Width of Output feature map post Transpose Convolution](/markdown%20images/TC2.png)
    </div>

4. Skip Connections

    Skip Connections, also known as shortcuts, are used to pass the outputs of one layer, to a layer further down the ConvNet, skipping the next layer (or more).

    A visual representation of a skip connection of one layer is as such:
    <div style="text-align: center;">


    ![Visual representation of Skip Connections](/markdown%20images/SC1.png)
    
    [Image Source](https://tiefenauer.github.io/ml/deep-learning/4)
    </div>

    The output activation at Layer l, is passed directly to Layer l+2, pre activation. The concatenated result then goes through Layer l+2’s activation.

<br>

### 2.2. U-Net's Architecture

<br />
<div style="text-align: center;">

![Visual representation of U-Net's Architecture](/markdown%20images/U-net.png)
    
[Image Source](https://arxiv.org/abs/1505.04597)
</div>

<br>

<br />

> Excerpt from U-Net: Convolutional Networks for Biomedical Image Segmentation
> 
> *"It consists of a contracting
path (left side) and an expansive path (right side). The contracting path follows
the typical architecture of a convolutional network. It consists of the repeated
application of two 3x3 convolutions (unpadded convolutions), each followed by
a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2
for downsampling. At each downsampling step we double the number of feature
channels. Every step in the expansive path consists of an upsampling of the
feature map followed by a 2x2 convolution (“up-convolution”) that halves the
number of feature channels, a concatenation with the correspondingly cropped
feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU.* 
>
>*At the final layer a 1x1 convolution is used to map each 64-
component feature vector to the desired number of classes. In total the network
has 23 convolutional layers."*

<br>

As stated in the excerpt, the architecture is made up of an encoder (contracting) and decoder (expansive), with no fully connected layers present.

#### **Encoder**

In the encoding layers, it consists of four repetitions of these:

- 3*3 valid convolutions (stride = 1, with ReLU activation)
- 3*3 valid convolutions (stride = 1, with ReLU activation)
- 2*2 Max pooling (stride = 2, effectively halving the output feature map)
- Doubling of filter channels per downsampling repetition (for every first 3*3 convolution, after the first downsampling repetition)

#### **Decoder**

In the decoding layers, it consists of four repetitions of these:

- 2*2 Transpose convolution (stride = 2)
- Concatenation of feature channels from the corresponding contracting path (skip connections)
- 3*3 valid convolutions (stride = 1, with ReLU activation)
- 3*3 valid convolutions (stride = 1, with ReLU activation)

#### **Output Layer**
A 1*1 convolution was used to reduce the feature space from 64 channels to the desired number of classes. A softmax activation and cross entropy loss was then applied pixel-wise.

<br />

## 3. Code Implementation (using Pytorch)

Let's take a look at how we can implement an image segmentation model using the Pytorch framework and a pre-trained VGG-19 model.

The image dataset is taken from....

### 3.1. Import relevant libraries

```py
#Necessary Pytorch dependencies
import torch
from torchvision.transforms import ToTensor
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset , Subset, random_split

#For Image Augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

#For image viewing and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import sys

#Importing Utility script
from src import utils #
```

### 3.2. Set a seed for reproducibility
```py
torch.manual_seed(42)
np.random.seed(42)
```

### 3.3. Loading Image from Folder
```py
image_path = os.listdir(os.path.join('archive', 'Image'))
mask_path = os.listdir(os.path.join('archive', 'Mask'))
```


### 3.4. Creating a Train-Test Split
```py
train_indices = np.random.choice(len(image_path), size=int(0.8*len(image_path)), replace=False)
test_indices = np.array(list(set(range(len(image_path))) - set(train_indices)))
train_image_path = np.take(image_path, train_indices)
train_mask_path = np.take(mask_path, train_indices)
test_image_path = np.take(image_path, test_indices)
test_mask_path = np.take(mask_path, test_indices)

train_image = [Image.open(os.path.join('archive', 'Image', i)) for i in train_image_path]
train_mask = [Image.open(os.path.join('archive', 'Mask', i)) for i in train_mask_path]
test_image = [Image.open(os.path.join('archive', 'Image', i)) for i in test_image_path]
test_mask = [Image.open(os.path.join('archive', 'Mask', i)) for i in test_mask_path]
```

### 3.5. Changing all images to "RGB" mode
```py
for i in range(len(train_image)):
    train_image[i] = train_image[i].convert('RGB')
for i in range(len(test_image)):
    test_image[i] = test_image[i].convert('RGB')
```

### 3.6. Data Augmentation
```py
Transforms = A.Compose([
    A.Resize(224,224),
    A.HorizontalFlip(p=0.75),
    A.VerticalFlip(p=0.75),
    # A.GridDistortion(p=0.75),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1),
    # A.PixelDropout(dropout_prob=0.5),
    # A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.5),
    ToTensorV2(),
])

base = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2(),
])
```
### 3.7. Creating the U-Net model using VGG-19
\
**Encoder Block**
```py
import torchvision.models as models


class VGG19UNet(nn.Module):
    def __init__(self, num_classes=35):
        super(VGG19UNet, self).__init__()
        self.features = models.vgg19(weights="VGG19_Weights.DEFAULT").features
        assert num_classes > 0 and isinstance(num_classes, int)
        if num_classes == 2:
            self.num_classes = 1
        else:
            self.num_classes = num_classes
        for param in self.features.parameters():
            param.requires_grad = False

        #ENCODER
        self.layer1 = nn.Conv2d(
            in_channels=64,
            out_channels=self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )
```
\
**Decoder Block**
```py     
        self.up_sample_54 = nn.ConvTranspose2d(512, 512, 2, stride=2)

        self.up_sample_43 = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.up_sample_32 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.up_sample_21 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 35 classes
```

\
**Forward Pass**
```py     
def forward(self, x):
        # Encoder path
        X1 = self.features[:4](x)
        # 64
        X2 = self.features[4:9](X1)
        # 128
        X3 = self.features[9:18](X2)
        # 256
        X4 = self.features[18:27](X3)
        # 512
        X5 = self.features[27:-1](X4)

        # Decoder path
        X = self.up_sample_54(X5)
        X4 = torch.cat([X, X4], dim=1)
        X4 = self.layer5(X4)

        X = self.up_sample_43(X4)
        X3 = torch.cat([X, X3], dim=1)
        X3 = self.layer4(X3)

        X = self.up_sample_32(X3)
        X2 = torch.cat([X, X2], dim=1)
        X2 = self.layer3(X2)

        X = self.up_sample_21(X2)
        X1 = torch.cat([X, X1], dim=1)
        X1 = self.layer2(X1)

        X = self.layer1(X1)

        return X
```


### 3.8. Creating a custom data set using Pytorch's Dataset Class
```py
class CustomFloodDataset(Dataset):
    """
    Custom dataset for flood segmentation

    Parameters
    ----------
    Dataset : torch.utils.data.Dataset
        Dataset class from torch.utils.data
    """
    def __init__(self, image_list, mask_list, transform=None):
        self.image_list = image_list
        self.mask_list = mask_list
        self.transform = transform

    def __getitem__(self, idx):
        image = self.image_list[idx]
        mask = self.mask_list[idx]
        image = np.array(image)
        mask = np.array(mask)
        
        # mask = np.reshape(mask, mask.shape + (1,))
        if self.transform:
            augment = self.transform(image=image, mask=mask)
            image = augment['image']
            mask = augment['mask'] / 255
        return image, mask
    
    def __len__(self):
        return len(self.image_list)

#Carrying out the data augmentation to the train and test set defined in 3.6
train_dataset = CustomFloodDataset(train_image, train_mask, transform=Transforms) 
test_dataset = CustomFloodDataset(test_image, test_mask, transform=base) 
```

### 3.9. Setting up a DataLoader
```py
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
```

### 3.10. Setting up the model for training
```py
sys.path.append('..')
from src.vgg import VGG19UNet
Unet = VGG19UNet(num_classes=1).to(device)

#Defining the optimizer and Loss function
optimizer = torch.optim.Adam(Unet.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()
es = utils.EarlyStopper(patience=50, verbose=True, model=Unet)
```
### 3.11. Creating Random images and training the model
```py
# empty list to store images
TO_GIF = []
# generate a random image to form into gif
random_idx = np.random.randint(0, len(test_image))
sample_image = ToTensor()(test_image[random_idx]).unsqueeze(0)
sample_image_transformed = Transforms(image=np.array(train_image[0]))['image'].unsqueeze(0).to(device)
scaler = GradScaler()

EPOCH = 200
PATIENCE = 20
es = utils.EarlyStopper(patience=PATIENCE, verbose=True,min_epoch=10, model = Unet)
import time
for epoch in range(EPOCH):
    train_loss = []
    Unet.train()

    # Training 
    t1 = time.time()
    for image, mask in train_dataloader:
        optimizer.zero_grad()
        with autocast():
            image = image.to(device)
            mask = mask.to(device)
            output = Unet(image).squeeze(1)
            loss = criterion(output.float(), mask.float())
        
        train_loss.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # Validation
    Unet.eval()
    test_loss = []
    with torch.no_grad():
        for image, mask in test_dataloader:
            with autocast():
                image = image.to(device)
                mask = mask.to(device)
                output = Unet(image).squeeze(1)
                loss = criterion(output.float(), mask.float())
            test_loss.append(loss.item())
        
        # Append to GIF
        sample_mask = Unet(sample_image_transformed)
        prob = F.sigmoid(sample_mask)
        predicted = (prob > 0.5).float()
        predicted = predicted.squeeze(0).squeeze(0).cpu().numpy()
        TO_GIF.append(predicted)
    
    t2 = time.time()
    if epoch % 1 == 0:
        print(f"Epoch: {epoch+1} / {EPOCH}, Loss: {np.mean(train_loss)}\t"
            f"Val Loss: {np.mean(test_loss)} \t Time: {t2-t1:.2f}")
    if es.early_stop(epoch, np.mean(test_loss)):
        TO_GIF = TO_GIF[:-PATIENCE]
        break


es.load_best_model(Unet)
```


### 3.12. Fine-Tuning of the model
```py
EPOCH = 100
PATIENCE = 10
for count, layer in enumerate(Unet.features):
    if count > 33:
        layer.requires_grad = True

es = utils.EarlyStopper(patience=PATIENCE, verbose=True,min_epoch=0, model = Unet)

optimizer = torch.optim.Adam(Unet.parameters(), lr=0.0001)
import time
for epoch in range(EPOCH):
    train_loss = []
    Unet.train()

    # Training 
    t1 = time.time()
    for image, mask in train_dataloader:
        optimizer.zero_grad()
        with autocast():
            image = image.to(device)
            mask = mask.to(device)
            output = Unet(image).squeeze(1)
            loss = criterion(output.float(), mask.float())
        
        train_loss.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # Validation
    Unet.eval()
    test_loss = []
    with torch.no_grad():
        for image, mask in test_dataloader:
            with autocast():
                image = image.to(device)
                mask = mask.to(device)
                output = Unet(image).squeeze(1)
                loss = criterion(output.float(), mask.float())
            test_loss.append(loss.item())
            # trying out print
        sample_mask = Unet(sample_image_transformed)
        prob = F.sigmoid(sample_mask)
        predicted = (prob > 0.5).float()
        predicted = predicted.squeeze(0).squeeze(0).cpu().numpy()
        TO_GIF.append(predicted)
    
    t2 = time.time()
    if epoch % 1 == 0:
        print(f"Epoch: {epoch+1} / {EPOCH}, Loss: {np.mean(train_loss)}\t"
            f"Val Loss: {np.mean(test_loss)} \t Time: {t2-t1:.2f}")
    if es.early_stop(epoch, np.mean(test_loss)):
        TO_GIF = TO_GIF[:-PATIENCE]
        break
```

### 3.13. Segmentation Results

Let's take a look at the results!

```py
plot_number = 4
fig, ax = plt.subplots(plot_number,3,figsize=(10,plot_number*3))
Unet.eval()
test_mask = Unet(image.to(device))
prob = F.sigmoid(test_mask)
predicted = (prob > 0.5).float()
with torch.no_grad():
    for i in range(plot_number):
        ax[i,0].imshow(image[i].cpu().permute(1,2,0))
        ax[i,0].axis('off')
        ax[i,0].set_title('Image')
        ax[i,1].imshow(mask[i].squeeze(0).cpu().numpy())
        ax[i,1].axis('off')
        ax[i,1].set_title('True Mask')
        ax[i,2].imshow(predicted.cpu().detach().numpy()[i].squeeze(0))
        ax[i,2].axis('off')
        ax[i,2].set_title('Predicted Mask')
        fig.tight_layout()
plt.show()
```

<div style="text-align: center;">


![Results of our Segmentation Model](/markdown%20images/results.jpg)
</div>

<br>
## 4. Improvements to U-net

### 1. V-net:

Uses a convolutional layer to replace the up-sampling and down-sampling pooling layer. The idea behind V-Net is that using the Maxpool operation leads to a lot of information loss thus replacing it with another series of Conv operations without padding will help in preserving more information. However this also increases the number of trainable parameters which is more computationally expensive. [9]

### 2. U-net++ 
Improves upon U-net by adding nested and dense skip connections. Dense skip connections enable every layer in the decoder to be connected to every layer in the corresponding encoder path. While nested skip connections connect layers at the same resolution in different paths of the encoder and decoder. Specifically, the output of a layer in the encoder is concatenated with the output of a layer at the same resolution in the corresponding decoder path.

### 3. Attention U-net 
Incorporates the attention mechanism in NLP, giving the skip connections an extra idea of which region to focus on while segmenting the given object.

## 5. Concluding remarks
Overall, U-Net is a powerful image segmentation method that has been widely used in various fields, especially in medical imaging. Its unique architecture and skip connections enable it to accurately segment images while retaining high resolution details. However, there are still challenges that need to be addressed, such as class imbalance, small datasets, and overfitting. Despite these limitations, U-Net remains a popular choice for image segmentation tasks and has the potential to be further improved through future research.

## References

[1] https://towardsdatascience.com/u-net-b229b32b4a71

[2] https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/#:~:text=U%2DNet%20is%20a%20semantic,such%20as%20the%20Pix2Pix%20generator

[3] https://github.com/milesial/Pytorch-UNet

[4] https://www.analyticsvidhya.com/blog/2019/07/computer-vision-implementing-mask-r-cnn-image-segmentation/

[5] https://epoch.aisingapore.org/2023/02/image-compression-with-k-means/

[6] https://www.tensorflow.org/tutorials/images/segmentation

[7] https://www.v7labs.com/blog/image-segmentation-guide

[8] https://arxiv.org/abs/1505.04597

[9] https://medium.com/aiguys/attention-u-net-resunet-many-more-65709b90ac8b
