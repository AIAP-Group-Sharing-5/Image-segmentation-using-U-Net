
<div style="text-align: center;">
<h1>Image-Segmentation-using-U-Net</h1>
</div>







## 2. Overview of U-Net Architecture

### 2.1. Review of key concepts used in U-net’s architecture


<br />Before going into the network’s architecture, let’s have a quick review on a few critical concepts.

1. Convolutional Layer

    The Convolutional layer is a key building block in a Convolutional Neural Network (ConvNet). The convolution process takes in an input data of dimensions (C * H * W) and outputs a “feature map” of dimensions (F * H’ * W’), whereby F is the number of filters (or kernels) used in the convolution and H’ and W’ are the corresponding Height and Width of the output “feature map”.

    H’ and W’ can be calculated by the following formula:


    ![Formula for Height of output feature map](/markdown%20images/Convolution1.png)


    ![Formula for Width of output feature map](/markdown%20images/Convolution2.png)

    assuming padding of P, stride of s and filter size of k * k * C.

    For every stride in a convolution, the dot product of the elements within the filter and the corresponding elements in the cropped portion of the input data is calculated. A visual representation is as shown below:


    ![Visual representation of Convolution step](/markdown%20images/Convolution3.png)
    
    [Image Source](https://anhreynolds.com/blogs/cnn.html)

    Within the Convolutional Layer, it is common to have Batch normalization (not used in the U-Net) and an activation function before the feature map (Output) is produced.



2. Pooling Layers

    There are two common Pooling Layers found in ConvNet, Maximum Pooling and Average Pooling.

    Pooling Layers work in the same way as Convolutional Layers in terms of the Input and Output dimensions, with the difference being in the output elements.

    A visual representation of these Pooling layers of padding = 0, stride = 2 and filter size of 2 * 2  is as such:

   ![Visual representation of Pooling Layer](/markdown%20images/Pooling1.png)
    
    [Image Source](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

    In Max Pooling, the element with the maximum value of the cropped portion from the input data is taken, while in Average Pooling, the average of the cropped portion is taken.

3. Upsampling/Transpose Convolution

    Transpose convolution is a method used to increase the spatial dimensions of the output feature map compared to the input. 

    A visual representation of a transpose convolution with a 2 * 2 input and a 2 * 2 filter with padding = 0 and stride = 1 is as such:

   ![Visual representation of Upsampling](/markdown%20images/TC1.png)
    
    [Image Source](https://mriquestions.com/up-convolution.html)


    This method is commonly used in ConvNet with an encoder-decoder architecture to re-expand the resolution of the “image” after the initial downsampling done in the encoder portion of the network.

    The output spatial dimensions can be calculated by the following formula:

    ![Formula for Height and Width of Output feature map post Transpose Convolution](/markdown%20images/TC2.png)

4. Skip Connections

    Skip Connections, also known as shortcuts, are used to pass the outputs of one layer, to a layer further down the ConvNet, skipping the next layer (or more).

    A visual representation of a skip connection of one layer is as such:

    ![Visual representation of Skip Connections](/markdown%20images/SC1.png)
    
    [Image Source](https://tiefenauer.github.io/ml/deep-learning/4)

    The output activation at Layer l, is passed directly to Layer l+2, pre activation. The concatenated result then goes through Layer l+2’s activation.

<br>

### 2.2. U-Net's Architecture

    