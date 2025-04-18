# CMU 16-726 Learning-Based Image Synthesis / Spring 2025, Final Project
# The code base is based on the code in HW3

import torch
import torch.nn as nn


def up_conv(in_channels, out_channels, kernel_size, stride=1, padding=1,
            scale_factor=2, norm='batch', activ=None):
    """Create a transposed-convolutional layer, with optional normalization."""
    layers = []
    layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
    layers.append(nn.Conv2d(
        in_channels, out_channels,
        kernel_size, stride, padding, bias=norm is None
    ))
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1,
         norm='batch', init_zero_weights=False, activ=None):
    """Create a convolutional layer, with optional normalization."""
    layers = []
    conv_layer = nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        bias=norm is None
    )
    if init_zero_weights:
        conv_layer.weight.data = 0.001 * torch.randn(
            out_channels, in_channels, kernel_size, kernel_size
        )
    layers.append(conv_layer)

    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


# takes in image, returns 1 hot vector of the style in the image
class StyleIdentifier(nn.Module):
    # assume image is 128*128
    def __init__(self, conv_dim=64, init_zero_weights=False, norm='instance',num_classes = 31):
        super().__init__()
        # It's going to be something similar to the paper/hw3 but not the same
        # start off w/ conv like hw 3
        # then blocks of resblocks w/ avg pools that halve the image size to 4x4
        # finally linear + relu to get a 1 hot vector

        # # 1. Conv
        self.conv1 = conv(3, conv_dim, 4, norm = norm, activ='relu',init_zero_weights = init_zero_weights) # 128 -> 64
        self.conv2 = conv(conv_dim, conv_dim*2, 4, norm = norm, activ='relu',init_zero_weights = init_zero_weights) # 64 -> 32
        
        # 2. Resenet Block
        self.resnet_block = nn.Sequential(
            ResnetBlock(conv_dim*2, norm='instance', activ='relu'),
            nn.AveragePool2d(4, 2, 1), # 32 -> 16
            ResnetBlock(conv_dim*2, norm='instance', activ='relu'),
            nn.AveragePool2d(4, 2, 1), # 16 -> 8
            ResnetBlock(conv_dim*2, norm='instance', activ='relu'),
            nn.AveragePool2d(4, 2, 1), # 8 -> 4
        )

        # 3. Linear to num_classes
        self.conv3 = conv(conv_dim*2, conv_dim*2, 4, 2, 1, norm=norm, init_zero_weights=init_zero_weights, activ='leaky') # 4 -> 1
        
        self.lin1 = nn.Linear(conv_dim * 2, num_classes)  # Final layer for classification
        self.relu = nn.Relu()  

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.resnet_block(x)
        x = self.conv3(x)
        # x = x.view(-1, 1)  # Flatten for linear layer
        x.squeeze() # maybe this works better
        x = self.lin1(x)  
        x = self.relu(x)
        return x.squeeze()

# takes in the hot vector, tries to generate latent s.t generator will produce the input training image
# also used for cycle loss: get latent of generated image, pass in w/ original style, compare loss there 
class Generator(nn.Module):
    # assume image is 128*128
    # encoder + generator, based off of cycle generator
    def __init__(self, conv_dim=64, init_zero_weights=False, norm='instance',num_classes = 31):
        super().__init__() 
        # 1. Define the encoder part of the generator
        # img to latent space
        self.conv1 = conv(3, conv_dim, 4, norm = norm, activ='relu',init_zero_weights = init_zero_weights) # 128 -> 64
        self.conv2 = conv(conv_dim, conv_dim*2, 4, norm = norm, activ='relu',init_zero_weights = init_zero_weights) # 64 -> 32
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4, norm = norm, activ='relu',init_zero_weights = init_zero_weights) # 32 -> 16

        # 2. Define the transformation part of the generator
        # want it to take in the 1 hot vector to transform it
        self.resnet_block = nn.Sequential(
            ResnetBlock(conv_dim*4 + num_classes, norm='instance', activ='relu'), 
            ResnetBlock(conv_dim*4 + num_classes, norm='instance', activ='relu'), 
            ResnetBlock(conv_dim*4 + num_classes, norm='instance', activ='relu'), 
        )

        # 3. Define the decoder part of the generator
        # want it to take in the 1 hot vector too
        self.up_conv1 = up_conv(conv_dim*4 + num_classes, conv_dim*2, 3, norm = norm, activ='relu') 
        self.up_conv2 = up_conv(conv_dim*2 + num_classes, conv_dim, 3, norm = norm, activ='relu') 
        self.up_conv3 = up_conv(conv_dim + num_classes, 3, 3, norm = None, activ='tanh') 

    # assumes hotv of dim: batch * num_classes
    def forward(self, x, hotv):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        style = hotv.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2),x.size(3))
        x = torch.cat((x, style), dim=1)
        x = self.resnet_block(x) 

        style = hotv.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2),x.size(3))
        x = torch.cat((x, style), dim=1)
        x = self.up_conv1(x)

        style = hotv.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2),x.size(3))
        x = torch.cat((x, style), dim=1)
        x = self.up_conv2(x)
        
        style = hotv.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2),x.size(3))
        x = torch.cat((x, style), dim=1)
        x = self.up_conv3(x)
        return x.squeeze()

class Discriminator(nn.Module):
    # takes in image (128x128), identifies if it's real/fake
    def __init__(self, conv_dim=64, norm='instance'):
        super().__init__() 
        self.conv1 = conv(3, conv_dim, 4, 2, 1, norm, False, 'relu') #128 -> 64
        self.conv2 = conv(conv_dim, conv_dim*2, 4, 2, 1, norm, False, 'relu') # 64->32
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4, 2, 1, norm, False, 'relu') # 32->16
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4, 2, 1, norm, False, 'relu') # 16->8
        self.conv5 = conv(conv_dim*8, 1, 4,1, 0, None, False, None) # 8->4
        # based on these dim, its a patch discriminator 

    def forward(self, x):
        """Forward pass, x is (B, C, H, W)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.squeeze()

# class DCGenerator(nn.Module):

#     def __init__(self, noise_size, conv_dim=64):
#         super().__init__()
#         ###########################################
#         ##   FILL THIS IN: CREATE ARCHITECTURE   ##
#         ########################################### 
#         # For the first layer (up_conv1) it is better to directly apply convolution layer without any upsampling to get 4x4 output
#         # To do so, you’ll need to think about what you kernel and padding size should be in this case. 
#         self.up_conv1 = conv(noise_size, conv_dim*8, 4,1,3, norm='instance', activ='relu')
#         self.up_conv2 = up_conv(conv_dim*8, conv_dim*4, 3, norm='instance', activ='relu')
#         self.up_conv3 = up_conv(conv_dim*4, conv_dim*2, 3, norm='instance', activ='relu')
#         self.up_conv4 = up_conv(conv_dim*2, conv_dim, 3, norm='instance', activ='relu')
#         self.up_conv5 = up_conv(conv_dim, 3, 3, norm = None,activ='tanh') 

#     def forward(self, z):
#         """
#         Generate an image given a sample of random noise.

#         Input
#         -----
#             z: BS x noise_size x 1 x 1   -->  16x100x1x1

#         Output
#         ------
#             out: BS x channels x image_width x image_height  -->  16x3x64x64
#         """
#         ###########################################
#         ##   FILL THIS IN: FORWARD PASS   ##
#         ###########################################
#         z = self.up_conv1(z)
#         z = self.up_conv2(z)
#         z = self.up_conv3(z)
#         z = self.up_conv4(z)
#         z = self.up_conv5(z)
#         return z.squeeze()


class ResnetBlock(nn.Module):

    def __init__(self, conv_dim, norm, activ):
        super().__init__()
        self.conv_layer = conv(
            in_channels=conv_dim, out_channels=conv_dim,
            kernel_size=3, stride=1, padding=1, norm=norm,
            activ=activ
        )

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


# class CycleGenerator(nn.Module):
#     """Architecture of the generator network."""

#     def __init__(self, conv_dim=64, init_zero_weights=False, norm='instance'):
#         super().__init__()

#         ###########################################
#         ##   FILL THIS IN: CREATE ARCHITECTURE   ##
#         ###########################################

#         # 1. Define the encoder part of the generator
#         self.conv1 = conv(3, conv_dim, 4, norm = norm, activ='relu',init_zero_weights = init_zero_weights)
#         self.conv2 = conv(conv_dim, conv_dim*2, 4, norm = norm, activ='relu',init_zero_weights = init_zero_weights)

#         # 2. Define the transformation part of the generator
#         self.resnet_block = ResnetBlock(conv_dim*2, norm=norm, activ='relu')

#         # 3. Define the decoder part of the generator
#         self.up_conv1 = up_conv(conv_dim*2, conv_dim, 3, norm = norm, activ='relu') 
#         self.up_conv2 = up_conv(conv_dim, 3, 3, norm = None, activ='tanh')

#     def forward(self, x):
#         """
#         Generate an image conditioned on an input image.

#         Input
#         -----
#             x: BS x 3 x 32 x 32

#         Output
#         ------
#             out: BS x 3 x 32 x 32
#         """
#         ###########################################
#         ##   FILL THIS IN: FORWARD PASS   ##
#         ###########################################
#         x = self.conv1(x)
#         x = self.conv2(x)
        
#         x = self.resnet_block(x)
        
#         x = self.up_conv1(x)
#         x = self.up_conv2(x)
#         return x


# class DCDiscriminator(nn.Module):
#     """Architecture of the discriminator network."""

#     def __init__(self, conv_dim=64, norm='instance'):
#         super().__init__() 
#         self.conv1 = conv(3, conv_dim, 4, 2, 1, norm, False, 'relu')
#         self.conv2 = conv(conv_dim, conv_dim*2, 4, 2, 1, norm, False, 'relu')
#         self.conv3 = conv(conv_dim*2, conv_dim*4, 4, 2, 1, norm, False, 'relu')
#         self.conv4 = conv(conv_dim*4, conv_dim*8, 4, 2, 1, norm, False, 'relu')
#         self.conv5 = conv(conv_dim*8, 1, 4,1, 0, None, False, None)

#     def forward(self, x):
#         """Forward pass, x is (B, C, H, W)."""
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         return x.squeeze()


# class PatchDiscriminator(nn.Module):
#     """Architecture of the patch discriminator network."""

#     def __init__(self, conv_dim=64, norm='batch'):
#         super().__init__()

#         ###########################################
#         ##   FILL THIS IN: CREATE ARCHITECTURE   ##
#         ###########################################
#         self.conv1 = conv(3, conv_dim, 4, 2, 1, norm, False, 'relu')
#         self.conv2 = conv(conv_dim, conv_dim*2, 4, 2, 1, norm, False, 'relu')
#         self.conv3 = conv(conv_dim*2, conv_dim*4, 4, 2, 1, norm, False, 'relu')
#         # self.conv4 = conv(conv_dim*4, conv_dim*8, 4, 2, 1, norm, False, 'relu')
#         self.conv5 = conv(conv_dim*4,4,2,1, 0, None, False, None) # want to merge to 1x4x4

#         # Hint: it should look really similar to DCDiscriminator.


#     def forward(self, x):

#         ###########################################
#         ##   FILL THIS IN: FORWARD PASS   ##
#         ###########################################
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         # x = self.conv4(x)
#         x = self.conv5(x)
#         return x.squeeze()


if __name__ == "__main__":
    a = torch.rand(4, 3, 128, 128)
    D = Discriminator()
    print(D(a).shape)
    G = Generator()
    print(G(a).shape)
