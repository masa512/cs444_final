import torch
import torch.nn as nn
import numpy as np
# ----------------- U Net Main ---------------

class UNet(nn.Module):
    """ Unet network 

    Class attributes:
    __init__ -- Constructor for layer
    forward -- Forward pass through the model
    """

    def __init__(self, in_channels:int = 3, num_labels:int = 1, base_num_filters:int = 32):
        """ Constructor for UNet

        Keyword arguments:
        in_channels -- Number of channels in input image (Default 3 for RGB)
        num_labels -- Number of labels in the output image (Default 1 for binary)
        base_num_filters -- Number of intiial filer after input conv (Default 32)

        Returns: 
        None
        """

        super(UNet).__init__()

# -------------- U Net Blocks --------------

class DoubleConv(nn.Module):
    """DoubleConv layer inherited from nn.Module

    Class attributes:
    __init__ -- Constructor for layer
    forward -- Forward pass through the model
    """

    def __init__(self, in_channels:int = 1, out_channels:int = 1, kernel_size:int = 3) -> None:
        """ Constructor for DoubleConv

        Keyword arguments:
        in_channels -- Number of filters for input (default 1)
        out_channels -- Number of filters for output (default 1)

        Returns:
        None
        """
        
        super(DoubleConv).__init__()

        # Double convolution defined with nn.Sequential of ordered layer dictionary
        self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size = kernel_size, padding = 'same'),
                nn.BatchNorm2d(num_features = out_channels),
                nn.ReLU(inplace = True),
                nn.Conv2d(in_channels = out_channels, out_channels=out_channels, kernel_size = kernel_size, padding = 'same'),
                nn.BatchNorm2d(num_features = out_channels),
                nn.ReLU(inplace = True)
        )
        
        print(self.double_conv)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward operation

        Keyword arguments:
        x -- Input Torch tensor (default None)

        Returns:
        y -- Layer output
        """

        y = self.double_conv(x)
        return y

class OutConv(nn.Module):
    """ Output convolution layer

    Class attributes:
    __init__ -- Constructor for layer
    forward -- Forward pass through the model
    """

    def __init__(self, in_channels = 1, out_channels = 1, kernel_size = 3):
        """ Constructor for OutConv

        Keyword arguments:
        in_channels -- Number of filters for input (default 1)
        out_channels -- Number of filters for output (default 1)
        kernel_size -- Convolution kernel size (default 3)

        Returns:
        None
        """

        self.out_conv = nn.Sequential(
                nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size = kernel_size, padding = 'same'),
                nn.sigmoid()
        )

    
    def forward(self, x: torch.Tensor = None, x_skip: torch.Tensor = None):
        """ Forward operation for Outconv

        Keyword arguments:
        x -- Input image for forward propagation (default None)
        x_skip -- Input image for skip connection which can be set to None if unwanted (default None)

        Returns:
        y -- Output pixel-wise probability 
        """


        # Apply out conv 
        y = self.out_conv(x)

        return y

class Decoder(nn.Module):
    """ Decoder operation (Double conv followed by whichever upsampling operation needed)

    Class attributes:
    __init__ -- Constructor for layer
    forward -- Forward pass through the model
    """

    def __init__(self, in_channels:int = 1, out_channels:int = 1, kernel_size:int = 3, mode = 'bilinear'):
        """ Constructor for Decoder
        
        Keyword arguments 
        in_channels -- Number of filters for input (default 1)
        out_channels -- Number of filters for output (default 1)
        kernel_size -- Convolution kernel size (default 3)
        mode -- Mode used for upsampling (default 'bilinear')
        
        Returns
        None
        """
        super(Decoder).__init__()

        # Double conv block
        self.double_conv = DoubleConv(in_channels = in_channels, out_channels=out_channels, kernel_size = kernel_size)
        
        # Conditional assignment

        if mode == "conv":
            self.up_sampler = nn.Sequential(nn.ConvTranspose2d(kernel_size = 2,
                                                 in_channels = in_channels,
                                                 out_channels = out_channels,
                                                 stride = 2
                    ))
        
        elif mode == "bilinear":
            self.up_sampler = nn.Sequential(
                    nn.Upsample(scale_factor = 2, mode = mode),
                    nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1)
            )
        
        else:
            raise ValueError('Wrong modes', mode)
        
    
    def forward(self, x:torch.Tensor = None, x_skip:torch.Tensor = None):
        """ forward for the decoder block

        Keyword arguments
        x -- Input tensor (default None)
        x_skip -- Input tensor from skip connection

        Returns
        y -- Ouput from the decoder
        """

        # Upsampling block
        x = self.up_sampler(x)

        # Skip connection only when needed
        if x_skip is not None:
            # evaluate size difference in x and y and pad by half the difference each side
            del_x = abs(x.size()[-1]-x_skip.size()[-1]) ; del_y = abs(x.size()[-2]-x_skip.size()[-2]) 
            
            # padding
            x_skip = nn.functional.pad(x_skip, [del_x // 2, del_x - del_x // 2, del_y // 2, del_y - del_y // 2])

            # Concatenate two feature maps
            x = torch.cat([x,x_skip],dim=1)

        # Double Conv    
        y = self.double_conv(x)

        return y

class Encoder(nn.Module):
    """ Encoder implementation 

    Class attributes:
    __init__ -- Constructor for layer
    forward -- Forward pass through the model
    """

    def __init__(self, in_channels:int = 1, out_channels:int = 1, kernel_size:int = 3, mode:str = 'max'):
        """ Constructor for Encoder

        Keyword arguments
        in_channels -- Number of filters for input (default 1)
        out_channels -- Number of filters for output (default 1)
        kernel_size -- Convolution kernel size (default 3)

        Returns -- None
        """

        super(Encoder).__init__()

        # Double Conv
        self.double_conv = DoubleConv(in_channels = in_channels, out_channels=out_channels, kernel_size = kernel_size)

        # Downsampling using conditional assignment
        if mode == 'max':
            self.pool = nn.Sequential(
                    nn.MaxPool2d(kernel_size = 2, stride = 2)
            )
        
        elif mode == 'avg':
            self.pool = nn.Sequential(
                    nn.AvgPool2d(kernel_size = 2, stride = 2)
            )
        
        else:
            raise ValueError('Invalid mode for pooling', mode)

    def forward(self, x):
        """ Forward propagation through encoder

        Keyword argument
        x -- Input feature for the encoder
        
        Returns 
    
        outputs -- Dictionary consisting of The full output and Intermediate 
                   output feature map before pooling used for skip connection
        """
        
        # Double convolution
        x_skip = self.double_conv(x)
        
        # Pooling
        y = self.pool(x_skip)

        # Define dictionary for output
        outputs = {"y": y, "skip":x_skip}

        return outputs
        
class BottleNeck(nn.Module):

    """ BottleNeck block for bottom of U-Net

    Class attributes
    __init__ -- Constructor for layer
    forward -- Forward pass through the model
    """

    def __init__(self, in_channels:int = 1, out_channels:int=1, kernel_size:int = 3):
        """ Constructor for bottleNeck block

        Keyword arguments
        in_channels -- Number of filters for input (default 1)
        out_channels -- Number of filters for output (default 1)
        kernel_size -- Convolution kernel size (default 3)
        """

        super(BottleNeck).__init__()

        # Double conv
        self.double_conv = DoubleConv(in_channels = in_channels, out_channels=out_channels, kernel_size = kernel_size)

    def forward(self, x):
        """
        Keyword argument
        x -- Input feature for the encoder
        
        Returns
        y -- Output from the BottleNeck into the decoder
        """

        y = self.double_conv(x)

        return y