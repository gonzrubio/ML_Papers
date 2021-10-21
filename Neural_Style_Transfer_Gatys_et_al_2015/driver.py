"""

Paper: 

Created on Wed Oct 20 13:38:44 2021

@author: gonzo
"""

import torch
import torch.nn as nn


# initialize pretrained vgg19 or larger
# remove fc layers and freeze conv layers
# data: 3 images (random for now), image sc them
# Get output from all conv_x-1 layers for all three inputs
# put all 3xnumb_conv_x-1 layers outputs in the cost function
# define loss function and update pixel values?

