import brainchild.loader as bcl
import torch.nn as N
import torch.nn.functional as F

class reg_net(N.module):
    '''Neural network architecture for automatic brain image registration.
    Based on method proposed in the paper:
    Li, Hongming, and Yong Fan.
    "Non-rigid image registration using fully convolutional networks with deep self-supervision."
    arXiv preprint arXiv:1709.00799 (2017).

    '''
    def __init__ (self):
        '''Construct the registration network
        Args:
            n_channels:
                The number of channels in the input.
        '''
        conv1 = N.Conv3D(kernel_size=3,stride=2)
        pool1 = N.AdaptiveMaxPool3d(output_size=68)
        conv2 = N.Conv3D(kernel_size=3,stride=2)
        pool2 = N.AdaptiveMaxPool3d(output_size=34)
        conv3 = N.Conv3D(kernel_size=3,stride=2)
        conv4 = N.Conv3D(kernel_size=3,stride=2)
        regr1 = N.Conv3D()

class conv_block(N.module):

    def __init__(self, c_in, c_out, k, s):
        self. c
