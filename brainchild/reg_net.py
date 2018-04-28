import brainchild.loader as bcl
import torch.nn as N
import torch.nn.functional as F
import torch.optim as O
from torch import autograd
import torch

class reg_net(N.Module):
    '''Neural network architecture for automatic brain image registration.
    Based on method proposed in the paper:
    Li, Hongming, and Yong Fan.
    "Non-rigid image registration using fully convolutional networks with deep self-supervision."
    arXiv preprint arXiv:1709.00799 (2017).

    '''
    def __init__(self):
        super().__init__()
        '''Construct the registration network
        '''
        self.conv1 = conv_block(c_in=1, c_out=3, k=6, s=2, pad=130)
        self.pool1 = N.MaxPool3d(2)
        self.conv2 = conv_block(c_in=3, c_out=3, k=6, s=2, pad=66)
        self.pool2 = N.MaxPool3d(2)
        self.conv3 = conv_block(c_in=3, c_out=3, k=6, s=2, pad=34)

        self.regr3 = N.Sequential(N.ConvTranspose3d(in_channels=3, out_channels=3, kernel_size=4, stride=4),
                            N.Conv3d(in_channels=3,out_channels=1, kernel_size=1, stride=1))

        self.deconv1 = deconv_block(c_in=3, c_out=3, k=2, s=2)
        self.conv4 = conv_block(c_in=3, c_out=3, k=1, s=1)

        self.regr2 = N.Sequential(N.ConvTranspose3d(in_channels=3, out_channels=3, kernel_size=2, stride=2),
                            N.Conv3d(in_channels=3,out_channels=1, kernel_size=1, stride=1))

        self.deconv2 = deconv_block(c_in=3, c_out=3, k=2, s=2)

        self.regr1 = N.Sequential(N.Conv3d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
                            N.Conv3d(in_channels=3,out_channels=1, kernel_size=1, stride=1))
    def forward(self, input):
        self.seq1 = N.Sequential(self.conv1, self.pool1, self.conv2, self.pool2, self.conv3)
        self.seq2 = N.Sequential(self.deconv1, self.conv4)

        tmp = self.seq1(input)
        print("After Conv3 size:", tmp.size())
        output3 = self.regr3(tmp)
        print("Regr3 output size:", output3.size() )

        tmp= self.seq2(tmp)
        print("After Conv4 size:", tmp.size())
        output2 = self.regr2(tmp)
        print("Regr2 output size:", output2.size() )

        tmp = self.deconv2(tmp)
        print("After DeConv2 size:", tmp.size())
        output1 = self.regr1(tmp)
        print("Regr1 output size:", output1.size() )

        output = torch.add(output3, 0.6, output2)
        output = torch.add(output, 0.3, output1)
        print("Final output size:", output.size())
        return output



class conv_block(N.Module):

    def __init__(self, c_in, c_out, k, s, pad=0):
        super().__init__()
        self.conv = N.Conv3d(in_channels= c_in, out_channels=c_out, kernel_size=k, stride=s, padding=pad)
        self.bn = N.BatchNorm3d(num_features=c_out)
        self.relu = N.ReLU()
    def forward(self, input):
        out = self.bn(self.conv(input))
        return self.relu(out)

class deconv_block(N.Module):
    def __init__(self, c_in, c_out, k, s, pad=0):
        super().__init__()
        self.conv = N.ConvTranspose3d(in_channels= c_in, out_channels=c_out, kernel_size=k, stride=s, padding=pad)
        self.bn = N.BatchNorm3d(num_features=c_out)
        self.relu = N.ReLU()
    def forward(self, input):
        out = self.bn(self.conv(input))
        return self.relu(out)

class GenerateRegistration:
    def __init__(self, net, loss, optim):
        if torch.cuda.is_available():
            self.x_dtype = torch.cuda.FloatTensor
            self.y_dtype = torch.cuda.FloatTensor
        else:
            self.x_dtype = torch.FloatTensor
            self.y_dtype = torch.FloatTensor
        self.net = net
        self.loss = loss
        self.opt = optim

    def predict(self, x):
        self.net.train(False)
        x = autograd.Variable(x, volatile=False).type(self.x_dtype)
        h = self.net(x)
        return h

    def score(self, x, y, criteria=None):
        self.net.train(False)
        criteria = criteria or self.loss
        x = autograd.Variable(x, volatile=False).type(self.x_dtype)
        y = autograd.Variable(y, volatile=False).type(self.y_dtype)

        h = self.net(x)
        j = criteria(h, y)
        return j

    def partial_fit(self, x, y):
        self.net.train(True)
        x = autograd.Variable(x).type(self.x_dtype)
        y = autograd.Variable(y).type(self.y_dtype)

        self.opt.zero_grad()
        h = self.net(x)
        print("output_size:", h.size())
        j = self.loss(h, y)
        j.backward()
        self.opt.step()
        return j
