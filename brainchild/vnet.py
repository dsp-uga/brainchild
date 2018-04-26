import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

def passthrough(x, **kwargs):
    '''
     Just for passing the input without any processing. 
    '''
    return x

def ELUCons(elu, nchan):
    '''
        Current implementation contains two nn transfer functions, 
        ELU and PRelu. PRelu is in the actual V-Net implementation. 
        The selection depends on elu argument's value and if the PRelu is
        selected, nchan defines the number of channles in that.
    '''
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    '''
     Normalize the input and check if the convolution size matrix is 5*5.
     For more info check documentation of torch.nn.BatchNorm1d .
    '''
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
       # super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        print('Check is OK.')
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    '''
     This class creates a convolution layer gien the input and the number of output channles. 
     The ELU function is either PRelu or ELU.
    '''
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        print('LU Conv.:',x.size())
        out = self.conv1(x)
        print('LU Conv: after conv1:',out.size())
        out = self.bn1(out)
        print('Lu Conv: after bn1:', out.size())
        out = self.relu1(out)
        print('LU Output after Relu1:', out.size())
        return out


def _make_nConv(nchan, depth, elu):
    '''
     This function is applied for the applying n convolution on the input data, which 
     would produce output with n-channel.
    '''
    print('Channles:', nchan)
    print('depth:',depth)
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
        print('Layers Size:', len(layers))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    '''
      In V-Net, the first layer starts with 1 channle and 128*128*64 sized
      input. The input is convolved with 5*5 kernel and normalized to 16
      channels separately. In the end summation of both is passed through 
      the PRelu function, which gives a 16 channel output.
    '''
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        print('Inside Input Transition:',x.size())
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        print('Printing the current output of Batch Normalization1:',out.size())
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 1)
        print('Converting into 16 channles',x16.size())
        x16_out = torch.add(out,x16)
       # x16_out = out + x
        print('Sum:', x16_out.size())
        out = self.relu1(x16_out)
        print('After Relu1:',out.size())
        return out


class DownTransition(nn.Module):
    '''
      The DownTransition first convolves the input by doubling the challales,
      and reducing the input size by half in each dimensions using 2*2 filters and stride 2.
      After that it is being applied with n number of convolution and then  
      a fully connected layer with PRelu.
    '''
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        print('Input:', x.size())
        down = self.relu1(self.bn1(self.down_conv(x)))
        print('Down:', down.size())
        out = self.do1(down)
        print('After Do Nothing:', out.size())
        out = self.ops(out)
        print('After Optimization:', out.size())
        out = self.relu2(torch.add(out, down))
        print('After RELU:',out.size())
        return out


class UpTransition(nn.Module):
    '''
        Upransition de-convolves the input by reducing the channles to half(not in all cases)
        and doubling the output dimensions using 2*2 filters of 2 strides. The same 5*5 filter
        convolution is applied without updating the size and allowing the input to pass through
        the PRelu. One important thing is that the fine-grain features are forwarded from the
        parallel down-layer from the architecture and are merged with the input in the up-layer.
    '''
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        print('Uptransition Output of Relu:', out.size())
        print('The features of the DownTransition:',skipxdo.size())
        xcat = torch.cat((out, skipxdo), 1)
        print('Contactanation:',xcat.size())
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    '''
        This layer converts 32 layer input into 2 layers of output, witout changing any size
        using regular 5*5 filters. In the end 1*1 filter is applied to get the final 128*128*64 sized 
        output of the 2 channel. The softmax function is used for generating the final features, which
        can be in the logarithmic form.
        
    '''
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = ELUCons(elu, 2)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        
        ## I don't understand use of these two lines.
        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        out = out.view(out.numel() // 2, 2)
        
        out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out


class VNet(nn.Module):
    '''
        Implements V-Net Configuration as described in the 
	
	Args: 
	 elu : if true selects the ELU as the network function and if false uses the PRelu.
	 nll : If to use the logarithmic softmax output or not. 
        
        Milletari, Fausto, Nassir Navab, and Seyed-Ahmad Ahmadi. 
        "V-net: Fully convolutional neural networks for volumetric medical image segmentation." 
        3D Vision (3DV), 2016 Fourth International Conference on. IEEE, 2016.
    '''
    def __init__(self,elu=False,nll=False):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu,dropout=True)
        self.down_tr128 = DownTransition(64, 2, elu,dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu,dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu,dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu,dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu,dropout=True)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu, nll)
    
    def forward(self, x):
        print('Inside Modelling')
        out16 = self.in_tr(x)
        gc.collect()
        print('L1')
        out32 = self.down_tr32(out16)
        gc.collect()
        print('D1')
        out64 = self.down_tr64(out32)
        gc.collect()
        print('D2')
        out128 = self.down_tr128(out64) 
        gc.collect()
        print('D3')
        out256 = self.down_tr256(out128) 
        gc.collect()
        print('D4')
        out = self.up_tr256(out256,out128)
        gc.collect()
        print('U1')
        out = self.up_tr128(out, out64)
        gc.collect()
        print('U2')
        out = self.up_tr64(out, out32)
        gc.collect()
        print('U3')
        out = self.up_tr32(out, out16)
        gc.collect()
        print('U4')
        out = self.out_tr(out)
        print('Final')
        return out
