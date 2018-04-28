"""

This code trains the V-Net model from the given dataset. The code inturn calls
the vnet.py method to 

Example:
    How to run::
        $ python vnet_train.py
		
References: 
Actual Implementation: https://github.com/mattmacy/vnet.pytorch
https://github.com/wildphoton/torchbiomed
All the reference of pytorch from : http://pytorch.org/
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchbiomed.loss as bioloss
import torchbiomed.utils as utils

import os
import sys
import math
import time
import setproctitle

import loader
import vnet





def weights_init(m):
    """
	  Initailize the weights of the network of the given model class. 
	  Check the pytorch documentation for the kaiming_normal for further information.
	  Args: 
	    m is the model object.
	"""
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()




def datestr():
    """
		Generate String to save the model with the time stamp.
	"""
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)




def load_dataset():
    """
	 Loads the data using the DataLoader implementation in loader.py.
	 Hard Coding: For this project the data is Mindboggle only, so the values are hard coded.
	"""
    data_dir = '../data/Mindboggle'
    data = loader.load_dataset(data_dir, dataset='Mindboggle', goal='segment')
    return data





def adjust_opt(optAlg, optimizer, epoch):
    """
		Adjusting the optimization Parameters based on the number of the epochs.
		Keeping from the original implementation, as it is meaningful. 
		Args:
		  optAlg is the optimization algorithm. 
		  optimizer is the optimizer object. 
		  epoch is the total number of epoch in the network.
	"""
    if optAlg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train_nll(is_cuda, epoch, model, trainLoader, optimizer, trainF, weights):
    """
		This method trains the model using the vnet.py file and saves the model on the disk.
		Args: 
		  is_cuda Is using the GPU?
		  epoch is the total number of epoch going to be used in to train network.
		  model is the vnet model object.
		  trainLoader is the DataLoader object for the current data. 
		  trainF is for saving the intermediate model.
		  weights are the class distribution.

	"""
    print('Training Model')

    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)

    print('Training Examples:',nTrain)
    for batch_idx, (data, target) in enumerate(trainLoader):

     	#Resizing the data according to the need.
		#Hard-coded value for setting the input to (1,1,128,128,64) channel.
        data = data.view(-1,1,128,128,64)
        data = torch.chunk(data,16,0)[0]
        target = target.view(-1,1,128,128,64)
        target = torch.chunk(target,16,0)[0]
     
	 if is_cuda:
            data, target = data.cuda(), target.cuda()

        print('Data:',data.size())
        print('Target:',target.size())
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        print('Train Model')
        
		output = model(data)
        print('Get the trarget')
        
		target = target.view(target.numel())
        loss = F.nll_loss(output, target, weight=weights)
        dice_loss = bioloss.dice_error(output, target)
        
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*float(incorrect)/target.numel()
        partialEpoch = int(epoch) + int(batch_idx) / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tError: {:.3f}\t Dice: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err, dice_loss))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()
    



def test_nll(is_cuda, epoch, model, testLoader, optimizer, testF, weights):
	"""
		This method tests the trained model.
		Args: 
		  is_cuda Is using the GPU?
		  epoch is the total number of epoch going to be used in to train network.
		  model is the vnet model object.
		  testLoader is the DataLoader object for the current test data. 
		  optimizer is the Optmizer object.
		  testF is for saving the intermediate model.
		  weights are the class distribution.
	"""
    print('Model Evaluation')
    model.eval()
    test_loss = 0
    dice_loss = 0
    incorrect = 0
    numel = 0
    print('Initiate Parameter Settings.')
    for data, target in testLoader:
        print('Inside Loop')
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        print('Setting Variables')
        data, target = Variable(data, volatile=True), Variable(target)
        target = target.view(target.numel())
        numel += target.numel()
        output = model(data)
        test_loss += F.nll_loss(output, target, weight=weights).data[0]
        dice_loss += bioloss.dice_error(output, target)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss /= len(testLoader)  # loss function already averages over batch size
    dice_loss /= len(testLoader)
    err = 100.*incorrect/numel
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.3f}%) Dice: {:.6f}\n'.format(
        test_loss, incorrect, numel, err, dice_loss))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()
    return err


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    """
	  This method saves the checkpoint of the model on the disk after each epoch.
	  Args:
	  state is the current state of the model.
	  is_best is the flag to check if the model is best or not. 
	  path is the path to save the model. 
	  prefix is the path prefix.
	  filename is the name of the saved file.
	"""
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')





def noop(x):
    """
	  Do nothing. Just return the current value.
	"""
    return x


def main():
    """
	 Main method is organizing all the code to be called upon while executing this file. 
	 All the parameters are hard coded due to problem resolution of the memory.
	"""

    best_prec1 = 100.
    seed = 100
    is_cuda = False
    is_cuda = is_cuda and torch.cuda.is_available()
    nGPUs = 1
    unit_batch_size = 10
    save_model_path = 'work/vnet.base.{}'.format(datestr())
    nll = True #Softmax function (regular or logarithmic)
    weight_decay = 0 #weight decay 
    is_resume = False
    opt_algo = 'sgd'
    print('Parameters Initialized.')
    
    #Set Parameters.
    setproctitle.setproctitle(save_model_path)
    #Create directories if the path does not exist.
    if os.path.exists(save_model_path):
        shutil.rmtree(save_model_path)
    os.makedirs(save_model_path, exist_ok=True)
    
	#Set the seeds and change the parameter type according to the machine type.
    torch.manual_seed(seed)
    if is_cuda:
        torch.cuda.manual_seed(args.seed)
    print('Process Started with Seed,',seed)
   
    print("VNET Config.")
	# The model is VNet and the elu uses the ELU function, while the nll
	# is for the softmax function, if to use logarithmic type or not. 
    model = vnet.VNet(elu=True, nll=nll)
    model = model.double()
    batch_size = nGPUs*unit_batch_size
    gpu_ids = range(nGPUs)
   # model = nn.parallel.DataParallel(model, device_ids=gpu_ids)

    #Check if the model is already there and needed to be trained for another set of data.
    #If not create a new model.
    if is_resume:
         if os.path.isfile(args.resume):
             print("=> loading checkpoint '{}'".format(args.resume))
             checkpoint = torch.load(args.resume)
             args.start_epoch = checkpoint['epoch']
             best_prec1 = checkpoint['best_prec1']
             model.load_state_dict(checkpoint['state_dict'])
             print("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch']))
         else:
             print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model.apply(weights_init)



    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    if is_cuda:
        model = model.cuda()



    print("loading training set")
    trainLoader = load_dataset()
    print("loading test set")
    testLoader = load_dataset()
    
    #Setting up the class distribution.For now the hard coding the weights 
	#to 90% for the background and 10% for the foreground.
    bg_weight = 0.9
    fg_weight = 0.1
    class_weights = torch.DoubleTensor([bg_weight, fg_weight])
    if is_cuda:
        class_weights = class_weights.cuda()
    print('Setting Class Weights')
    


    print("Setting the Optimization Algorithm.")
    if opt_algo == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=1e-1,
                              momentum=0.99, weight_decay=weight_decay)
    elif opt_algo == 'adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    elif opt_algo == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), weight_decay=weight_decay)

    #Writing things to the file.
    trainF = open(os.path.join(save_model_path, 'train.csv'), 'w')
    testF = open(os.path.join(save_model_path, 'test.csv'), 'w')
    err_best = 100.
    total_epoch = 10
    print('Open Files for saving records')
    for epoch in range(1, total_epoch + 1):
        print(epoch)
        #Set Optimization parameters.
        adjust_opt(opt_algo, optimizer, epoch)
        print('Adjust the Parameters')
        train_nll(is_cuda, epoch, model, trainLoader, optimizer, trainF, class_weights)
        print('Training Done')
        err = test_nll(is_cuda, epoch, model, testLoader, optimizer, testF, class_weights)
        is_best = False
        if err < best_prec1:
            is_best = True
            best_prec1 = err
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_prec1},
                        is_best, save_model_path, "vnet")
        os.system('./plot.py {} {} &'.format(len(trainLoader),save_model_path))

    trainF.close()
    testF.close()
    print('Done.')


if __name__ == '__main__':
    main()

