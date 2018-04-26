import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

#import torchbiomed.datasets as dset
#import torchbiomed.transforms as biotransforms
import torchbiomed.loss as bioloss
import torchbiomed.utils as utils

import os
import sys
import math
import time
import setproctitle

import loader
import vnet




#Initailize the weights 
#Check the documentation for the kaiming_normal for further information.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()



#Generate String to save the model with the time stamp.
def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)



#Loads the data.
def load_dataset():
    data_dir = '../data/Mindboggle'
    data = loader.load_dataset(data_dir, dataset='Mindboggle', goal='register')
#     batch = next(data)
#     return batch
    return data




#Adjust the optimization Parameters based on the number of the epochs.
def adjust_opt(optAlg, optimizer, epoch):
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




#Train the model and save it.
def train_nll(is_cuda, epoch, model, trainLoader, optimizer, trainF, weights):
    print('Training Model')
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    print('Training Examples:',nTrain)
    for batch_idx, (data, target) in enumerate(trainLoader):
        #Resizing the data according to the need.
        data = data.view(-1,1,128,128,64)
        data = torch.chunk(data,16,0)[0]
        target = target.view(-1,1,128,128,64)
        target = torch.chunk(target,16,0)[0]
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        #print('Setting Variables')
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
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
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
    


# In[106]:


#Test data and return error.
def test_nll(is_cuda, epoch, model, testLoader, optimizer, testF, weights):
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


# In[107]:


#Save check point on the disk.
def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')





# In[109]:
def noop(x):
    return x


# In[117]:


def main():
    best_prec1 = 100.
    seed = 100
    
    is_cuda = False
    is_cuda = is_cuda and torch.cuda.is_available()
   
    print('GPU ENABLED:',is_cuda)
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
    
    torch.manual_seed(seed)
    if is_cuda:
        torch.cuda.manual_seed(args.seed)
    print('Process Started with Seed,',seed)
    
    print("VNET Config.")
    model = vnet.VNet(elu=True, nll=nll)
    model = model.double()
    batch_size = nGPUs*unit_batch_size
    gpu_ids = range(nGPUs)
   # model = nn.parallel.DataParallel(model, device_ids=gpu_ids)

    #Check if the model is already there and needed to be trained for another set of data.
    #If not create a new model.
    if is_resume:
        exit()
#         if os.path.isfile(args.resume):
#             print("=> loading checkpoint '{}'".format(args.resume))
#             checkpoint = torch.load(args.resume)
#             args.start_epoch = checkpoint['epoch']
#             best_prec1 = checkpoint['best_prec1']
#             model.load_state_dict(checkpoint['state_dict'])
#             print("=> loaded checkpoint '{}' (epoch {})"
#                   .format(args.evaluate, checkpoint['epoch']))
#         else:
#             print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model.apply(weights_init)



    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    if is_cuda:
        model = model.cuda()



#     kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    print("loading training set")
    trainLoader = load_dataset()
    print("loading test set")
    testLoader = load_dataset()
    
    #Setting up the class weight.For now the hard coding. 
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


# In[120]:


if __name__ == '__main__':
    main()

