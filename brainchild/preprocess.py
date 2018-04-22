import torch
import torch.nn as N
import torch.nn.functional as F

def zero_pad_tensor(tensor, d, w, h):
    '''Function to pad tensors with zeros to same batch_size

    Args:
        array (numpy array): The tensor that is to be padded
        d (int): Size of this function output in first dimension
        w (int): Size of this function output in second dimension
        h (int): Size of this function output in third dimension
    '''
    array = tensor.numpy()
    c, x, y, z = array.shape

    pad_x1= 0
    pad_x2= 0
    pad_y1= 0
    pad_y2= 0
    pad_z1= 0
    pad_z2= 0

    d_x = d-x
    d_y = w-y
    d_z = h-z

    if d_x%2 ==0:
        pad_x1 = pad_x2 = d_x//2
    else:
        pad_x1 = math.floor(d_x//2)
        pad_x2 = math.ceil(d_x//2)

    if d_y%2 ==0:
        pad_y1 = pad_y2 = d_y//2
    else:
        pad_y1 = math.floor(d_y//2)
        pad_y2 = math.ceil(d_y//2)

    if d_z%2 ==0:
        pad_z1 = pad_z2 = d_z//2
    else:
        pad_z1 = math.floor(d_z//2)
        pad_z2 = math.ceil(d_z//2)

    pad_width = ((pad_x1,pad_x2),(pad_y1,pad_y2),(pad_z1,pad_z2))
    print(type(pad_width))
    ret = np.pad(array,pad_width, mode='constant', constant_values=0)
    ret = torch.from_numpy(ret)
    return ret
