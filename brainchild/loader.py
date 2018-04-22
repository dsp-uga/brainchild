import torch
import nibabel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import math


# Adapted from dataloader built for previous course assigment

class MindboggleData(Dataset):
    '''An implementation of the torch `Dataset` interface for Mindboggle dataset.
    '''
    def __init__(self, data_dir, goal):
        '''
        Args:
            data_dir (path):   Root directory of dataset
            goal (string arg): string 'register' or 'segment' to determine what
                               labels will be
        '''
        if goal =='register':
            self.meta = get_registration_data(data_dir)
        elif goal == 'segment':
            self.meta = get_segmentation_data(data_dir)
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        meta = self.meta[i]
        img = nibabel.load(meta['path'])
        x = img.get_fdata(caching='unchanged')
        x = zero_pad_array(x, 256, 256, 256)
        x = np.reshape(x,(1,256,256,256))
        x = torch.from_numpy(x)
        trgt_img = nibabel.load(meta['label'])
        y = trgt_img.get_fdata(caching='unchanged')
        y = zero_pad_array(x, 256, 256,  256)
        y = np.reshape(y, (1,256,256,256))
        y = torch.from_numpy(y)
        return x, y
def zero_pad_array(array, d, w, h):
    '''Function to pad tensors with zeros to same batch_size

    Args:
        array (numpy array): The tensor that is to be padded
        d (int): Size of this function output in first dimension
        w (int): Size of this function output in second dimension
        h (int): Size of this function output in third dimension
    '''
    x, y, z = array.shape

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
    return ret
class PPMIData(Dataset):
    '''An implementation of the torch `Dataset` interface for PPMI dataset.

    Initialize with a data directory for PPMI files and generates a dataset
    from list of paths.
    '''
    def __init__(self, data_dir):
        '''
        Args:
            data_dir: Path to root folder containing all PPMI files.
        '''
        self.meta = metadata(data_dir)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        meta = self.meta[i]
        img = nibabel.load(meta['path'])
        x = img.get_fdata(caching='unchanged')
        x = torch.from_numpy(x)
        y = meta['label']
        return x, y
def metadata(data_dir):
    '''Load the metadata for the PPMI dataset.

    A file `metadata.csv` should be stored in the same directory as the data.
    This file is cross-referenced with the actual data to only include metadata
    for the samples that actually exist locally.

    Args:
        data_dir (path):
            The root directory of the dataset.

    Returns (list of dict, one for each observation):
        path (path):
            The directory containing the nifti file for the observation.
        label (bool):
            True if the observation is in the Parkinson's class.
    '''
    # The list of metadata dicts for each observation.
    ret = []

    # Things which might exist in the data_dir that are not data.
    ignore = ['.DS_Store', 'metadata.csv']

    # Open the metadata csv
    dir = Path(data_dir)
    meta = pd.read_csv(dir / 'metadata.csv')
    meta = meta.sort_values('Subject')
    meta = meta[['Subject', 'Group']]
    meta = meta.drop_duplicates()
    meta = meta.set_index('Subject')

    for path in dir.glob('*'):
        if path.name in ignore: continue
        subject = int(path.name)
        label = 'PD' in meta.loc[subject]['Group']
        label = int(label)
        path = Path(path).glob('**/*.nii')
        path = list(path)[0].as_uri()
        ret.append({
            'path': path,
            'label': label})

    return ret
def get_registration_data(data_dir):
    '''Load the Mindboggle dataset.

    Args:
        data_dir (path):
            The root directory of the dataset.

    Returns (list of dict, one for each observation):
        path (path):
            The directory containing the nifti file for the observation.
        label (path):
            The directory containing the registered image.
    '''
    # The list of metadata dicts for each observation.
    ret = []

    # Things which might exist in the data_dir that are not data.
    ignore = ['.DS_Store', 'subject_list_Mindboggle101.txt']

    # Open the metadata csv
    # data_dir = Path(data_dir)
    with open(f'{data_dir}/subject_list_Mindboggle101.txt') as f:
        subjects = f.readlines()
    subjects = [x.strip() for x in subjects]
    for person in subjects:
        label = f'{data_dir}/Registered/{person}/t1weighted_brain.MNI152.nii.gz'
        path =  f'{data_dir}/Unregistered/{person}/t1weighted_brain.nii.gz'
        tmp_dict = {}
        tmp_dict['path'] = path
        tmp_dict['label'] = label
        ret.append(tmp_dict)

    return ret
def get_segmentation_data(data_dir, seg_type='DKT31'):
    '''Load the Mindboggle dataset.

    Args:
        data_dir (path):
            The root directory of the dataset.

    Returns (list of dict, one for each observation):
        path (path):
            The directory containing the nifti file for the observation.
        label (path):
            The directory containing the segmentation label.
    '''
    # The list of metadata dicts for each observation.
    ret = []

    # Things which might exist in the data_dir that are not data.
    ignore = ['.DS_Store', 'subject_list_Mindboggle101.txt']


    # Open the metadata csv
    with open(f'{data_dir}/subject_list_Mindboggle101.txt') as f:
        subjects = f.readlines()
    subjects = [x.strip() for x in subjects]
    for person in subjects:
        for type in ['Registered', 'Unregistered']:
            if type == 'Registered':
                path = f'{data_dir}/{type}/{person}/t1weighted_brain.MNI152.nii.gz'
                labels_fname = f'labels.{seg_type}.manual.MNI152.nii.gz'
            elif type == 'Unregistered':
                path = f'{data_dir}/{type}/{person}/t1weighted_brain.nii.gz'
                labels_fname = f'labels.{seg_type}.manual.nii.gz'
            label = f'{data_dir}/{type}/{person}/{labels_fname}'
            tmp_dict = {}
            tmp_dict['path'] = path
            tmp_dict['label'] = label
            ret.append(tmp_dict)
    return ret
def load_dataset(data_dir, dataset, goal, **kwargs):
    '''Creates a pytorch DataLoader that iterates over dataset out of core.
    Args:
        data_dir (path):
            The path to dataset.
        dataset (Dataset):
            Which dataset class to use. (PPMI or Mindboggle)
        goal (string):
            One of three options that set what the network's training objective is:
            'classify': used for the PPMIData
            'register': used for MindboggleData to get registered images as labels
                        and unregistered images as data
            'segment': used for MindboggleData to get segmentation label as labels
                        and images as data

    Kwargs:
        The kwargs are forwarded to the DataLoader constructor.
        The following kwrags have different defaults:

        batch_size (int):
            The number of samples per batch.
            Defaults to 1.
        shuffle (bool):
            Iterate over the epoch in a random order.
            Defaults to True.
        num_workers (int):
            The number of background processes prefetching the data.
            Defaults to 4.
        pin_memory (bool):
            Load the data into cuda pinned memory.
            Defaults to True if cuda is available.

        For the full list of kwargs, see `torch.utils.data.DataLoader`.

    Returns (torch.utils.data.DataLoader):
        A DataLoader that iterates over batches of the dataset for one epoch.
    '''
    kwargs.setdefault('batch_size', 4)
    kwargs.setdefault('shuffle', True)
    kwargs.setdefault('num_workers', 4)
    kwargs.setdefault('pin_memory', torch.cuda.is_available())
    if dataset=='Mindboggle':
        ds = MindboggleData(data_dir, goal)
    elif dataset=='PPMI':
        ds = PPMIData(data_dir)
    return DataLoader(ds, **kwargs)
