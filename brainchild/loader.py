import torch
import nibabel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Adapted from dataloader built for previous course assigment

class MindboggleData(Dataset):
    '''An implementation of the torch `Dataset` interface for Mindboggle dataset.
    '''
    def __init__(self, data_dir, goal):
        if goal =='register':
            self.meta = get_registration_data(data_dir)
        elif goal == 'segment':
            self.meta = get_segmentation_data(data_dir)
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        meta = self.meta[i]
        img = nibabel.load(meta['path'])
        x = img.get_fdata(caching='unchanged'))
        x = torch.from_numpy(x)
        trgt_img = nibabel.load(meta['label'])
        y = trgt_img.get_fdata(caching='unchanged')
        y = torch.from_numpy(y)
        return x, y
class PPMIData(Dataset):
    '''An implementation of the torch `Dataset` interface for PPMI dataset.
    '''
    def __init__(self, data_dir):
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
    data_dir = Path(data_dir)
    meta = pd.read_csv(data_dir / 'metadata.csv')
    meta = meta.sort_values('Subject')
    meta = meta[['Subject', 'Group']]
    meta = meta.drop_duplicates()
    meta = meta.set_index('Subject')

    for path in data_dir.glob('*'):
        if path.name in ignore: continue
        subject = int(path.name)
        label = 'PD' in meta.loc[subject]['Group']
        label = int(label)
        paths = Path(path).glob(f'**', recursive=True)
        paths = (p for p in paths if p.name not in ignore)
        ret.extend({
            'path': p,
            'label': label,
        } for p in paths)

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
    data_dir = Path(data_dir)
    subjects = np.loadtxt(fname=data_dir/ 'subject_list_Mindboggle101.txt', dtype='str')
    for person in subjects:
        label = data_dir / 'Registered' / person / 't1weighted_brain.MNI152.nii.gz'
        path =  data_dir / 'Unregistered' / person / 't1weighted_brain.nii.gz'
        ret.extend({
            'path': path,
            'label': label,
        })
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
    labels_fname = f'labels.{seg_type}.manual.nii.gz'
    # Open the metadata csv
    data_dir = Path(data_dir)
    subjects = np.loadtxt(fname=data_dir/ 'subject_list_Mindboggle101.txt', dtype='str')
    for person in subjects:
        for type in ['Registered', 'Unregistered']:
            label = data_dir / type / person / labels_fname
            if type == 'Registered':
                path = data_dir / type / person / 't1weighted_brain.MNI152.nii.gz'
            elif type == 'Unregistered':
                path = data_dir / type / person / 't1weighted_brain.nii.gz'
            ret.extend({
                'path': path,
                'label': label,
            })
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
    kwargs.setdefault('batch_size', 1)
    kwargs.setdefault('shuffle', True)
    kwargs.setdefault('num_workers', 4)
    kwargs.setdefault('pin_memory', torch.cuda.is_available())
    if dataset=='Mindboggle'
        ds = MindboggleData(data_dir, goal)
    elif dataset=='PPMI'
        ds = PPMIData(data_dir)
    return DataLoader(ds, **kwargs)
