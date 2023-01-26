'''See class OSICLungHeartTracheaDataModule'''
import os
import io
import csv
import math
import random
from zipfile import ZipFile

import nrrd 

import torch
import torchio as tio

from ..util import verify_sha256
from ..base_data_module import BaseDataModule

__all__ = [
    'OSICLungHeartTracheaDataModule'
]

class OSICLungHeartTracheaDataModule(BaseDataModule):
    '''Data module for loading OSIC lung, heart and trachea segmentations 

    There are 
      - 111 smooth lungs masks
      - 110 noisy lung masks
      - 87 heart masks
      - 110 trachea masks

    The masks are from
    Lung segmentation dataset by Kónya et al., 2020
    https://www.kaggle.com/sandorkonya/ct-lung-heart-trachea-segmentation

    Prepared from data released by the Open Source Imaging Consortium (OSIC) at
    https://www.kaggle.com/competitions/osic-pulmonary-fibrosis-progression/overview
    '''
    # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 data_dir,
                 data_info_path,
                 batch_size,                     
                 download = False,
                 data_archive = 'archive.zip',
                 use_anatomies=('lung', 'noisy', 'heart', 'trachea'),
                 as_images=False,
                 train_ratio=0.4,
                 val_ratio=0.2,
                 **kwargs,
                 ):
        '''
        Parameters
        ----------
        data_dir : str
          The root data directory. IT is expected that `data_arhchive` is stored in this directory
           All preprocessed data will be stored with `data_dir` as root.
        
        data_info_path : str
          EITHER
           an existing csv file containing *at least* the following named columns
             dataset,image_path,fov_path,annotation_path,weight_path
           all other columns are ignored.
           The dataset column must contain values from {train, validation, test, predict}.
          OR
           a path to store data info in.
          If the file exists it will be used, if it does not exist it will be generated.
        
        batch_size : int
          Batch size

        download : bool, optional
          If True, download data to `data_dir`.

        data_archive : str, optional
          Name of data archive.                  

        use_anatomies : sequence, optional
          Anatomies to include. Subset of ('lung', 'noisy', 'heart', 'trachea')

        as_images : bool, optional
          If True, interpret data as images instead of as labels. This influences how transforms are
          applied, e.g. interpolation is always nearest neighbor for labels.

        train_ratio : float in [0,1], optional
          How large a proportion of the images to use for training. Must satisfy
          0 <= `train_ratio` + `val_ratio` <= 1
          test_ratio is given by 1 (1 - `train_ratio` - `val_ratio`)

        val_ratio : float in [0,1], optional
          How large a proportion of the images to use for validation.
        '''
        # pylint: disable=too-many-arguments
        super().__init__(data_info_path, batch_size, **kwargs)
        self.data_dir = data_dir
        self.download = download
        self.data_archive = data_archive        
        self.use_anatomies = use_anatomies
        self.as_images = as_images
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.bad_files = {'ID00167637202237397919352_lung.nrrd'}
        self.lung_is_2 = {
            'ID00019637202178323708467_lung.nrrd',
            'ID00048637202185016727717_lung.nrrd',
            'ID00122637202216437668965_lung.nrrd',
            'ID00149637202232704462834_lung.nrrd',
        }
        self.heart_is_2 = {'ID00139637202231703564336_lung.nrrd'}
            
    def create_subject(self, row):
        '''Create a Subject. Fields depend on self.use_anatomies.
        No anatomy is present for all PIDs.
        Missing values are represented as empty 4D tensors with shape (0,0,0,0)
        The corresponding <anatomy>-valid key is False for missing values
        '''
        row = row._asdict()
        subject_kwargs = {}
        # If there is missing data we need to create an empty image in the correct space
        # so we need to get an affine transformation and shape from one of the avilable images
        available = {}
        missing = []
        for anatomy in self.use_anatomies:
            path = row[anatomy]
            if isinstance(path, str) and len(path) > 0:
                available[anatomy] = path
            else:
                missing.append(anatomy)        

        affine = None
        shape = None
        for anatomy, path in available.items():
            subject_kwargs[f'{anatomy}-valid'] = True
            if self.as_images: 
                subject_kwargs[anatomy] = tio.ScalarImage(path)
            else:
                subject_kwargs[anatomy] = tio.LabelMap(path)

            if affine is None:
                affine = subject_kwargs[anatomy].affine
                shape = subject_kwargs[anatomy].shape
                
        if affine is None:
            affine = torch.eye(4)
            shape = (1,1,1,1)

        for anatomy in missing:
            subject_kwargs[f'{anatomy}-valid'] = False
            if self.as_images: 
                subject_kwargs[anatomy] = tio.ScalarImage(
                    tensor=torch.zeros(shape, dtype=torch.uint8),
                    affine=affine
                )
            else:
                subject_kwargs[anatomy] = tio.LabelMap(
                    tensor=torch.zeros(shape, dtype=torch.uint8),
                    affine=affine
                )
        return tio.Subject(**subject_kwargs)
    
    def prepare_data(self):
        '''Do the following
        1. Unpack the data
        2. Create data info file
        '''
        if self.download:
            self._download()

        if not os.path.exists(self.data_info_path):
            self._create_data_info()
                
    def _download(self):
        # pylint: disable=too-many-locals
        os.makedirs(self.data_dir, exist_ok=True)
        data_source = \
            'https://www.kaggle.com/datasets/sandorkonya/ct-lung-heart-trachea-segmentation'
        dl_file = os.path.join(self.data_dir, self.data_archive)
        if not os.path.exists(dl_file):
            raise IOError(f"Data archive not found: {dl_file}\n"
                          "OSIC lung, heart, trachea data download requires login. Please go to\n"
                          f"{data_source}\n"
                          f"and download the data to {self.data_dir}")
        data_sha256 = '9b31853b9ad826c77c5bf54993e72a0748ebd3880bf0f1ab32dc895c6b94e8bf'
        if not verify_sha256(dl_file, data_sha256):
            raise ValueError(
                f'Downloaded file: "{dl_file}" does not match expected checksum'
            )
        print(
            f'{dl_file} checksum match.\n'
            f'Extracting to {self.data_dir}. This will take some time'
        )
        # There are a couple of issues we would like to fix
        # 1. Make nicer directory structure
        #    The zip file has superflous subdirectories, e.g nrrd_heart/nrrd_heart
        #    The nrrd_ prefix is also superflous
        # 2. Data is stored as uncompressed 16-bit integers
        #    We dont need 16bits to store a binary mask. Depending on use case it might be nice to
        #    store as uint8 (for mask) or float (for input to models). To optimize for space we
        #    store as uint8
        #    Regardless of datatype we want to compress the data inside the NRRD files, which reduce
        #    size on disk by a factor of around 100-500
        # 3. A couple of files are problematic.
        #    Wrong spacing
        #        ID00167637202237397919352_lung.nrrd
        #    Heart included in lung segmentation with pixel value 2
        #        ID00139637202231703564336_lung.nrrd 
        #    Pixel value 2 instead of 1
        #        ID00019637202178323708467_lung.nrrd
        #        ID00048637202185016727717_lung.nrrd
        #        ID00122637202216437668965_lung.nrrd
        #        ID00149637202232704462834_lung.nrrd
        dirmap = {
            'nrrd_lung/nrrd_lung' : 'lung',
            'nrrd_noisy/nrrd_noisy' : 'noisy',
            'nrrd_trachea/nrrd_trachea' : 'trachea',
            'nrrd_heart/nrrd_heart' : 'heart',
        }
        for dst_dir in dirmap.values():
            os.makedirs(os.path.join(self.data_dir, dst_dir), exist_ok=True)

        with ZipFile(dl_file) as archive:
            total_files = len(archive.namelist())
            digits = 1 + math.floor(math.log10(total_files))
            backspaces = ''.join(('\b',) * (1+2*digits))
            print(f'Processing {total_files:{digits}} files', flush=True)
            for i, inpath in enumerate(archive.namelist()):
                print(f'{backspaces}{i+1:{digits}}/{total_files:{digits}}', end='', flush=True)
                # One file has a space before _
                filename = os.path.basename(inpath).replace(' ', '')
                if filename in self.bad_files:
                    print('Skipping {filename}')
                    continue
                subdir = dirmap[os.path.dirname(inpath)]
                outpath = os.path.join(self.data_dir, subdir, filename)
                stream = io.BytesIO(archive.read(inpath))
                header = nrrd.read_header(stream)
                data = nrrd.read_data(header, stream)
                if filename in self.lung_is_2:
                    print(f' Setting lung field to 1 instead of 2 in {filename}')
                    data[data==2] = 1
                elif filename in self.heart_is_2:
                    print(f' Removing heart in {filename}')
                    data[data==2] = 0
                header['type'] = 'uint8'
                header['encoding'] = 'gzip'
                nrrd.write(outpath, data.astype('uint8'), header, compression_level=9)


    def _create_data_info(self):
        pids = set()
        for anatomy in self.use_anatomies:
            for entry in os.scandir(os.path.join(self.data_dir, anatomy)):
                if entry.is_file():
                    pids.add(entry.name.split('_')[0])
        n_pids = len(pids)
        n_train = round(self.train_ratio * n_pids)
        n_val = round(self.val_ratio * n_pids)
        n_test = n_pids - n_val - n_train
        datasets = ['train']*n_train + ['validation']*n_val + ['test']*n_test
        random.shuffle(datasets)

        paths = {anatomy : [] for anatomy in self.use_anatomies}
        for pid in pids:
            for anatomy in self.use_anatomies:
                path = os.path.join(self.data_dir, anatomy, f'{pid}_{anatomy}.nrrd')
                if not os.path.exists(path):
                    path = ''
                paths[anatomy].append(path)

        header = ['dataset'] + list(paths.keys())
        with open(self.data_info_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            writer.writerows(zip(datasets, *paths.values()))
            
