# pylint: disable=trailing-whitespace
'''See class STAREDataModule'''
import os
import mmap
import glob
import tarfile
import gzip
import hashlib
import csv
import random
from zipfile import ZipFile

import wget
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torchio as tio
from skimage.io import imread, imsave
from torch.utils.data import DataLoader


__all__ = [
    'STAREDataModule'
    ]


class STAREDataModule(pl.LightningDataModule):
    '''Data module for loading STARE data
    STARE: STructured Analysis of the Retina 
    https://cecas.clemson.edu/~ahoover/stare/

    A. Hoover, V. Kouznetsova and M. Goldbaum, "Locating Blood Vessels in Retinal Images by Piece-
    wise Threshold Probing of a Matched Filter Response", IEEE Transactions on Medical Imaging, 
    vol. 19 no. 3, pp. 203-210, March 2000.

    A. Hoover and M. Goldbaum, "Locating the optic nerve in a retinal image using the fuzzy
    convergence of the blood vessels", IEEE Transactions on Medical Imaging , vol. 22 no. 8, 
    pp. 951-958, August 2003.
    '''
    # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 image_dir,
                 label_dir_ah, 
                 label_dir_vk,
                 data_info_path,
                 batch_size,
                 download=False,
                 prepare_data_for_processing=False,
                 preprocessing_transform=None,
                 augmentation_transform=None,
                 train_ratio=0.4,
                 val_ratio=0.2,
                 num_workers=1,
    ):
        '''
        Parameters
        ----------
        image_dir : str
          The directory containing the 397 images from STARE. 
        
        label_dir_ah: str
          The directory containing the 20 manual segmentations done by Adam Hoover

        label_dir_vk : str
          The directory containing the 20 manual segmentations done by Adam Hoover

        data_info_path : str
          EITHER
           an existing csv file containing *at least* the following named columns
             dataset, imagename
           all other columns are ignored.
           The dataset column must contain values from {train, validation, test, predict}.
           The imagename column must match the imagenames of the STARE images.
          OR
           a path to store data info in. If the file exists it will be used, if it does not exist
           it will be generated.
        
        batch_size : int
          Batch size

        download : bool, optional
          If True, download data to the specified directories.

        prepare_data_for_processing : bool, optional
          If True, convert ppm to png, create empty annotations for image that do not have
          annotations, create annotation weights

        preprocessing_transform : torchio.transforms.Transform, optional
          Transforms to apply to all images

        augmentation_transform : torchio.transforms.Transform, optional
          Transforms to apply to training images

        train_ratio : float in [0,1], optional
          How much data to use for training when generating datainfo. Must satisfy
          0 <= `train_ratio` + `val_ratio` <= 1
          test_ratio is given by 1 (1 - `train_ratio` - `val_ratio`)

        val_ratio : float in [0,1], optional
          How much data to use for validation when generating datainfo. 

        num_workers : int, optional
          How many workers to use for generating data          
        '''
        # pylint: disable=too-many-arguments
        super().__init__()
        self.image_dir = image_dir
        self.label_dir_ah = label_dir_ah
        self.label_dir_vk = label_dir_vk
        self.preprocess = preprocessing_transform
        self.augment = augmentation_transform
        self.data_info_path = data_info_path
        self.batch_size = batch_size
        self.download = download
        self.prepare_data_for_processing = prepare_data_for_processing
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.datasets = {}
        self.num_workers = num_workers
        
    def prepare_data(self):
        if self.download:
            self._download()

        if self.prepare_data_for_processing:
            self._prepare_data_for_processing()

        if not os.path.exists(self.data_info_path):
            self._create_data_info()
                
    def _download(self):
        for directory in [self.image_dir, self.label_dir_ah, self.label_dir_vk]:
            os.makedirs(directory, exist_ok=True)
           
        image_src = 'https://cecas.clemson.edu/~ahoover/stare/images/all-images.zip'
        print('Downloading', image_src)
        image_file = wget.download(image_src, self.image_dir)
        image_file_sha256 = '6428ecc394f1b49a7192134990934f62fcd7d36110fd7c344b912dc43925e853'
        if not self._check_sha256(image_file, image_file_sha256):
            raise ValueError(
                f'Downloaded all-images.zip file: "{image_file}" does not match expected checksum'
            )
        print(f'{image_file} checksum match')
        self._extract_image_archive(image_file, self.image_dir)

        ah_src = 'https://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar'
        print('Downloading', ah_src)
        ah_file = wget.download(ah_src, self.label_dir_ah)
        ah_file_sha256 = 'ebf2f1e17ca955f24579d9edd990e2dae79a5c82def69f0985d8e24f826ddd2f'
        if not self._check_sha256(ah_file, ah_file_sha256):
            raise ValueError(
                f'Downloaded labels-ah.tar file: "{ah_file}" does not match expected checksum'
            )
        print(f'{ah_file} checksum match')
        self._extract_label_archive(ah_file, self.label_dir_ah)

        vk_src = 'https://cecas.clemson.edu/~ahoover/stare/probing/labels-vk.tar'
        print('Downloading', vk_src)
        vk_file = wget.download(vk_src, self.label_dir_vk)
        vk_file_sha256 = '47474a701536b0cfdb369fdce012be36141e9f44d80387f0179446b5cb0f5576'
        if not self._check_sha256(vk_file, vk_file_sha256):
            raise ValueError(
                f'Downloaded labels-vk.tar file: "{vk_file}" does not match expected checksum'
            )
        print(f'{vk_file} checksum match')
        self._extract_label_archive(vk_file, self.label_dir_vk)

    def _check_sha256(self, path, checksum):
        # pylint: disable=no-self-use
        hasher = hashlib.sha256()
        with open(path, 'r+b') as infile:
            mmapped = mmap.mmap(infile.fileno(), 0)
            hasher.update(mmapped)
        return hasher.hexdigest() == checksum
        
    def _extract_image_archive(self, archive_path, outdir):        
        '''The STARE images are in zip archive containing ppm images.
        We trust that the archive is as expected, so only call this if the check sums match'''
        # pylint: disable=no-self-use
        with ZipFile(archive_path) as archive:
            archive.extractall(outdir)
        
        
    def _extract_label_archive(self, archive_path, outdir):
        '''The STARE label files are tar archives containing gzipped ppm images.
        This method will gunzip the ppm images outdir
        We trust that the archive is as expected, so only call this if the check sums match
        '''
        # pylint: disable=no-self-use
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                with archive.extractfile(member) as member_file:                    
                    ppm_data = gzip.decompress(member_file.read())
                    # We would like to have the images stored as png files because they are read by
                    # TorchIO. So we first store the data as a ppm then convert in
                    # _prepare_data_for_processing.
                    # It might be cleaner to store directly as png, but then we have to parse the
                    # data as a ppm file, for example with PIL.Image.frombytes, which requires that
                    # we know mode, size, etc. Things that are handled if we make the roundtrip to a
                    # file.                    
                    out_ppm_file = os.path.join(outdir, member.name[:-3])
                    with open(out_ppm_file, 'wb') as out:
                        out.write(ppm_data)


    def _prepare_data_for_processing(self):
        # The images are stored as 2D 8-bit unsiged RGB images in ppm format. We want to use
        # TorchIO that does not handle ppm, so we convert everything to png images that are
        # handled by TorchIO.
        # Only 20 images have reference segmentations, so to simplify processing we create fake
        # segmentations for the rest and use a weight map to control which pixel predictions should
        # be included in the loss.
        imagenames = [
            os.path.splitext(os.path.basename(path))[0]
            for path in glob.glob(os.path.join(self.image_dir, '*.ppm'))
        ]            
        for imagename in imagenames:
            ppm_path = os.path.join(self.image_dir, imagename + '.ppm')
            png_path = os.path.join(self.image_dir, imagename + '.png')
            image = imread(ppm_path)
            imsave(png_path, image, check_contrast=False)
            os.remove(ppm_path)
                
            for mask_dir, lbl in zip([self.label_dir_ah, self.label_dir_vk], ['ah', 'vk']):
                mask_ppm_path = os.path.join(mask_dir, f'{imagename}.{lbl}.ppm' )
                mask_png_path = os.path.join(mask_dir, f'{imagename}.{lbl}.png' )
                if os.path.exists(mask_ppm_path):
                    mask = imread(mask_ppm_path)
                    weights = np.ones_like(mask)                    
                else:
                    mask = np.zeros(image.shape, dtype='uint8')
                    weights = np.zeros_like(mask)                    
                imsave(mask_png_path, mask, check_contrast=False)
                if os.path.exists(mask_ppm_path):
                    os.remove(mask_ppm_path)                    
                weights_path = os.path.join(mask_dir, f'{imagename}.{lbl}.weights.png')
                imsave(weights_path, weights, check_contrast=False)
                
                    

    def _create_data_info(self):
        imagenames = [
            os.path.splitext(os.path.basename(path))[0]
            for path in glob.glob(os.path.join(self.image_dir, '*.png'))
        ]
        random.shuffle(imagenames)
        n_images = len(imagenames)
        n_train = round(self.train_ratio * n_images)
        n_val = round(self.val_ratio * n_images)
        n_test = n_images - n_val - n_train
        datasets = ['train']*n_train + ['validation']*n_val + ['test']*n_test
        with open(self.data_info_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['dataset', 'imagename'])
            writer.writerows(zip(datasets, imagenames))                
                    
    def setup(self, stage=None):
        # pylint: disable=too-many-branches
        data_info = pd.read_csv(self.data_info_path)

        # If we are being called by a lightning Trainer we only make the
        # relevant datasets. Otherwise (stage is None) we make everything,
        if stage == 'fit':
            data_info = data_info[(data_info['dataset'] == 'train') |
                                  (data_info['dataset'] == 'validation')]
        elif stage == 'validate':
            data_info = data_info[data_info['dataset'] == 'validation']
        elif stage == 'test':
            data_info = data_info[data_info['dataset'] == 'test']
        elif stage == 'predict':
            data_info = data_info[data_info['dataset'] == 'predict']
        
        # Now we can create the actual samples
        samples = {
            'train' : [],
            'validation' : [],
            'test' : [],
            'predict' : []
        }
        for row in data_info.itertuples():
            image_path = os.path.join(self.image_dir, f'{row.imagename}.png')
            mask_ah_path = os.path.join(self.label_dir_ah, f'{row.imagename}.ah.png')
            weight_ah_path = os.path.join(self.label_dir_ah, f'{row.imagename}.ah.weights.png')
            mask_vk_path = os.path.join(self.label_dir_vk, f'{row.imagename}.vk.png')
            weight_vk_path = os.path.join(self.label_dir_vk, f'{row.imagename}.vk.weights.png')                             
            subject_kwargs = {
                'image' : tio.ScalarImage(image_path),
                'mask-1' : tio.LabelMap(mask_ah_path),
                'mask-1-weight' : tio.LabelMap(weight_ah_path),
                'mask-2' : tio.LabelMap(mask_vk_path),
                'mask-2-weight' : tio.LabelMap(weight_vk_path)
            }
            try:
                samples[row.dataset].append(tio.Subject(**subject_kwargs))
            except KeyError:
                print('Uknown dataset', row.dataset, 'Expected train, validation, test or predict')
                continue

        if self.augment is None:
            train_transform = self.preprocess
        else:
            if self.preprocess is None:
                train_transform = self.augment
            else:
                train_transform = tio.Compose([self.preprocess, self.augment])

        if len(samples['train']) > 0:
            self.datasets['train'] = tio.SubjectsDataset(samples['train'],
                                                         transform=train_transform)
        for ds_name in ['validation', 'test', 'predict']:
            if len(samples[ds_name]) > 0:
                self.datasets[ds_name] = tio.SubjectsDataset(samples[ds_name],
                                                             transform=self.preprocess)

    def _get_dataloader(self, ds_name, shuffle=False):
        if not ds_name in self.datasets:
            raise ValueError(f'No {ds_name} data available. Ensure that at least one row in '
                             '"{self.data_info_path}" has dataset = {ds_name}')
        return DataLoader(self.datasets[ds_name],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=shuffle)
        
    def train_dataloader(self):
        return self._get_dataloader('train', True)
    
    def val_dataloader(self):
        return self._get_dataloader('validation', False)

    def test_dataloader(self):
        return self._get_dataloader('test', False)

    def predict_dataloader(self):
        return self._get_dataloader('predict', False)
