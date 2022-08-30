'''See class HRFDataModule'''
import os
import csv
import glob
import random
from zipfile import ZipFile

import wget
import numpy as np
from skimage.io import imsave, imread

from .util import verify_sha256
from .retina_data_module import RetinaDataModule


__all__ = [
    'HRFDataModule'
    ]


class HRFDataModule(RetinaDataModule):
    '''Data module for loading High-Resolution Fundus (HRF) data
    https://www5.cs.fau.de/research/data/fundus-images/


    Budai, Attila; Bock, Rüdiger; Maier, Andreas; Hornegger, Joachim; Michelson, Georg.
    Robust Vessel Segmentation in Fundus Images.
    International Journal of Biomedical Imaging, vol. 2013, 2013 

    '''
    # pylint: disable=too-many-instance-attributes, too-few-public-methods
    def __init__(self,
                 data_dir,
                 data_info_path,
                 batch_size,
                 download=False,
                 prepare_data_for_processing=False,
                 stratify_pathologies=False,
                 transforms=None,
                 train_ratio=0.4,
                 val_ratio=0.2,
                 num_workers=1,
    ):
        '''
        Parameters
        ----------
        data_dir : str
          The directory containing HRF data. Expects the following sub directories
            images/
            manual1/
            mask/
        
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
          If True, download data to the specified directories.

         prepare_data_for_processing : bool, optional
          If True, rename *.JPG to *.jpg and create annotation weights

        stratify_pathologies : bool
          If True, then there will be a balanced distribution of the three patient groups {healthy,
          diabetic retinopathy, glaucomatous} in train, validation and test data.
          If False, patient groups are ignored for randomization in train, validation and test.

        transforms : dict of callables, optional
          Dictionary of transforms to apply to each dataset. Example: If
            transforms = {'train' : <some-transform>}
          then <some-transform> will be applied to the "train" dataset and all other datasets will
          not be transformed.

        train_ratio : float in [0,1], optional
          How large a proportion of the 20 train images to use for training. Must satisfy
          0 <= `train_ratio` + `val_ratio` <= 1
          test_ratio is given by 1 (1 - `train_ratio` - `val_ratio`)
          This only concerns how the training images are split. If the test images are included,
          `use_test_as_unlabeled_train_data` = True, they are always used for training.

        val_ratio : float in [0,1], optional
          How large a proportion of the train images to use for validation.

        num_workers : int, optional
          How many workers to use for generating data          
        '''
        # pylint: disable=too-many-arguments
        super().__init__()
        self.data_dir = data_dir
        self.data_info_path = data_info_path
        self.batch_size = batch_size
        self.prepare_data_for_processing = prepare_data_for_processing        
        self.stratify_pathologies = stratify_pathologies
        self.download = download
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.datasets = {}
        self.num_workers = num_workers
        if transforms is not None:
            self.transforms = transforms
        
    def prepare_data(self):
        '''Do the following
        1. Download data
        2. Rename *JPG to jpg, convert *tiff masks to *png, create weights
        3. Create data info file
        '''
        if self.download:
            self._download()
            
        if self.prepare_data_for_processing:
            self._prepare_data_for_processing()
            
        if not os.path.exists(self.data_info_path):
            self._create_data_info()
                
    def _download(self):
        os.makedirs(self.data_dir, exist_ok=True)
        data_source = 'https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip'
        data_sha256 = 'a914d02cda161b7f33f25d0397c276d50e9a6cbc705e9b364a54f0adafed57e4'        
        print('Downloading', data_source)
        dl_file = wget.download(data_source, self.data_dir)
        if not verify_sha256(dl_file, data_sha256):
            raise ValueError(
                f'Downloaded file: "{dl_file}" does not match expected checksum'
            )
        print(f'{dl_file} checksum match, Extracting to {self.data_dir}.')
        with ZipFile(dl_file) as archive:
            archive.extractall(self.data_dir)

    def _prepare_data_for_processing(self):
        image_dir = os.path.join(self.data_dir, 'images')
        fov_dir = os.path.join(self.data_dir, 'mask')
        annotation_dir = os.path.join(self.data_dir, 'manual1')        
        weight_dir = os.path.join(self.data_dir, 'weights')
        os.makedirs(weight_dir, exist_ok=True)

        print('Renaming *.JPG to *.jpg')
        for path in glob.glob(os.path.join(image_dir, '*.JPG')):
            os.rename(path, path[:-3] + 'jpg')

        # We convert to png because they are smaller and to avoid a warning from SimpleITK about
        # invalid data type for EXIFIFDOffset.
        print('Converting annotations to png and creating weight maps')
        for entry in os.scandir(annotation_dir):
            if entry.is_file():
                annotation = imread(entry.path)
                imsave(f'{os.path.splitext(entry.path)[0]}.png', annotation, check_contrast=False)
                os.remove(entry.path)
                weight_path = \
                    os.path.join(weight_dir, f'{os.path.splitext(entry.name)[0]}_weight.png')
                imsave(weight_path, np.ones_like(annotation, dtype='uint8'), check_contrast=False)

        print('Converting field of view masks to png')
        for entry in os.scandir(fov_dir):
            if entry.is_file():
                fov = imread(entry.path)
                imsave(f'{os.path.splitext(entry.path)[0]}.png', fov, check_contrast=False)
                os.remove(entry.path)

    def _create_data_info(self):
        '''There are 45 images in HRF. 15 healthy, 15 diabetic retinopathy, 15 glaucomatous.
        All have a field-of-view (FOV) mask.
        All have a reference segmentation
        '''
        # pylint: disable=too-many-locals
        if self.stratify_pathologies:
            n_images = 15 
        else:
            n_images = 45
        n_train = round(self.train_ratio * n_images)
        n_val = round(self.val_ratio * n_images)
        n_test = n_images - n_val - n_train
        datasets = ['train']*n_train + ['validation']*n_val + ['test']*n_test
        random.shuffle(datasets)
        if self.stratify_pathologies:
            datasets = datasets * 3

        image_dir = os.path.join(self.data_dir, 'images')
        fov_dir = os.path.join(self.data_dir, 'mask')
        annotation_dir = os.path.join(self.data_dir, 'manual1')
        weight_dir = os.path.join(self.data_dir, 'weights')
        
        image_paths, fov_paths, annotation_paths, weight_paths = [], [], [], []
        for pathology in ('h', 'dr', 'g'):
            for image_number in range(1, 16):
                image_paths.append(os.path.join(image_dir, f'{image_number:02d}_{pathology}.jpg'))
                fov_paths.append(os.path.join(fov_dir, f'{image_number:02d}_{pathology}_mask.png'))
                annotation_paths.append(
                    os.path.join(annotation_dir, f'{image_number:02d}_{pathology}.png')
                )
                weight_paths.append(
                    os.path.join(weight_dir, f'{image_number:02d}_{pathology}_weight.png')
                )
        
        with open(self.data_info_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['dataset', 'image_path', 'fov_path', 'annotation_path', 'weight_path'])
            writer.writerows(zip(datasets, image_paths, fov_paths, annotation_paths, weight_paths))
