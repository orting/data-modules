'''See class DRIVEDataModule'''
import os
import csv
import random
from zipfile import ZipFile

import wget
import numpy as np
from PIL import Image
from skimage.io import imsave, imread

from .util import verify_sha256
from .retina_data_module import RetinaDataModule


__all__ = [
    'DRIVEDataModule'
    ]


class DRIVEDataModule(RetinaDataModule):
    '''Data module for loading DRIVE data
    https://drive.grand-challenge.org/
    '''
    # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 data_dir,
                 data_info_path,
                 batch_size,
                 download=False,
                 prepare_data_for_processing=False,
                 use_test_as_unlabeled_train_data=True,
                 preprocessing_transform=None,
                 augmentation_transform=None,
                 train_ratio=0.4,
                 val_ratio=0.2,
                 num_workers=1,
    ):
        '''
        Parameters
        ----------
        data_dir : str
          The directory containing DRIVE data. Expects the following sub directories
          training/
            1st_manual/
            images/
            mask/
          test/
            images/
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

        use_test_as_unlabeled_train_data : bool, optional
          Predictions on the test image can be submitted to the DRIVE grand challenge. If this is
          the aim, then the test images should be excluded completely from training.
          Otherwise the test set can be used for training as additional unlabeled data (default).

        prepare_data_for_processing : bool, optional
          If True, convert ppm to png, create empty annotations for image that do not have
          annotations, create annotation weights

        preprocessing_transform : torchio.transforms.Transform, optional
          Transforms to apply to all images

        augmentation_transform : torchio.transforms.Transform, optional
          Transforms to apply to training images

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
        self.preprocess = preprocessing_transform
        self.augment = augmentation_transform
        self.data_info_path = data_info_path
        self.batch_size = batch_size
        self.download = download
        self.prepare_data_for_processing = prepare_data_for_processing
        self.use_test_as_unlabeled_train_data = use_test_as_unlabeled_train_data
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.datasets = {}
        self.num_workers = num_workers
        
    def prepare_data(self):
        '''Do the following
        1. Download data
        2. Convert ppm to png, create empty annotations for image that do not have
           annotations, create annotation weights
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

        train_source = \
            'https://www.dropbox.com/sh/z4hbbzqai0ilqht/AADp_8oefNFs2bjC2kzl2_Fqa/training.zip?dl=1'
        test_source = \
            'https://www.dropbox.com/sh/z4hbbzqai0ilqht/AABuUJQJ5yG5oCuziYzYu8jWa/test.zip?dl=1'
        sources = [train_source, test_source]

        train_sha256 = '7101e19598e2b7aacdbd5e6e7575057b9154a4aaec043e0f4e28902bf4e2e209'
        test_sha256 = 'd76c95c98a0353487ffb63b3bb2663c00ed1fde7d8fdfd8c3282c6e310a02731'
        checksums = [train_sha256, test_sha256]
        
        for source, checksum in zip(sources, checksums):
            print('Downloading', source)
            dl_file = wget.download(source, self.data_dir)
            if not verify_sha256(dl_file, checksum):
                raise ValueError(
                    f'Downloaded file: "{dl_file}" does not match expected checksum'
                )
            print(f'{dl_file} checksum match, Extracting to {self.data_dir}.')
            with ZipFile(dl_file) as archive:
                archive.extractall(self.data_dir)

    def _prepare_data_for_processing(self):
        # Annotations and FOV masks are stored as 2D 8-bit unsiged grayscale images in gif format.
        # We want to use TorchIO that does not handle gif, so we convert gif to png images that are
        # handled by TorchIO.
        #
        # If the test images are used for training we create empty segmentations for those images.
        # We create pixel weights for all images to control which pixel predictions should be
        # included in the loss.
        #
        # torchio does not
        if self.use_test_as_unlabeled_train_data:
            image_dir = os.path.join(self.data_dir, 'test', 'images')
            fov_dir = os.path.join(self.data_dir, 'test', 'mask')
            annotation_dir = os.path.join(self.data_dir, 'test', 'dummy-annotations')
            os.makedirs(annotation_dir, exist_ok=True)
            for imagename in [f'{i:02d}' for i in range(1, 21)]:
                size = Image.open(os.path.join(image_dir, f'{imagename}_test.tif')).size
                dummy = np.zeros(size, dtype='uint8')
                imsave(os.path.join(annotation_dir, f'{imagename}_dummy.png'), dummy,
                       check_contrast=False)
                imsave(os.path.join(annotation_dir, f'{imagename}_dummy_weight.png'), dummy,
                       check_contrast=False)
                
                fov_gif_path = os.path.join(fov_dir, f'{imagename}_test_mask.gif')
                fov_png_path = os.path.join(fov_dir, f'{imagename}_test_mask.png')
                imsave(fov_png_path, imread(fov_gif_path), check_contrast=False)
                os.remove(fov_gif_path)

        for imagename in [f'{i:02d}' for i in range(21, 41)]:
            image_dir = os.path.join(self.data_dir, 'training', 'images')
            size = Image.open(os.path.join(image_dir, f'{imagename}_training.tif')).size

            annotation_dir = os.path.join(self.data_dir, 'training', '1st_manual')
            weight = np.ones(size, dtype='uint8')
            imsave(os.path.join(annotation_dir, f'{imagename}_manual1_weight.png'), weight,
                   check_contrast=False)
            
            annotation_gif_path = os.path.join(annotation_dir, f'{imagename}_manual1.gif')
            annotation_png_path = os.path.join(annotation_dir, f'{imagename}_manual1.png')
            imsave(annotation_png_path, imread(annotation_gif_path), check_contrast=False)
            os.remove(annotation_gif_path)

            fov_dir = os.path.join(self.data_dir, 'training', 'mask')
            fov_gif_path = os.path.join(fov_dir, f'{imagename}_training_mask.gif')
            fov_png_path = os.path.join(fov_dir, f'{imagename}_training_mask.png')
            imsave(fov_png_path, imread(fov_gif_path), check_contrast=False)
            os.remove(fov_gif_path)
                            
    def _create_data_info(self):
        '''There are 20 train and 20 test images in DRIVE.
        All have a field-of-view (FOV) mask.
        Train images have a reference segmentation
        '''
        n_images = 20
        n_train = round(self.train_ratio * n_images)
        n_val = round(self.val_ratio * n_images)
        n_test = n_images - n_val - n_train
        datasets = ['train']*n_train + ['validation']*n_val + ['test']*n_test
        random.shuffle(datasets)
        image_paths, fov_paths, annotation_paths, weight_paths = [], [], [], []
        train_dir = os.path.join(self.data_dir, 'training')
        for imagename in range(21, 41):
            image_paths.append(os.path.join(train_dir, 'images', f'{imagename}_training.tif'))
            fov_paths.append(os.path.join(train_dir, 'mask', f'{imagename}_training_mask.png'))
            annotation_paths.append(
                os.path.join(train_dir, '1st_manual', f'{imagename}_manual1.png')
            )
            weight_paths.append(
                os.path.join(train_dir, '1st_manual', f'{imagename}_manual1_weight.png')
            )

        if self.use_test_as_unlabeled_train_data:
            test_dir = os.path.join(self.data_dir, 'test')
            datasets += ['train'] * 20
            for imagename in [f'{i:02d}' for i in range(1, 21)]:
                image_paths.append(os.path.join(test_dir, 'images', f'{imagename}_test.tif'))
                fov_paths.append(os.path.join(test_dir, 'mask', f'{imagename}_test_mask.png'))
                annotation_paths.append(
                    os.path.join(test_dir, 'dummy-annotations', f'{imagename}_dummy.png')
                )
                weight_paths.append(
                    os.path.join(test_dir, 'dummy-annotations', f'{imagename}_dummy_weight.png')
                )
        
        with open(self.data_info_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['dataset', 'image_path', 'fov_path', 'annotation_path', 'weight_path'])
            writer.writerows(zip(datasets, image_paths, fov_paths, annotation_paths, weight_paths))
