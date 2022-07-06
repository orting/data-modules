'''See class STAREDataModule'''
import os
import tarfile
import gzip
import csv
import random
from zipfile import ZipFile

import wget
import numpy as np
from skimage.io import imread, imsave

from .util import verify_sha256
from .retina_data_module import RetinaDataModule

__all__ = [
    'STAREDataModule'
    ]


class STAREDataModule(RetinaDataModule):
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
    # pylint: disable=too-many-instance-attributes, too-few-public-methods
    def __init__(self,
                 data_dir,
                 data_info_path,
                 batch_size,
                 download=False,
                 prepare_data_for_processing=False,
                 annotation_merge_style='bitmask',
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
          If `download` is True, data will be downloaded to this directory.
          If `download` is False the directory should contain the STARE data in the following
          subdirectories
            images/    : Directory with the 397 images from STARE. 
            labels-ah/ : Directory with 20 manual segmentations by Adam Hoover
            labels-vk/ : Directory with 20 manual segmentations by Valentina Kouznetsova

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
          If True, convert ppm to png, create empty annotations for image that do not have
          annotations, create annotation weights, merge_annotations

        annotation_merge_style : one of {'bitmask', 'union', 'intersection'}
          There are two annotations for each annontated image. They need to be represented as a
          single image using one of the following strategies
          'bitmask' : Bitmask with 0 = bg, 1 = AH, 2 = VK, 3 = AH + VK
          'union' : The union of the two annotations
          'intersection' : The intersection of the two annotations
                          

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
        self.data_dir = data_dir
        self.preprocess = preprocessing_transform
        self.augment = augmentation_transform
        self.data_info_path = data_info_path
        self.batch_size = batch_size
        self.download = download
        self.prepare_data_for_processing = prepare_data_for_processing
        self.annotation_merge_style = annotation_merge_style
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.labeled = [1,2,3,4,5,44,77,81,82,139,162,163,235,236,239,240,255,291,319,324]
        self.missing = [47,108,109,144,167]
        self.unlabeled = [
            i for i in range(1, 403) if not i in self.labeled and not i in self.missing
        ]
        
    def prepare_data(self):
        '''Do the following
        1. Download data
        2. Convert ppm to png, create empty annotations for image that do not have
           annotations, create annotation weights, merge_annotations
        3. Create data info file
        '''
        if self.download:
            self._download()

        if self.prepare_data_for_processing:
            self._prepare_data_for_processing()

        if not os.path.exists(self.data_info_path):
            self._create_data_info()
                
    def _download(self):
        image_dir = os.path.join(self.data_dir, 'images')
        label_dir_ah = os.path.join(self.data_dir, 'labels-ah')
        label_dir_vk = os.path.join(self.data_dir, 'labels-vk')
        for directory in [image_dir, label_dir_ah, label_dir_vk]:
            os.makedirs( directory, exist_ok=True)
           
        image_src = 'https://cecas.clemson.edu/~ahoover/stare/images/all-images.zip'
        print('Downloading', image_src)
        image_file = wget.download(image_src, self.data_dir)
        image_file_sha256 = '6428ecc394f1b49a7192134990934f62fcd7d36110fd7c344b912dc43925e853'
        if not verify_sha256(image_file, image_file_sha256):
            raise ValueError(
                f'Downloaded all-images.zip file: "{image_file}" does not match expected checksum'
            )
        print(f'{image_file} checksum match')
        with ZipFile(image_file) as archive:
            archive.extractall(image_dir)

        ah_src = 'https://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar'
        print('Downloading', ah_src)
        ah_file = wget.download(ah_src, self.data_dir)
        ah_file_sha256 = 'ebf2f1e17ca955f24579d9edd990e2dae79a5c82def69f0985d8e24f826ddd2f'
        if not verify_sha256(ah_file, ah_file_sha256):
            raise ValueError(
                f'Downloaded labels-ah.tar file: "{ah_file}" does not match expected checksum'
            )
        print(f'{ah_file} checksum match')
        self._extract_label_archive(ah_file, label_dir_ah)

        vk_src = 'https://cecas.clemson.edu/~ahoover/stare/probing/labels-vk.tar'
        print('Downloading', vk_src)
        vk_file = wget.download(vk_src, self.data_dir)
        vk_file_sha256 = '47474a701536b0cfdb369fdce012be36141e9f44d80387f0179446b5cb0f5576'
        if not verify_sha256(vk_file, vk_file_sha256):
            raise ValueError(
                f'Downloaded labels-vk.tar file: "{vk_file}" does not match expected checksum'
            )
        print(f'{vk_file} checksum match')
        self._extract_label_archive(vk_file, label_dir_vk)
        
        
    def _extract_label_archive(self, archive_path, outdir):
        '''The STARE label files are tar archives containing gzipped ppm images.
        This method will gunzip the ppm images outdir
        We trust that the archive is as expected, so only call this if the check sums match
        '''
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
        # pylint: disable=too-many-locals
        # The images are stored as 2D 8-bit unsiged RGB images in ppm format. We want to use
        # TorchIO that does not handle ppm, so we convert everything to png images that are
        # handled by TorchIO.
        # Only 20 images have reference segmentations, so to simplify processing we create fake
        # segmentations for the rest and use a weight map to control which pixel predictions should
        # be included in the loss.
        assert self.annotation_merge_style in ('bitmask', 'union', 'intersection')
        
        image_dir = os.path.join(self.data_dir, 'images')
        fov_dir = os.path.join(self.data_dir, 'fov')
        label_dir_ah = os.path.join(self.data_dir, 'labels-ah')
        label_dir_vk = os.path.join(self.data_dir, 'labels-vk')
        annotation_dir = os.path.join(self.data_dir, 'labels-merged')
        for directory in [fov_dir, annotation_dir]:
            os.makedirs(directory, exist_ok=True)

        for imagename in [f'im{i:04d}' for i in self.labeled + self.unlabeled]:
            ppm_path = os.path.join(image_dir, imagename + '.ppm')
            png_path = os.path.join(image_dir, imagename + '.png')
            image = imread(ppm_path)
            imsave(png_path, image, check_contrast=False)
            os.remove(ppm_path)

            ah_path = os.path.join(label_dir_ah, f'{imagename}.ah.ppm' )
            vk_path = os.path.join(label_dir_vk, f'{imagename}.vk.ppm' )
            if os.path.exists(ah_path):
                ah_mask = (imread(ah_path) > 0).astype('uint8')
            else:
                ah_mask = np.zeros(image.shape[:2], dtype='uint8')
            if os.path.exists(vk_path):
                vk_mask = (imread(vk_path) > 0).astype('uint8')
            else:
                vk_mask = np.zeros(image.shape[:2], dtype='uint8')

            # Merge annotations
            if self.annotation_merge_style == 'bitmask':
                mask = ah_mask + 2*vk_mask
            elif self.annotation_merge_style == 'union':
                mask = np.logical_or(ah_mask, vk_mask).astype('uint8')
            else:
                mask = np.logical_and(ah_mask, vk_mask).astype('uint8')
            annotation_path = os.path.join(annotation_dir,
                                           f'{imagename}.{self.annotation_merge_style}.png')
            imsave(annotation_path, mask, check_contrast=False)

            # Create weights
            if np.any(mask):
                weights = np.ones_like(mask)
            else:
                weights = np.zeros_like(mask)
            weights_path = os.path.join(annotation_dir,
                                        f'{imagename}.{self.annotation_merge_style}.weight.png')
            imsave(weights_path, weights, check_contrast=False)

            # Create FOVs
            # TODO: Find FOVs
            fov = np.ones_like(mask)
            fov_path = os.path.join(fov_dir, f'{imagename}.fov.png')
            imsave(fov_path, fov, check_contrast=False)

            
    def _create_data_info(self):
        # pylint: disable=too-many-locals
        assert self.annotation_merge_style in ('bitmask', 'union', 'intersection')

        random.shuffle(self.labeled)
        n_unlabeled = len(self.unlabeled)
        n_labeled = len(self.labeled)
        n_train = round(self.train_ratio * n_labeled)
        n_val = round(self.val_ratio * n_labeled)
        n_test = n_labeled - n_val - n_labeled
        datasets = ['train']*(n_unlabeled+n_train) + ['validation']*n_val + ['test']*n_test
        
        image_paths, fov_paths, annotation_paths, weight_paths = [], [], [], []
        image_dir = os.path.join(self.data_dir, 'images')
        fov_dir = os.path.join(self.data_dir, 'fov')        
        annotation_dir = os.path.join(self.data_dir, 'labels-merged')        
        for im_number in self.unlabeled + self.labeled:
            im_name = f'im{im_number:04d}'
            image_paths.append(os.path.join(image_dir, f'{im_name}.png'))
            fov_paths.append(os.path.join(fov_dir, f'{im_name}.fov.png'))
            annotation_paths.append(
                os.path.join(annotation_dir, f'{im_name}.{self.annotation_merge_style}.png')
            )
            weight_paths.append(
                os.path.join(annotation_dir, f'{im_name}.{self.annotation_merge_style}.weight.png')
            )            

        with open(self.data_info_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['dataset', 'image_path', 'fov_path', 'annotation_path', 'weight_path'])
            writer.writerows(zip(datasets, image_paths, fov_paths, annotation_paths, weight_paths))
            
