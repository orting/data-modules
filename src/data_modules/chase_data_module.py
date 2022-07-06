'''See class CHASEDB1DataModule'''
import os
import csv
import random
from zipfile import ZipFile

import wget
import numpy as np
from skimage.io import imsave, imread

from .util import verify_sha256
from .retina_data_module import RetinaDataModule


__all__ = [
    'CHASEDB1DataModule'
    ]


class CHASEDB1DataModule(RetinaDataModule):
    '''Data module for loading CHASEDB1 data

    https://blogs.kingston.ac.uk/retinal/chasedb1/

    Christopher G. Owen, Alicja R. Rudnicka, Claire M. Nightingale, Robert Mullen, Sarah A. Barman,
    Naveed Sattar, Derek G. Cook, and Peter H. Whincup
    Retinal Arteriolar Tortuosity and Cardiovascular Risk Factors in a Multi-Ethnic Population Study
    of 10-Year-Old Children; the Child Heart and Health Study in England (CHASE)
    Arteriosclerosis, Thrombosis, and Vascular Biology, vol. 31, no. 8, pp. 1933-1938, 2011
    http://dx.doi.org/10.1161/ATVBAHA.111.225219
    '''
    # pylint: disable=too-many-instance-attributes, too-few-public-methods
    def __init__(self,
                 data_dir,
                 data_info_path,
                 batch_size,
                 download=False,
                 prepare_data_for_processing=False,
                 stratify_subjects=True,
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
          The directory containing CHASEDB1 data. Expects all files in one directory
        
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
          If True, merge annotations, create field-of-view (FOV) masks, create annotation weights

        stratify_subjects : bool, optional
          If True, stratify such that left/right images from same subject is in same dataset.
          If False, subjects are ignored for randomization.

        annotation_merge_style : one of {'bitmask', 'union', 'intersection'}
          There are two annotations for each annotated image. They need to be represented as a
          single image using one of the following strategies
          'bitmask' : Bitmask with 0 = bg, 1 = 1st, 2 = 2nd, 3 = 1st + 2nd
          'union' : The union of the two annotations
          'intersection' : The intersection of the two annotations

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

        Examples
        --------
        >>> import os
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from chase_data_module import CHASEDB1DataModule
        >>> datadir = 'tmp/chasedb1'
        >>> datafile = os.path.join(datadir, 'chasedb1-data-info.csv')
        >>> dm = CHASEDB1DataModule(datadir, datafile, 1, True, True)
        >>> dm.prepare_data()
        Downloading https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip
        100% [...................................................................] 2513926 / 2513926
        tmp/chasedb1/CHASEDB1.zip checksum match, Extracting to tmp/chasedb1.
        Merging annotations and creating field-of-view and weight maps
        >>> dm.setup()
        >>> dl = dm.train_dataloader()
        >>> sample = next(iter(dl))
        >>> image = np.moveaxis(sample['image']['data'].squeeze().numpy(), 0, -1)
        >>> fov = sample['fov']['data'].squeeze().numpy()
        >>> annotation = sample['annotation']['data'].squeeze().numpy()
        >>> weight = sample['weight']['data'].squeeze().numpy()
        >>> fig, axs = plt.subplots(2,2)
        >>> _ = axs[0,0].imshow(image)
        >>> _ = axs[0,1].imshow(fov, cmap='gray')
        >>> _ = axs[1,0].imshow(annotation, cmap='gray')
        >>> _ = axs[1,1].imshow(weight, cmap='gray', vmin=0, vmax=1)
        >>> plt.show()

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
        self.stratify_subjects = stratify_subjects
        self.annotation_merge_style = annotation_merge_style
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.datasets = {}
        self.num_workers = num_workers
        
    def prepare_data(self):
        '''Do the following
        1. Download data
        2. Merge annotations, create field-of-view masks and annotation weights
           FOVs are created by thresholding the images. See `_prepare_data_for_processing` for
           details.
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
        data_source = 'https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip'
        data_sha256 = 'bb750b73633492d700e56d91578ee3cdb7e4dbd08279e7465583e9efd26790b6'
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
        print('Merging annotations and creating field-of-view and weight maps')
        for side in ('L', 'R'):
            for image_number in range(1, 15):
                image = imread(os.path.join(self.data_dir, f'Image_{image_number:02d}{side}.jpg'))

                # FOV can be obtained by simple thresholding at 0
                fov_path = os.path.join(self.data_dir, f'Image_{image_number:02d}{side}_fov.png')
                # TODO: fov have a bit of noise at the boundary, consider fixing it.
                fov = (image.mean(-1) > 5).astype(dtype='uint8')
                imsave(fov_path, fov, check_contrast=False)
                weight_path = \
                    os.path.join(self.data_dir, f'Image_{image_number:02d}{side}_weight.png')
                imsave(weight_path, np.ones_like(fov), check_contrast=False)

                # Merge annotations
                ann1st = imread(
                    os.path.join(self.data_dir, f'Image_{image_number:02d}{side}_1stHO.png')
                ) > 0
                ann2nd = imread(
                    os.path.join(self.data_dir, f'Image_{image_number:02d}{side}_2ndHO.png')
                ) > 0
                if self.annotation_merge_style == 'bitmask':
                    annotation = ann1st.astype('uint8') + 2*ann2nd.astype('uint8')
                elif self.annotation_merge_style == 'union':
                    annotation = np.logical_or(ann1st, ann2nd).astype('uint8')
                else:
                    annotation = np.logical_and(ann1st, ann2nd).astype('uint8')
                annotation_path = \
                    os.path.join(self.data_dir, f'Image_{image_number:02d}{side}_merged.png')
                imsave(annotation_path, annotation, check_contrast=False)

                            
    def _create_data_info(self):
        '''There are 14 subjects in CHASEDB1, each with an image of the left and the right eye
        None have a field-of-view (FOV) mask.
        All have a reference segmentation
        '''
        if self.stratify_subjects:
            n_images = 14
        else:
            n_images = 28
        n_train = round(self.train_ratio * n_images)
        n_val = round(self.val_ratio * n_images)
        n_test = n_images - n_val - n_train
        datasets = ['train']*n_train + ['validation']*n_val + ['test']*n_test
        random.shuffle(datasets)
        if self.stratify_subjects:
            datasets = datasets * 2

        image_paths, fov_paths, annotation_paths, weight_paths = [], [], [], []
        for side in ('L', 'R'):
            for image_number in range(1, 15):
                image_paths.append(
                    os.path.join(self.data_dir, f'Image_{image_number:02d}{side}.jpg')
                )
                fov_paths.append(
                    os.path.join(self.data_dir, f'Image_{image_number:02d}{side}_fov.png')
                )
                weight_paths.append(
                    os.path.join(self.data_dir, f'Image_{image_number:02d}{side}_weight.png')
                )
                annotation_paths.append(
                    os.path.join(self.data_dir, f'Image_{image_number:02d}{side}_merged.png')
                )
        
        with open(self.data_info_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['dataset', 'image_path', 'fov_path', 'annotation_path', 'weight_path'])
            writer.writerows(zip(datasets, image_paths, fov_paths, annotation_paths, weight_paths))
