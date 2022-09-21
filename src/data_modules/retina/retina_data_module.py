'''See class RetinaDataModule'''
import pandas as pd
import pytorch_lightning as pl
import torchio as tio
from torch.utils.data import DataLoader

__all__ = [
    'RetinaDataModule'
]

class RetinaDataModule(pl.LightningDataModule):
    '''Base class for data modules loading retina data
    Subclasses should implement the remaining LightningDataModule key methods, see documentation for
    LightningDataModule. 
    Subclasses must define the data_info_path attribute that must point to a 
    '''
    # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 data_info_path,
                 batch_size,
                 transforms=None,
                 num_workers=0,
                 generate_patches=False,
                 patch_size=(128,128),
                 queue_length=300,
                 samples_per_volume=10,
                 shuffle_patches=True,
                 ):
        '''
        Parameters
        ----------       
        data_info_path : str
            csv file with the following columns:
            dataset,image_path,fov_path,annotation_path,weight_path
            where
            dataset has values in {train, validation, test, predict}
            image_path is the path to an image
            fov_path is the path to a field-of-view mask for the image
            annotation_path is the path to an annotation mask for the image
            weight_path is the path to an weight mask for the image

        batch_size : int
          Batch size

        transforms : dict of callables, optional
          Dictionary of transforms to apply to each dataset. Example: If
            transforms = {'train' : <some-transform>}
          then <some-transform> will be applied to the "train" dataset and all other datasets will
          not be transformed.

        num_workers : int, optional
          How many workers to use for generating data          

        generate_patches : bool, optional
          If True, sample patches from the images instead of generating full images.
          The following parameters are used to control the patch generation:
            patch_size, queue_length, samples_per_volume, shuffle_patches
          See https://torchio.readthedocs.io/patches/patch_training.html for more info.
          
        patch_size : tuple of int, optional
          Size of generated patches. Ignored unless generate_patches = True

        queue_length : int, optional
          Maximum number of patches to hold in the queue.

        samples_per_volume : int, optional
          Number of samples per image

        shuffle_patches : bool, optional
          If True, the queue will be shuffled before batches are generated from it

        Notes
        -----
        If a patch-based pipeline is used and all patches from an image should be in the same batch,
        then batch_size, queue_length, samples_per_volume and shuffle_patches need to be set
        as follows
          shuffle_patches must be False.
          batch_size and queue_length must be multiples of samples_per_volume.

        For example,
          shuffle_patches = False
          batch_size = 16
          samples_per_volume = 8
          queue_length = 64
        Then each batch will contain patches from two subjects, the first 8 from one subject and the
        last 8 from the other subject.
        '''
        # pylint: disable=too-many-arguments
        super().__init__()
        self.data_info_path = data_info_path
        self.batch_size = batch_size
        self.transforms = {} if transforms is None else transforms
        self.num_workers = num_workers
        self.generate_patches = generate_patches
        self.patch_size = patch_size
        self.queue_length = queue_length
        self.samples_per_volume = samples_per_volume
        self.shuffle_patches = shuffle_patches
        self.patch_sampling_label_name = 'fov'
        self.datasets = {}
                    
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
            subject_kwargs = {
                'image' : tio.ScalarImage(row.image_path),
                'fov' : tio.LabelMap(row.fov_path),
                'annotation' : tio.LabelMap(row.annotation_path),
                'weight' : tio.LabelMap(row.weight_path),
            }
            try:
                samples[row.dataset].append(tio.Subject(**subject_kwargs))
            except KeyError:
                print('Uknown dataset', row.dataset, 'Expected train, validation, test or predict')
                continue
            
        for ds_name in ['train', 'validation', 'test', 'predict']:
            if len(samples[ds_name]) > 0:
                transform = self.transforms[ds_name] if ds_name in self.transforms else None
                self.datasets[ds_name] = tio.SubjectsDataset(samples[ds_name],transform=transform)

                
    def _get_dataloader(self, ds_name, shuffle=False):
        if not ds_name in self.datasets:
            raise ValueError(f'No {ds_name} data available. Ensure that at least one row in '
                             f'"{self.data_info_path}" has dataset = {ds_name}.\n'
                             'Did you forget to call prepare_data() or setup()?')
        if self.generate_patches:
            return self._get_patch_dataloader(ds_name, shuffle)
        return self._get_fullimage_dataloader(ds_name, shuffle)

    
    def _get_fullimage_dataloader(self, ds_name, shuffle=False):
        return DataLoader(self.datasets[ds_name],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=shuffle)

    
    def _get_patch_dataloader(self, ds_name, shuffle=False):
        sampler = tio.data.LabelSampler(
            self.patch_size,
            self.patch_sampling_label_name,
        )
        queue = tio.Queue(
            self.datasets[ds_name],
            self.queue_length,
            self.samples_per_volume,
            sampler,
            num_workers=self.num_workers,
            shuffle_subjects=shuffle,
            shuffle_patches=self.shuffle_patches,
        )
        return DataLoader(queue,
                          batch_size=self.batch_size,
                          num_workers=0)
        
        
    def train_dataloader(self):
        return self._get_dataloader('train', True)
    
    def val_dataloader(self):
        return self._get_dataloader('validation', False)

    def test_dataloader(self):
        return self._get_dataloader('test', False)

    def predict_dataloader(self):
        return self._get_dataloader('predict', False)
