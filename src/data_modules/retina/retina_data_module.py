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
    Subclasses must define the data_info_path attribute that must point to a csv file with the
    following columns:
        dataset,image_path,fov_path,annotation_path,weight_path
    where
      dataset has values in {train, validation, test, predict}
      image_path is the path to an image
      fov_path is the path to a field-of-view mask for the image
      annotation_path is the path to an annotation mask for the image
      weight_path is the path to an weight mask for the image
    '''
    def __init__(self):
        super().__init__()
        self.data_info_path = None
        self.augment = None
        self.preprocess = None
        self.datasets = {}
        self.batch_size = None
        self.num_workers = None
                    
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
