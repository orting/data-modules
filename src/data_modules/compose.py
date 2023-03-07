import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

__all__ = [
    'Compose'
]

class Compose(pl.LightningDataModule):
    '''Compose multiple data modules'''
    # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 data_modules,
                 batch_size,
                 num_workers=0,
                 extra_data_loader_kwargs=None
                 ):
        '''
        Parameters
        ----------       
         data_modules : sequence of BaseDataModule
        '''
        super().__init__()
        self.data_modules = data_modules
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.extra_data_loader_kwargs = \
            {} if extra_data_loader_kwargs is None else extra_data_loader_kwargs
        self.datasets = {}

    def prepare_data(self):
        for data_module in self.data_modules:
            data_module.pepare_data()
        
    def setup(self, stage=None):
        for data_module in self.data_modules:
            data_module.setup(stage)
        for ds_name in ['train', 'validation', 'test', 'predict']:            
            ds_list = []
            for data_module in self.data_modules:
                try:
                    ds_list.append(data_module.get_dataset(ds_name))
                except Exception:
                    pass
            if len(ds_list) > 0:
                self.datasets[ds_name] = ConcatDataset(ds_list)

    def _get_dataloader(self, ds_name, shuffle=False):
        if not ds_name in self.datasets:
            raise KeyError(f'No {ds_name} data available. '
                           'Did you forget to call prepare_data() or setup()?')
        return DataLoader(
            self.datasets[ds_name],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            **self.extra_data_loader_kwargs
        )
            
    def train_dataloader(self, shuffle=True):
        return self._get_dataloader('train', shuffle)
    
    def val_dataloader(self, shuffle=False):
        return self._get_dataloader('validation', shuffle)

    def test_dataloader(self, shuffle=False):
        return self._get_dataloader('test', shuffle)

    def predict_dataloader(self, shuffle=False):
        return self._get_dataloader('predict', shuffle)
                
