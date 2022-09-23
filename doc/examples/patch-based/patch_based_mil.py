'''See class RetinaDataModule'''
import os
import csv
import pytorch_lightning as pl
import torchio as tio
import torch

from data_modules.base_data_module import BaseDataModule

class PatchBasedMIL(BaseDataModule):
    '''Implements `create_subject` as required by BaseDataModule. Subclasses should implement the
    remaining LightningDataModule key methods, see documentation for LightningDataModule. 
    '''
    # pylint: disable=too-many-instance-attributes
    def __init__(self, *args, **kwargs):
        '''See BaseDataModule'''
        super().__init__(*args, **kwargs)
                    
    def create_subject(self, row):
        '''Create a Subject with the following fields
        image : ScalarImage
        label : torch.tensor
        '''
        subject_kwargs = {
            'image' : tio.ScalarImage(type=tio.INTENSITY, tensor=torch.randn((1,512,512,300))),
            'id' : torch.IntTensor([row.id]),
            'label' : torch.FloatTensor([row.label]),
        }
        return tio.Subject(**subject_kwargs)
    
    def prepare_data(self):
        '''Create data info file with 6 subjects'''
        n = 3
        datasets = ['train']*n*2
        ids = range(n*2)
        labels = [1,0]*n
        
        with open(self.data_info_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['dataset', 'id', 'label'])
            writer.writerows(zip(datasets, ids, labels))            


def main():
    dm = PatchBasedMIL('pbm_data_info.csv',
                       generate_patches=True,
                       batch_size=20,         # To ensure patches from the same image are in the
                       queue_length=60,       # same batch, we need to have batch_size and
                       samples_per_volume=10, # queue_size be multiples of samples_per_volume and
                       shuffle_patches=False, # shuffle_patches must be False.
                       patch_size=(64,64,38))
    dm.prepare_data()
    dm.setup()
    dl = dm.train_dataloader()
    for sample in dl:
        print(list(sample.keys()))
        print(sample['image']['data'].shape)
        print(torch.cat([sample['id'], sample['label']], 1))
        print(sample['location'].shape)
        print(sample['location'])

if __name__ == '__main__':
    main()
