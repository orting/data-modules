'''See class RetinaDataModule'''
import pandas as pd
import pytorch_lightning as pl
import torchio as tio
from torch.utils.data import DataLoader
from ..base_data_module import BaseDataModule

__all__ = [
    'RetinaDataModule'
]

class RetinaDataModule(BaseDataModule):
    '''Base class for data modules loading retina data
    
    Implements `create_subject` as required by BaseDataModule. Subclasses should implement the
    remaining LightningDataModule key methods, see documentation for LightningDataModule. 
    '''
    # pylint: disable=too-many-instance-attributes
    def __init__(self, *args, **kwargs):
        '''See BaseDataModule'''
        super().__init__(*args, **kwargs)
                    
    def create_subject(self, row):
        '''Create a Subject with the following fields
        image : ScalarImage
        fov : LabelMap indicating the field of view
        annotation : LabelMap with annotations
        weight : LabelMap with pixel weights
        '''
        subject_kwargs = {
            'image' : tio.ScalarImage(row.image_path),
            'fov' : tio.LabelMap(row.fov_path),
            'annotation' : tio.LabelMap(row.annotation_path),
            'weight' : tio.LabelMap(row.weight_path),
        }
        return tio.Subject(**subject_kwargs)
