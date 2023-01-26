'''See class RetinaDataModule'''
import torchio as tio
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
    def __init__(self, *args, use_soft_annotations=False, **kwargs):
        '''See BaseDataModule'''
        self.use_soft_annotations = use_soft_annotations
        super().__init__(*args, **kwargs)
                    
    def create_subject(self, row):
        '''Create a Subject with the following fields
          image : ScalarImage
        and optionally
          fov : LabelMap indicating the field of view
          annotation : LabelMap with annotations
        if the paths exist
        '''
        subject_kwargs = {
            'image' : tio.ScalarImage(row.image_path),
        }
        # pylint: disable=bare-except
        try:
            subject_kwargs['fov'] = tio.LabelMap(row.fov_path)
        except:
            pass
        try:
            if self.use_soft_annotations:
                subject_kwargs['annotation'] = tio.ScalarImage(row.annotation_path)
            else:
                subject_kwargs['annotation'] = tio.LabelMap(row.annotation_path)
        except:
            pass
        return tio.Subject(**subject_kwargs)
