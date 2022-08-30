'''Transformations that are not in TorchIO'''
import numpy as np
import torchio as tio
import albumentations as A

__all__ = [
    'RandomShiftRGB',
    'RGB2Gray',
    'WrapAlbumentations',
]

class RandomShiftRGB():
    # pylint: disable=too-few-public-methods
    '''Randomly shift RGB values. Use with torchio.transforms.Lambda
    
    Parameters
    ----------
    r_shift_limit, g_shift_limit, b_shift_limit : float, optional
      Maximum shift of each channel

    prob : float, optional
      Probability of shifting any channel.
    '''
    def __init__(self, r_shift_limit=0.078, g_shift_limit=0.078, b_shift_limit=0.078, prob=0.5):
        self.r_shift_limit = r_shift_limit
        self.g_shift_limit = g_shift_limit
        self.b_shift_limit = b_shift_limit
        self.prob = prob

    def __call__(self, image):
        '''
        Parameters
        ----------
        image : tensor of shape (3, ...)

        Returns
        --------
        tensor of shape (3, ...) with shifted values
        '''
        rng = np.random.default_rng()
        if rng.random() > self.prob:
            return image
        r_shift = rng.uniform(-self.r_shift_limit, self.r_shift_limit)
        g_shift = rng.uniform(-self.g_shift_limit, self.g_shift_limit)
        b_shift = rng.uniform(-self.b_shift_limit, self.b_shift_limit)
        image[0,...] += r_shift
        image[1,...] += g_shift
        image[2,...] += b_shift
        return image

    
class RGB2Gray():
    # pylint: disable=too-few-public-methods
    '''Convert RGB image to grayscale. Use with torchio.transforms.Lambda

    Default coefficients are from 
    Recommendation ITU-R BT.601-7
    Section 2.5.1 Construction of luminance (EY) and colour-difference (ER-EY) and (EB-EY) signals
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.601-7-201103-I!!PDF-E.pdf

    Parameters
    ----------
    r_coeff, g_coeff, b_coeff : float, optional
      Coefficients for each channel.
    '''
    def __init__(self, r_coeff=0.299, g_coeff=0.587, b_coeff=0.114):
        super().__init__()
        self.r_coeff = r_coeff
        self.g_coeff = g_coeff
        self.b_coeff = b_coeff

    def __call__(self, tensor):
        '''Apply grayscale transformation to tensor. Assumes channels first.

        Parameters
        ----------
        tensor : torch.tensor of shape (3, ...)

        Returns
        -------
        torch.tensor of shape (1, ...)
        '''
        if tensor.shape[0] == 1:
            return tensor
        return (self.r_coeff*tensor[0,...] + 
                self.g_coeff*tensor[1,...] +
                self.b_coeff*tensor[2,...]).unsqueeze(0)

    
    
class WrapAlbumentations(tio.transforms.augmentation.RandomTransform,
                         tio.SpatialTransform,
                         tio.IntensityTransform):
    '''Wrapper class to easily use transformations from the Albumentations library with TorchIO
    datasets.

    TODO: Consider making transform invertible when possible.
          Not obvious how this should be done nicely, unless each albumentation transform knows how
          it should be inverted.
          Even if the user supplies inverted versions of transforms, they still need the correct
          parameters that will often be randomly sampled for each image.

    Notes
    -----
    Many transforms in Albumentations assume 2D images.
    This transform is not invertible.

    Parameters
    ----------
    albumentations_augmentation : Albumentations.BasicTransform
    '''
    
    def __init__(self, albumentations_augmentation):
        super().__init__()
        self.aug = albumentations_augmentation

    def apply_transform(self, subject):
        '''
        Parameters
        ----------
        subject : torchio.Subject

        Returns
        -------
        torcio.Subject with transformed images
        '''
        # c/p from tio docs, but why do we do this? Is it just to fail gracefully?
        # TODO: Check TorchIO source for details
        subject.check_consistent_spatial_shape()
        targets = {}
        params = {}
        shapes = {}
        for name, image in subject.get_images_dict(intensity_only=False).items():
            if not isinstance(image, tio.ScalarImage):
                targets[f'{name}'] = 'mask'
            else:
                targets[f'{name}'] = 'image'
            shapes[f'{name}'] = image.shape
            params[f'{name}'] = image.numpy().squeeze()

        transform = A.Compose([self.aug], additional_targets=targets)
        transformed = transform(**params)

        for name, image in subject.get_images_dict(intensity_only=False).items():
            image.set_data(transformed[name].reshape(shapes[name]))
            
        return subject
