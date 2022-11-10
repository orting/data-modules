import numpy as np
import matplotlib.pyplot as plt
from data_modules.chest import OSICLungHeartTracheaDataModule
import torchio as tio

if __name__ == '__main__':
    transforms = {
        'train' :
        tio.Compose([
            tio.transforms.Resample((5,5,10)),
            tio.transforms.CropOrPad((100,100,10))
        ])
    }
    data_module = OSICLungHeartTracheaDataModule('osic_masks_data', 'osic_masks_data/data-info.csv', 10, False, transforms=transforms)
    data_module.prepare_data()
    data_module.setup()
    data_loader = data_module.train_dataloader()
    sample = next(iter(data_loader))

    print(sample['lung-valid'])
    print(sample['noisy-valid'])    
    print(sample['heart-valid'])
    print(sample['trachea-valid'])
    
    lung = sample['lung']['data']
    noisy = sample['noisy']['data']    
    heart = sample['heart']['data']
    trachea = sample['trachea']['data']

    print(lung.shape)
    print(noisy.shape)
    print(heart.shape)
    print(trachea.shape)

    fig, axs = plt.subplots(2,2)


    if sample['lung-valid'][0]:
        axs[0,0].imshow(lung[0,0,:,:,lung.shape[-1]//2])

    # if sample['lung-valid'][1]:
    #     axs[0,1].imshow(lung[1,0,:,:,lung.shape[-1]//2])
        


    
    if sample['heart-valid'][0]:
        axs[1,0].imshow(heart[0,0,:,:,heart.shape[-1]//2])
        
    # if sample['heart-valid'][1]:
    #     axs[1,1].imshow(heart[1,0,:,:,heart.shape[-1]//2])        

    plt.show()
                                              
    
