'''Example illustrating the use of CHASEDB1DataModule
This will download the CHASE DB1 dataset, which is 2.5MB
'''
import numpy as np
import matplotlib.pyplot as plt
from data_modules.retina import CHASEDB1DataModule

if __name__ == '__main__':
    data_module = CHASEDB1DataModule('chasedb1', 'chasedb1/data-info.csv', 1, True, True)
    data_module.prepare_data()
    data_module.setup()
    data_loader = data_module.train_dataloader()
    sample = next(iter(data_loader))
    image = np.moveaxis(sample['image']['data'].squeeze().numpy(), 0, -1)
    fov = sample['fov']['data'].squeeze().numpy()
    annotation = sample['annotation']['data'].squeeze().numpy()
    weight = sample['weight']['data'].squeeze().numpy()
    fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(image)
    axs[0,1].imshow(fov, cmap='gray')
    axs[1,0].imshow(annotation, cmap='gray')
    axs[1,1].imshow(weight, cmap='gray', vmin=0, vmax=1)
    plt.show()
