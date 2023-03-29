from torch.utils.data import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
import imageio
import io

class CustomFloodDataset(Dataset):
    """
    Custom dataset for flood segmentation

    Parameters
    ----------
    Dataset : torch.utils.data.Dataset
        Dataset class from torch.utils.data
    """
    def __init__(self, image_list, mask_list, transform=None):
        self.image_list = image_list
        self.mask_list = mask_list
        self.transform = transform

    def __getitem__(self, idx):
        image = self.image_list[idx]
        mask = self.mask_list[idx]
        image = np.array(image)
        mask = np.array(mask)
        
        # mask = np.reshape(mask, mask.shape + (1,))
        if self.transform:
            augment = self.transform(image=image, mask=mask)
            image = augment['image']
            mask = augment['mask'] / 255
        return image, mask
    
    def __len__(self):
        return len(self.image_list)

class EarlyStopper:
    """
    Early stopping class
    """
    def __init__(self, patience:int = 0, verbose:bool = False,min_epoch:int = 0, model = None):
        """
        Init function for EarlyStopper class

        Parameters
        ----------
        patience : int, optional
            Patience to wait before stopping, by default 0
        verbose : bool, optional
            Only prints Early stopping
        min_epoch : int, optional
            Min epoch before starting to check for early stopping, by default 0
        model : torch.nn.Module
            Pytorch model
        """
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose
        self.model = model
        self.min_epoch = min_epoch
    def early_stop(self, epoch, loss):
        if epoch > self.min_epoch:
            if self._loss <= loss:
                self._step += 1
                if self._step > self.patience:
                    if self.verbose:
                        print("Early Stopping")
                    return True
            else:
                self._step = 0
                self._loss = loss
                self._store_best_model()
    
    def _store_best_model(self):
        self.best_model = deepcopy(self.model.state_dict())

    def load_best_model(self, model):
        model.load_state_dict(self.best_model)


def plot_image_mask_pred(test_idx, model, test_dataset, device=torch.device('cpu')):
    model.eval()
    with torch.no_grad():
        
        test_mask = model(test_dataset[test_idx][0].unsqueeze(0).to(device))

    _, ax =  plt.subplots(1, 3, figsize=(10, 10))

    ax[0].imshow(test_dataset[test_idx][0].permute(1, 2, 0))
    ax[0].axis('off')

    ax[1].imshow(test_dataset[test_idx][1])
    ax[1].axis('off')
    
    ax[2].imshow((test_mask.argmax(1)+1).squeeze(0).type(torch.uint8).cpu())
    ax[2].axis('off')


def create_gif(images,sample_image, gif_name,**kwargs):
    if gif_name.endswith('.gif') == False:
        gif_name = gif_name + '.gif'
    with imageio.get_writer(gif_name, mode='I',**kwargs) as writer:
        for i, image in enumerate(images):
            if i % 1 == 0:
                fig, ax = plt.subplots(1,2, figsize=(10,5))
                # Add the frame number to the image
                ax[0].imshow(sample_image)
                ax[0].set_title("Image")
                ax[0].axis('off')

                ax[1].imshow(image)
                ax[1].set_title(f"Frame {i}")
                ax[1].axis('off')
                
                # Save the current figure to a BytesIO object
                buf = io.BytesIO()
                plt.savefig(buf, format='jpg')
                buf.seek(0)
                
                # Add the image to the GIF
                image = imageio.v2.imread(buf)
                writer.append_data(image)
                plt.close()