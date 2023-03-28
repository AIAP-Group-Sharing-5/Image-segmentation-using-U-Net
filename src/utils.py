from torch.utils.data import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    """
    Custom dataset for sidewalk segmentation

    Parameters
    ----------
    Dataset : torch.utils.data.Dataset
        Dataset class from torch.utils.data
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = np.array(self.dataset['pixel_values'][idx])
        mask = np.array(self.dataset['label'][idx]) -1
        if self.transform:
            augment = self.transform(image=image, mask=mask)
            image = augment['image']
            mask = augment['mask']
        return image, mask
    
from copy import deepcopy
class EarlyStopper:
    def __init__(self, patience = 0, verbose = False,min_epoch = 0, model = None):
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