import sklearn.metrics as metrics 
import torch
import numpy as np
from numpy import r_
import scipy



def dct2(im):
    """
    return jpeg dct of image
    """
    im2dct = lambda x : scipy.fftpack.dct( scipy.fftpack.dct( x, axis=0, norm='ortho' ), axis=1, norm='ortho' )
    im = rgb2ycb(np.array(im))
    imsize = im.shape
    dct = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            dct[i:(i+8),j:(j+8)] = im2dct( im[i:(i+8),j:(j+8)] - 128 )
    return torch.tensor(im, dtype= torch.float).permute(2,0,1)

def rgb2ycb(im): 
    out = np.zeros(im.shape) 
    out[:,:,0] = 0.299*im[:,:,0] + 0.587*im[:,:,1] + 0.114*im[:,:,2]
    out[:,:,1] = 128 -  0.168736*im[:,:,0] - 0.331264*im[:,:,1] + 0.5*im[:,:,2]
    out[:,:,2] = 128 + 0.5*im[:,:,0] - 0.418688*im[:,:,1] - 0.081312*im[:,:,2]
    return out

def get_available_devices():
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids



class AverageMeter:
    """Keep track of average values over time.

    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val#* num_samples
        self.avg = self.sum / self.count

def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2,   1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)
    print(fpr,tpr,thresholds)
    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)
    print(normalization)
    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        # pdb.set_trace()

        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min  # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric

    return competition_metric / normalization