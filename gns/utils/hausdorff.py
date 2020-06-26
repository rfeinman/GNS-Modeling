"""
Taken from Omniglot repo:
https://github.com/brendenlake/omniglot/blob/9afc3137708d2c6e7b780ce3ed545492fcc873e3/python/one-shot-classification/demo_classification.py
"""
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist

from . import parallel


def modHausdorffDistance(itemA, itemB):
    """
    Modified Hausdorff Distance.

    M.-P. Dubuisson, A. K. Jain (1994). A modified hausdorff distance for object matching.
     International Conference on Pattern Recognition, pp. 566-568.

    :param itemA: [(n,2) array] coordinates of "inked" pixels
    :param itemB: [(m,2) array] coordinates of "inked" pixels
    :return dist: [float] distance
    """
    D = cdist(itemA, itemB)
    mindist_A = D.min(axis=1)
    mindist_B = D.min(axis=0)
    mean_A = np.mean(mindist_A)
    mean_B = np.mean(mindist_B)
    dist = max(mean_A,mean_B)
    return dist

def img_to_points(I):
    """
    Take an image and return coordinates of 'inked' pixels in the binary image.

    NOTE: this version assumes that 1 indicates "on" pixel and 0 indicates "off"
    (reversed from Omniglot github)

    :param I: [(H,W) array] image
    :return D: [(n,2) array] rows are coordinates
    """
    if not I.dtype == np.bool:
        I = I.astype(np.bool)
    (row,col) = I.nonzero()
    D = np.array([row,col])
    D = np.transpose(D)
    D = D.astype(float)
    n = D.shape[0]
    mean = D.mean(axis=0)
    for i in range(n):
        D[i,:] = D[i,:] - mean
    return D

def img_distance(img1, img2):
    p1 = img_to_points(img1)
    p2 = img_to_points(img2)
    dist = modHausdorffDistance(p1, p2)
    return dist



# ---- code for pairwise distances ----

def pairwise_distances(imgs1, imgs2, size=None, progbar=False):
    """

    :param imgs1: [(n,H,W) ndarray]
    :param imgs2: [(m,H,W) ndarray]
    :param size: [2-d tuple]
    :param progbar: [bool]
    :return D: [(n,m) ndarray]
    """
    # check inputs
    assert imgs1.shape[1:] == imgs2.shape[1:]
    imgs1 = check_images(imgs1)
    imgs2 = check_images(imgs2)

    # downsample images
    if size is not None:
        imgs1 = downsample_images(imgs1, size)
        imgs2 = downsample_images(imgs2, size)

    # convert images to ink locations
    imgs1 = parallel(img_to_points, imgs1)
    imgs2 = parallel(img_to_points, imgs2)

    # compute distances
    n = len(imgs1)
    m = len(imgs2)
    D = np.zeros((n,m))
    iterator = tqdm(range(n)) if progbar else range(n)
    for i in iterator:
        D[i] = parallel(
            f=modHausdorffDistance,
            x=[(imgs1[i], im2) for im2 in imgs2],
            starmap=True
        )

    return D

def check_images(imgs):
    assert len(imgs.shape) == 3
    assert imgs.shape[1] == imgs.shape[2]
    assert isinstance(imgs, np.ndarray)
    if not imgs.dtype == np.bool:
        imgs = imgs > 0.5
    return imgs

def downsample_images(x, size):
    """
    :param x: (n,H,W) ndarray
    :return x: (n,h,w) ndarray
    """
    assert isinstance(size, tuple)
    assert len(size) == 2

    x = torch.from_numpy(x).float() # (n,H,W)
    x = x.unsqueeze(1) # (n,1,H,W)
    x = F.interpolate(x, size=size, mode='bilinear', align_corners=False) # (n,1,h,w)
    x = x.squeeze(1) # (n,h,w)
    x = (x > 0.5).numpy()
    return x