"""This file holds algorithms for data augmentation of images."""

import numpy as np


BITS_8 = 255


def adjust_contrast(img):
    """
    Function to adjust the contrast of the input image.

    Parameters
    ----------
    img : ndarray
        Input image data.

    Returns
    -------
    out : ndarray
        Contrast adjusted version of the input image data.
    """
    # TODO
    print('WARNING: the function `adjust_contrast` was not implemented yet')
    return img


def adjust_sharpness(img):
    """
    Function to adjust the sharpness of the input image.

    Parameters
    ----------
    img : ndarray
        Input image data.

    Returns
    -------
    out : ndarray
        Sharpness adjusted version of the input image data.
    """
    # TODO
    print('WARNING: the function `adjust_sharpness` was not implemented yet')
    return img


def add_noise(img, mean, std):
    """
    Function to insert noise in the input image. The noise insertion is carried
    out with random generation based on Gaussian distribution.

    Parameters
    ----------
    img : ndarray
        Input image data.
    mean : float
        Mean of random distribution.
    std : float
        Standart deviation of random distribution.

    Returns
    -------
    out : ndarray
        Noisy version of the input image data.
    """
    # Generate a random noisy image
    noise = np.random.normal(mean, std, img.shape)
    # Insert the noise in the input image
    out = img + noise
    # Clip back to the original range
    out = np.clip(out, img.min(), img.max())
    # Return the noisy image
    return out.astype(np.uint8)


def rotate(img, angle):
    """
    Function to rotate the input image.

    Parameters
    ----------
    img : ndarray
        Input image data.
    angle : float
        Rotation angle.

    Returns
    -------
    out : ndarray
        Rotated version of the input image data.
    """
    # TODO
    print('WARNING: the function `adjust_sharpness` was not implemented yet')
    return img
