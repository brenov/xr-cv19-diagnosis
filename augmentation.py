"""This file holds algorithms for data augmentation of images."""

import numpy as np


def adjust_contrast(img):
    """
    Function to adjust the contrast of the given image.

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
    pass


def adjust_sharpness(img):
    """
    Function to adjust the sharpness of the given image.

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
    pass


def add_noise(img, mean, std):
    """
    Function to insert noise in the given image. The noise insertion is carried
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
    # Generate random noise
    noise = np.random.normal(mean, std, img.shape)
    # Insert noise in the given image
    out = img + noise
    # Clip back to the original range
    out = np.clip(out, img.min(), img.max())
    # Return the noisy image
    return out.astype(np.uint8)


def clockwise_rotation(img):
    """
    Function to rotate the given image 45 degrees clockwise.

    Parameters
    ----------
    img : ndarray
        Input image data.

    Returns
    -------
    out : ndarray
        45 degrees rotated version of the input image data.
    """
    # TODO
    pass


def anticlockwise_rotation(img):
    """
    Function to rotate the given image 45 degrees anticlockwise.

    Parameters
    ----------
    img : ndarray
        Input image data.

    Returns
    -------
    out : ndarray
        -45 degrees rotated version of the input image data.
    """
    # TODO
    pass
