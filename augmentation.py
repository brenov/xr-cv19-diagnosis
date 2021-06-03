"""This file holds algorithms for data augmentation of images."""

import numpy as np


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
    # Initialize the output image
    out = np.zeros(img.shape)

    # Calculate the rotation matrix
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, s], [-s, c]])

    # Calculate image center
    center = ((img.shape[0]) // 2, (img.shape[1]) // 2)

    # Build matrix of points
    for x in range(img.shape[0]): # Row
        for y in range(img.shape[1]): # Col
            # Calculate the new position
            (nx, ny) = np.dot(R, np.array([x - center[1], y - center[0]]))
            # Add offset
            nx += center[0]
            ny += center[1]
            # Convert the new position to integer
            (nx, ny) = (int(nx), int(ny))
            # Ignore points out of bounds
            if nx >= 0 and ny >= 0 and nx < img.shape[0] and ny < img.shape[1]:
                # Set the pixel to its new position
                out[x, y] = img[nx, ny]

    # Return the rotated image
    return out.astype(np.uint8)
