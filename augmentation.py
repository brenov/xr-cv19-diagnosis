"""This file holds algorithms for data augmentation of images."""

import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift

def adjust_contrast(img,factor):
    """
    Function to adjust the contrast of the input image.

    Parameters
    ----------
    img : ndarray
        Input image data.

    factor: float
        Input contrast factor

    Returns
    -------
    out : ndarray
        Contrast adjusted version of the input image data.
    """
    factor=float(factor)
    return np.clip(128 + factor * img - factor * 128, 0, 255).astype(np.uint8)



def gaussian_filter (k,sigma) :

    arx = np . arange ((-k // 2 ) + 1.0 , ( k // 2 ) + 1.0 )
    x , y = np . meshgrid ( arx , arx )
    filt = np.exp ( -(1/2)*( np.square(x) + np.square ( y ) ) / np.square ( sigma ) )
    
    return filt/np.sum(filt)*255


def apply(f,sigma,k):
    '''blur function'''

    h = gaussian_filter(k=k, sigma=sigma)
    # computing the number of padding on one side
    a = int(f.shape[0]//2 - h.shape[0]//2)
    h_pad = np.pad(h, ((a,a), (a,a)), 'constant', constant_values=(0))

    # computing the Fourier transforms
    F = fftn(f)
    H = fftn(h_pad)
    

    # convolution
    G = np.multiply(F,H)

    # Inverse Transform
    # - we have to perform FFT shift before reconstructing the image in the space domain
    g = fftshift(ifftn(G).real)
    

    return normalize(g)

def normalize(img):
    img-=img.min()
    img= img/img.max()*255
    return img

def adjust_sharpness(img,amount,sigma,k):
    """
    Function to adjust the sharpness of the input image.

    Parameters
    ----------
    img : ndarray
        Input image data.

    alpha: float
        image offset
    sigma1: float
        first standard deviation
    sigma2: float
        second standard deviation

    Returns
    -------
    out : ndarray
        Sharpness adjusted version of the input image data.
    """
    blur = normalize(apply(img, sigma,k))
    sharpened = img + amount * (img - blur)
    out= normalize(sharpened)
    return out.astype(np.uint8)



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

    # Perform image rotation
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
