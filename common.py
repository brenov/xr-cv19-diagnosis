"""This file holds common functions."""

import os


# Class of images
NORMAL_CLASS = 'Normal'
COVID_CLASS = 'COVID'
LUNG_OPACITY_CLASS = 'Lung_Opacity'
VIRAL_PNEUMONIA_CLASS = 'Viral_Pneumonia'


def create_folder(foldername):
    """
    If there is no folder with the input folder name, then create it.

    Parameters
    ----------
    foldername : str
        The name of the folder to be created.
    """
    if not os.path.isdir(foldername):
        os.mkdir(foldername)


def get_class(filename):
    """
    Return the class of the input image through its filename.

    Parameters
    ----------
    filename : str
        Input image filename.

    Returns
    -------
    out : str
        The class of the input image.
    """
    if NORMAL_CLASS in filename:
        return NORMAL_CLASS
    if COVID_CLASS in filename:
        return COVID_CLASS
    if LUNG_OPACITY_CLASS in filename:
        return LUNG_OPACITY_CLASS
    if VIRAL_PNEUMONIA_CLASS in filename:
        return VIRAL_PNEUMONIA_CLASS
