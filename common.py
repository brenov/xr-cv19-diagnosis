"""This file holds common functions."""

import os


# Class of images
COVID_CLASS = 'COVID'
NON_COVID_CLASS = 'NON_COVID'

# Name of the folders of images
COVID_FOLDER = '/COVID'
NON_COVID_FOLDER = '/NON_COVID'


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
    if COVID_FOLDER in filename:
        return COVID_CLASS
    if NON_COVID_FOLDER in filename:
        return NON_COVID_CLASS
