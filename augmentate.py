#!/usr/bin/python3

import common
import augmentation as aug
from pathlib import Path
import imageio


# Dataset paths
DATASET_PATH = 'Dataset'
TRAIN_PATH = 'Train'
TEST_PATH = 'Test'

# Target directory
AUGMENTED_PATH = 'Augmented'

# PNG extension
PNG = '.png'


def load():
    """
    Load and return the paths of each image in the Dataset.
    """
    images = []
    for p in Path(DATASET_PATH).rglob('*' + PNG):
        images.append(str(p))
    return images


def augmentate(filenames):
    """
    Perform data augmentation of images.

    Parameters
    ----------
    filenames : [str]
        The filenames of the input images.
    """
    # Create folder for the augmented images
    common.create_folder(AUGMENTED_PATH)

    # For each image, generate new images
    for i in range(len(filenames)):
        # Get the image filename
        filename = filenames[i]

        # Load the image
        img = imageio.imread(filename)

        # Get image class
        cls = common.get_class(filename)

        # Get the image purpose
        purpose = TRAIN_PATH if TRAIN_PATH in filename else TEST_PATH

        # Remove path prefix and PNG extension
        filename = filename.replace(PNG, '') \
            .replace(DATASET_PATH + '/', '') \
            .replace(purpose + '/', '') \
            .replace(cls + '/', '')

        # Build the path prefix of the purpose (train or test)
        prefix = AUGMENTED_PATH + '/' + purpose + '/'
        # Create a folder correponding to the image class
        common.create_folder(prefix)

        # Build the path prefix of the image class
        prefix = prefix + cls + '/'
        # Create a folder correponding to the image class
        common.create_folder(prefix)

        # Update filename
        filename = prefix + filename

        # Adjust contrast 
        img1 = aug.adjust_contrast(img,1.4)
        imageio.imsave(filename + '_contrast' + PNG, img1)

        # Adjust sharpness 
        img2 = aug.adjust_sharpness(img,0.3,3,11)
        imageio.imsave(filename + '_sharpness' + PNG, img2)

        # Insert noise
        img3 = aug.add_noise(img, 10, 10)
        imageio.imsave(filename + '_noisy' + PNG, img3)

        # Rotate 15 degrees
        img4 = aug.rotate(img, 15)
        imageio.imsave(filename + '_15rotated' + PNG, img4)

        # Rotate -15 degrees
        img5 = aug.rotate(img, -15)
        imageio.imsave(filename + '_-15rotated' + PNG, img5)


def main():
    augmentate(load())


if __name__ == "__main__":
    main()
