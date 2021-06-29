#!/usr/bin/python3

import common
import augmentation as aug
from pathlib import Path
import imageio


# PNG extension
PNG = '.png'

# Dataset paths
DATASET_PATH = 'Dataset/Train'
TRAIN_PATH = 'Train'

# Target directory
AUGMENTED_PATH = 'Augmented'

# Processing techniques
CONTRAST = 'Contrast'
SHARPNESS = 'Sharpness'
ROTATION = 'Rotation'
NOISE = 'Noise'


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

    # Control the progress
    total = len(filenames)

    # For each image, generate new images
    for i in range(total):
        print('Augmentating {} of {}'.format(i + 1, total))

        # Get the image filename
        filename = filenames[i]

        # Load the image
        img = imageio.imread(filename)

        # Get image class
        cls = common.get_class(filename)

        # Remove path prefix and PNG extension
        filename = filename.replace(PNG, '') \
            .replace(DATASET_PATH + '/', '') \
            .replace(cls + '/', '')

        # Build the Augmented Dataset path
        prefix = AUGMENTED_PATH + '/'

        # Contrast

        # Build the path prefix of the image class
        prefixC = prefix + CONTRAST + '/'
        # Create a folder correponding to the image class
        common.create_folder(prefixC)

        # Build the path prefix of the image class
        prefixC = prefixC + cls + '/'
        # Create a folder correponding to the image class
        common.create_folder(prefixC)

        # Update filename
        filenameC = prefixC + filename

        # Adjust contrast
        img1 = aug.adjust_contrast(img, 1.2)
        imageio.imsave(filenameC + '_' + CONTRAST + '20' + PNG, img1)
        img1 = aug.adjust_contrast(img, 1.4)
        imageio.imsave(filenameC + '_' + CONTRAST + '40' + PNG, img1)

        # Sharpness

        # Build the path prefix of the image class
        prefixS = prefix + SHARPNESS + '/'
        # Create a folder correponding to the image class
        common.create_folder(prefixS)

        # Build the path prefix of the image class
        prefixS = prefixS + cls + '/'
        # Create a folder correponding to the image class
        common.create_folder(prefixS)

        # Update filename
        filenameS = prefixS + filename

        # Adjust sharpness
        img2 = aug.adjust_sharpness(img, 0.3, 3, 11)
        imageio.imsave(filenameS + '_' + SHARPNESS + '30-3-11' + PNG, img2)
        img2 = aug.adjust_sharpness(img, 0.1, 1, 5)
        imageio.imsave(filenameS + '_' + SHARPNESS + '10-1-5' + PNG, img2)

        # Noise

        # Build the path prefix of the image class
        prefixN = prefix + NOISE + '/'
        # Create a folder correponding to the image class
        common.create_folder(prefixN)

        # Build the path prefix of the image class
        prefixN = prefixN + cls + '/'
        # Create a folder correponding to the image class
        common.create_folder(prefixN)

        # Update filename
        filenameN = prefixN + filename

        # Insert noise
        img3 = aug.add_noise(img, 10, 10)
        imageio.imsave(filenameN + '_' + NOISE + '10' + PNG, img3)
        img3 = aug.add_noise(img, 20, 20)
        imageio.imsave(filenameN + '_' + NOISE + '20' + PNG, img3)

        # Rotation

        # Build the path prefix of the image class
        prefixR = prefix + ROTATION + '/'
        # Create a folder correponding to the image class
        common.create_folder(prefixR)

        # Build the path prefix of the image class
        prefixR = prefixR + cls + '/'
        # Create a folder correponding to the image class
        common.create_folder(prefixR)

        # Update filename
        filenameR = prefixR + filename

        img4 = aug.rotate(img, 15) # Rotate 15 degrees
        imageio.imsave(filenameR + '_' + ROTATION + '15' + PNG, img4)
        img5 = aug.rotate(img, -15) # Rotate -15 degrees
        imageio.imsave(filenameR + '_' + ROTATION + '-15' + PNG, img5)


def main():
    augmentate(load())


if __name__ == "__main__":
    main()
