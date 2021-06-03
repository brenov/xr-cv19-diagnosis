#!/usr/bin/python3

import common
import imageio
import augmentation as aug


# Dataset root
DATASET_PATH = 'Dataset'

# Target directory
AUGMENTED_PATH = DATASET_PATH + '/Augmented'

# PNG extension
PNG = '.png'


def load():
    """
    Load and return the paths of each image in the Dataset.
    """
    # TODO
    return ['Dataset/COVID/COVID-1.png']


def augmentate(filenames):
    """
    Perform data augmentation on images.

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

        # Remove path prefix and PNG extension
        filename = filename.replace(PNG, '') \
            .replace(DATASET_PATH + '/', '') \
            .replace(cls + '/', '')

        # Build the new path prefix
        prefix = AUGMENTED_PATH + '/' + cls + '/'

        # Create a folder correponding to the image class
        common.create_folder(prefix)

        # Adjust contrast - WARNING: the function `adjust_contrast` was not implemented yet
        img1 = aug.adjust_contrast(img)
        imageio.imsave(prefix + filename + '_contrast' + PNG, img1)

        # Adjust sharpness - WARNING: the function `adjust_sharpness` was not implemented yet
        img2 = aug.adjust_sharpness(img)
        imageio.imsave(prefix + filename + '_sharpness' + PNG, img2)

        # Insert noise
        img3 = aug.add_noise(img, 10, 10)
        imageio.imsave(prefix + filename + '_noisy' + PNG, img3)

        # Rotate 25 degrees
        img4 = aug.rotate(img, 25)
        imageio.imsave(prefix + filename + '_25rotated' + PNG, img4)

        # Rotate -25 degrees
        img5 = aug.rotate(img, -25)
        imageio.imsave(prefix + filename + '_-25rotated' + PNG, img5)


def main():
    augmentate(load())


if __name__ == "__main__":
    main()
