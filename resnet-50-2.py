#!/usr/bin/env python
# coding: utf-8

# ## Using ResNet50

# In[27]:


import glob
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.python.keras import models
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer


# ### Print the device configuration

# In[28]:


tf.keras.backend.clear_session()
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices()[0], end='')


# ### Get the images of train dataset

# In[29]:


def read_many(path):
    """
    Read all imagens in directory.
    
    Parameters
    ----------
    path: str
        Dataset path of a class (COVID or NON-COVID).
    
    Returns
    -------
    out : [ndarray]
        List of images.
    """
    # Get the path of all images
    list_imgs = list(glob.glob(path))
    out = []
    
    # Load all images of the given paths
    for i in range(len(list_imgs)):
        # Read the image in shape of (244, 244, 3)
        try:
            img = image.load_img(list_imgs[i], target_size=(224, 224, 3))
            x = image.img_to_array(img)
            out.append(x)
        # Print error
        except ValueError:
            print('Error reading the following image:', list_imgs[i])
    
    # Return the loaded images
    return out


def load_dir(paths):
    """
    Read images of COVID and NON-COVID cases.
    
    Parameters
    ----------
    paths: [str]
        Original and augmented dataset paths.
    
    Returns
    -------
    X : [ndarray]
        List of images.
    Y : [str]
        Labels of the images (i.e., 0 - NON-COVID; 1 - COVID).
    """
    # Arrays of images of COVID and NON-COVID cases
    covid = []
    non_covid = []
    
    # Read images
    for path in paths:
        # Read images of covid cases
        covid.extend(read_many('{}/COVID/*'.format(path)))
        # Read images of non-covid cases
        non_covid.extend(read_many('{}/NON_COVID/*'.format(path)))
    
    # Set COVID classes
    y_covid = np.asarray([1] * len(covid))
    y_non_covid = np.asarray([0] * len(non_covid))
    
    # Merge the read images
    X = np.concatenate([np.array(covid), np.array(non_covid)]) / 255
    Y = np.concatenate([np.array(y_covid), np.array(y_non_covid)])
    lb = LabelBinarizer()
    Y=lb.fit_transform(Y)
    
    assert len(X) == len(Y), 'The number of images and the number of classes are different!'
    print('Images read:', len(X))
    
    # Return the read images and their labels
    return (X, Y)


# In[30]:


TrainX, TrainY = load_dir(['Dataset/Train', 'Augmented/Train'])


# In[ ]:





# ### Convolutional Neural Network

# In[32]:


def CNN():
    """
    Return a Convolutional Neural Network (CNN) architecture.

    Returns
    -------
    model : Model
        The CNN model architeture.
    """
    # Create a new ResNet50
    conv_base = ResNet50(weights='imagenet',
                         # include_top = False,
                         # input_shape = (224,224, 3)
                        )
    conv_base.trainable = True
    # conv_base.summary()
    
    # model = conv_base
    # Instance of a sequential neural network
    model = models.Sequential()
    # Add the resnet50 to the sequential CNN
    model.add(conv_base)
    # Add a flaterns layer
    model.add(layers.Flatten())
    # Add a droput layer to avoid overfiting and minimize complexity
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(256, activation = 'relu'))
    # Add a final sigmoid layer for binary classification
    model.add(layers.Dense(1, activation = 'sigmoid'))
    # model.summary()
    
    # Compile the CNN model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'],
                    weighted_metrics=['accuracy']
                  # model configuration
                 )
    
    # Return the built CNN model
    return model, conv_base


# In[33]:


model,base = CNN()
model.summary()
#base.summary()


# In[ ]:





# ### Train the CNN

# In[34]:


model.fit(x=TrainX, y=TrainY, epochs=10)


# 

# In[36]:


print(model.predict(TrainX))

