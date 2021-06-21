# Investigating the Impact of Using Image Processing Techniques in Chest X-Ray images for COVID-19 Diagnosis via Deep Learning

## Motivation

Since World Health Organization (WHO) recognized COVID-19 as a global threat, several works of different areas about the topic emerged.
Regarding abnormalities detection, numerous works try to classify COVID-19 cases from Chest X-Ray images and images of other lung image acquisition methods.
Most of these works applied Deep Learning (DL) techniques for predicting COVID-19 cases in the original images.


## Objective

This work intends to investigate the impact of using image preprocessing techniques on the chest X-ray images in their classification in COVID-19 and non-COVID-19 cases.
We believe that such a process may improve the COVID-19 diagnosis performed by the ResNet-50.


## Dataset

COVID-19 Chest X-ray Database (https://www.kaggle.com/tawsifurrahman/covid19-radiography-database). The dataset present in this repository is just a small subset of the COVID-19 Chest X-ray Database.

### Example of input images

<table >
  <tr>
    <td>
      <img src=".github/Normal.png" alt="1" width=200px height=200px>
    </td>
    <td>
      <img src=".github/Lung_Opacity.png" alt="2" width=200px height=200px>
    </td>
    <td>
      <img src=".github/Viral_Pneumonia.png" alt="3" width=200px height=200px>
    </td>
    <td>
      <img src=".github/COVID.png" alt="4" width=200px height=200px>
    </td>
   </tr>
   <tr>
      <td>Normal</td>
      <td>Lung Opacity</td>
      <td>Viral Pneumonia</td>
      <td>COVID</td>
  </tr>
</table>


## Methodology

To do so, we perform data augmentation by using the following image processing techniques: rotation, contrast adjustment, sharpness enhancement, and noise insertion.
The generated images are used for both ResNet-50 training and test.
<!--- After data this stage, we perform the fine-tuning of the neural network aiming to improve the results.) --->


## Partial Results

The partial results are present in the folder [Augmented](Augmented) and in [Jupyter Notebook file](resnet-50-2.ipynb).
The folder [Augmented](Augmented) presents the augmented images generated from the [Dataset](Dataset).
The [Jupyter Notebook file](resnet-50-2.ipynb) presents the training of a CNN with images from both [Dataset](Dataset) and [Augmented](Augmented) folders.

### Augmented Images

TODO
<!--- [augmentate.py](augmentate.py) --->
<!--- [augmentation.py](augmentation.py) --->

### CNN Training

TODO


## Authors

- Breno Maurício de Freitas Viana
- Felipe Antunes Quirino
