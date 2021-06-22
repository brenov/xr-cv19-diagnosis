# Investigating the Impact of Using Image Processing Techniques in Chest X-Ray images for COVID-19 Diagnosis via Deep Learning


## Authors

- [Breno Maurício de Freitas Viana](https://github.com/brenov) (11920060)
- [Felipe Antunes Quirino](https://github.com/felipeaq) (12448645)


## Introduction

Since World Health Organization (WHO) recognized COVID-19 as a global threat, several works of different areas about the topic emerged.
Regarding abnormalities detection, numerous works try to classify COVID-19 cases from Chest X-Ray images and images of other lung image acquisition methods.
Most of these works applied Deep Learning (DL) techniques for predicting COVID-19 and non-COVID-19 cases in the original lung images.


## Objective

This work intends to investigate the impact of using image preprocessing techniques on the chest X-ray images in their classification in COVID-19 and non-COVID-19 cases.
We believe that such a process may improve the COVID-19 diagnosis performed by the ResNet-50 with the original chest X-ray images.


## Dataset

In this investigation, we are using the COVID-19 Chest X-ray Database from [Kaggle](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
(the dataset present in this repository is just a small subset of the COVID-19 Chest X-ray Database).
Furthermore, it was originally provided by Chowdhury et al. [1] and Rahman et al. [2].
This dataset contains chest X-ray images of healthy people (10,192) and people diagnosed with COVID-19, viral pneumonia (1,345), and lung opacity, i.e., non-COVID-19 lung infection (6,012).
All the images are in PNG file format, and their resolution is 299x299 pixels.
The following images present examples of such cases:

<table>
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


## References

[1] M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676. Paper DOI: (https://doi.org/10.1109/ACCESS.2020.3010287).

[2] Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images. Paper DOI: (https://doi.org/10.1016/j.compbiomed.2021.104319).
