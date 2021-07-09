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
The ResNet-50 version of the ResNet model contains 48 Convolution layers, 1 MaxPool, and 1 Average Pool layer.


## Dataset

In this investigation, we are using the COVID-19 Chest X-ray Database from [Kaggle](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
(the dataset present in this repository ([Dataset](Dataset)) has 40 images, and it is just a small subset of the COVID-19 Chest X-ray Database).
Furthermore, it was originally provided by Chowdhury et al. [1] and Rahman et al. [2].
This dataset contains chest X-ray images of healthy people (10,192) and people diagnosed with COVID-19, viral pneumonia (1,345), and lung opacity, i.e., non-COVID-19 lung infection (6,012).
All the images are gray-scale in PNG file format, and their resolution is 299x299 pixels.
The following images present examples of such cases:

<table>
  <tr>
    <td>
      <img src=".github/Normal.png" alt="1" width=250px height=250px>
    </td>
    <td>
      <img src=".github/Lung_Opacity.png" alt="2" width=250px height=250px>
    </td>
  </tr>
  <tr>
    <td>Normal</td>
    <td>Lung Opacity</td>
  </tr>
  <tr>
    <td>
      <img src=".github/Viral_Pneumonia.png" alt="3" width=250px height=250px>
    </td>
    <td>
      <img src=".github/COVID.png" alt="4" width=250px height=250px>
    </td>
  </tr>
  <tr>
    <td>Viral Pneumonia</td>
    <td>COVID-19</td>
  </tr>
</table>

Since this is a preliminary stage of our research, we selected the first 1.000 images of each class from the database to perform our experiments.
Therefore, we selected 4.000 images, where 3.000 of them are NON-COVID cases, and the remaining are COVID cases.


## Methodology

To carry out our investigation, we first perform data augmentation by using the following image processing techniques: noise insertion, rotation, contrast adjustment, and sharpness adjustment.
Before the augmentation, we verify if they are represented in a single grayscale channel or RBG images.
In the latter case, we perform the luminance method to convert the images into a single grayscale channel.

These image processing methods are implemented in the [augmentation.py](augmentation.py) file.
The data augmentation is performed by the script [augmentate.py](augmentate.py).
This script automatizes the data augmentation process by reading all the images from [Dataset](Dataset) and creating eight new versions of each original image, two for each processing image technique.
We rotate the images in 15 degrees and -15 degrees; we opt for these values to simulate and (maybe) fix some badly positioned chest X-rays.
We generated images with two intensities of noise (both means and standard deviations equal to 5 and 10), contrast adjustment (factor equal to 1.1 and 1.2), and sharpness adjustment (factors equal to 0.1 and 0.3, sigma values equal to 1.5 and 3, and k values equal to 7.5 and 11) for the remaining techniques.
We avoid using too high values since, so far, we did not have expert advice.

- **Noise Insertion**: consists basically in inserting random pixels in the input image.
- **Contrast Adjustment**: comprises the following equation `128 + C * F - C * 128`, where `F` is the input image, `C` is the contrast level, and the value `128` is the mid-value the [0-255] range.
- **Sharpeness Adjustment**: consists in applying the equation `F + C * (F - G)`, where, `F` is the input image, `G` is the blurred version of the input image `F`, and `C` is the adjustment level.
- **Rotation**: We apply the [rotation equation](https://www.sciencedirect.com/topics/computer-science/image-rotation) to rotate the input image.
When we rotate `θ` angles the point `(x1, y1)` arround the point `(x0, y0)` (in our case, the image center) we get the new point `(x2, y2)`.
  - `x2 = cos(θ) * (x1 − x0) + sin(θ) * (y1 − y0)` calculates the new `x` position of each pixel.
  - `y2 = −sin(θ) * (x1 − x0) + cos(θ) * (y1 − y0)` calculates the new `y` position of each pixel.

After the data augmentation, we used both the original images (from [Dataset](Dataset)) and the images generated by the data augmentation techniques (from [Augmented](Augmented)) for training the ResNet-50.
In the following section, we describe our experiments and their results.


## Results

As we said earlier, we selected 4.000 images (1.000 COVID and 3.000 NON-COVID images) for training and testing the ResNet-50.
However, to keep in this repository ([Dataset](Dataset)), we selected only 28 images of those images.

The results are present in the folder [Augmented](Augmented) and in [Python file](resnet-50.py).
Fisrt, the folder [Augmented](Augmented) presents a total of 224 images all of them are augmented images generated from the [Dataset](Dataset).
The following images present some results of the data augmentation process.

<table>
  <tr>
    <td>
      <img src="Dataset/Train/COVID/COVID-1.png" alt="1" width=250px height=250px>
    </td>
  </tr>
  <tr>
    <td><b>Input image</b></td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="Augmented/Noise/COVID/COVID-1_Noise5.png" alt="2" width=250px height=250px>
    </td>
    <td>
      <img src="Augmented/Noise/COVID/COVID-1_Noise10.png" alt="2" width=250px height=250px>
    </td>
  </tr>
  <tr>
    <td>Noisy image #1 (mean: 5, std: 5)</td>
    <td>Noisy image #2 (mean: 10, std: 10)</td>
  </tr>
  <tr>
    <td>
      <img src="Augmented/Contrast/COVID/COVID-1_Contrast10.png" alt="2" width=250px height=250px>
    </td>
    <td>
      <img src="Augmented/Contrast/COVID/COVID-1_Contrast20.png" alt="2" width=250px height=250px>
    </td>
  </tr>
  <tr>
    <td>Image with contrast adjusted #1 (factor: 1.1)</td>
    <td>Image with contrast adjusted #2 (factor: 1.2)</td>
  </tr>
  <tr>
    <td>
      <img src="Augmented/Rotation/COVID/COVID-1_Rotation15.png" alt="2" width=250px height=250px>
    </td>
    <td>
      <img src="Augmented/Rotation/COVID/COVID-1_Rotation-15.png" alt="2" width=250px height=250px>
    </td>
  </tr>
  <tr>
    <td>Rotated image #1 (angle: 15 degrees)</td>
    <td>Rotated image #2 (angle: -15 degrees)</td>
  </tr>
  <tr>
    <td>
      <img src="Augmented/Sharpness/COVID/COVID-1_Sharpness10-1d5-7d5.png" alt="2" width=250px height=250px>
    </td>
    <td>
      <img src="Augmented/Sharpness/COVID/COVID-1_Sharpness30-3-11.png" alt="2" width=250px height=250px>
    </td>
  </tr>
  <tr>
    <td>Image with sharpness adjusted #1 (factor: 0.1, sigma: 1.5, k: 7.5)</td>
    <td>Image with sharpness adjusted #2 (factor: 0.3, sigma: 3, k: 11)</td>
  </tr>
</table>

Regarding the ResNet-50, we performed the following experiments:

 - Train the CNN with 3.200 original images.
 - Train the CNN with 3.200 original images and 6.400 noisy images (total of 9.600 images).
 - Train the CNN with 3.200 original images and 6.400 images with contrast adjusted (total of 9.600 images).
 - Train the CNN with 3.200 original images and 6.400 images with sharpness adjusted (total of 9.600 images).
 - Train the CNN with 3.200 original images and 6.400 rotated images (total of 9.600 images).

For all these cases, we tested with the remaining 800 original images andn 60 epochs.

<table>
  <tr>
    <td>
      <img src="Results/metrics.png" alt="1" width=400px height=300px>
    </td>
  </tr>
</table>

As we can observe in the above image, all but the experiment with sharpness-adjusted images presented the same balanced accuracy.
This particular experiment was the only one to outperform in both balanced accuracy and F1-score.

In terms of precision, all the experiments that used the images generated processing techniques beat the experiment with only original images.
However, concerning the recall metric, most experiments were worse than the experiment with only original images.
The single exception was the experiment with sharpness-adjusted images that tied with the experiment with only original images.

We believe that the augmented images increased the dataset unbalance since there are more NON-COVID images than the COVID ones.

<table>
  <tr>
    <td>
      <img src="Results/dist_None.png" alt="1" width=400px height=300px>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="Results/dist_Noise.png" alt="1" width=400px height=300px>
    </td>
    <td>
      <img src="Results/dist_Contrast.png" alt="2" width=400px height=300px>
    </td>
  </tr>
  <tr>
    <td>
      <img src="Results/dist_Sharpness.png" alt="3" width=400px height=300px>
    </td>
    <td>
      <img src="Results/dist_Rotation.png" alt="4" width=400px height=300px>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="Results/heatmap_None.png" alt="1" width=300px height=250px>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="Results/heatmap_Noise.png" alt="1" width=300px height=250px>
    </td>
    <td>
      <img src="Results/heatmap_Contrast.png" alt="2" width=300px height=250px>
    </td>
  </tr>
  <tr>
    <td>
      <img src="Results/heatmap_Sharpness.png" alt="3" width=300px height=250px>
    </td>
    <td>
      <img src="Results/heatmap_Rotation.png" alt="4" width=300px height=250px>
    </td>
  </tr>
</table>


## What We Learned

The dataset presents some irregularities, such as children's lungs only of NON-COVID cases, and the X-ray images were collected from different X-ray machines.
These irregularities may affect badly the training and, consequently, the quality resulting model.

Deep Learning approaches are very computationally expensive, and we needed a powerful machine to perform our experiments.
Fortunately, we did not have a problem finding such a powerful machine.
This machine was vital since we ran the first experiment with a gross error in our code, and we had time to rerun the experiment with the fixed code.

Besides, the error was that the path to augmented images was wrong.
Therefore, in our first experiment, we trained the ResNet-50 only with the 3.200 original images.
Still, the executions presented significantly different results for the same set of images.
Then we could conclude that the random parameters may influence a lot of the resulting model.
However, this could have occurred due to the small dataset we used (a subset of the original one).


## References

[1] M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676. Paper DOI: (https://doi.org/10.1109/ACCESS.2020.3010287).

[2] Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images. Paper DOI: (https://doi.org/10.1016/j.compbiomed.2021.104319).
