# Master thesis progress report
This repository is used as a tracker for my master's thesis which I am currently working on.
The main purpose is to improve crack detection DL algorithm in road damage detection and railway sleeper crack detection.

## Table of contents

- [Dataset](#Dataset)
- [Road Damage Detection](#Road-damage-detection)
  - [Understanding object detection algorithm](#Understanding-object-detection-algorithm)
  - [Data augmentation](#Data-augmentation)
      - [Perspective-awareness](#Perspective-awareness)
      - [Erasing damage](#Erasing-damage)
  - [Object Detection Model](#Object-Detection-Model)
      - [Faster R-CNN](#Faster-R-CNN)
      - [Other](#Other)
  - [Results](#Results)

- [Citation](#Citation)

#
## Dataset 
1. Road damage dataset came from [RDDC2020](https://github.com/sekilab/RoadDamageDetector#dataset-for-global-road-damage-detection-challenge-2020). Images are collected from the Czech Republic, India, and Japan (three images below from left to right). The dataset is open to the public. More detail about the dataset can be found in [the link](https://www.sciencedirect.com/science/article/pii/S2352340921004170).

<img src="images/Czech.png" height="250" />
<img src="images/India.png" height="250" />
<img src="images/Japan.png" height="250" />

Please note that the dataset didn't come with annotation for the test dataset. The participant have to . Thus, participant have to submit the file in the submission format to the organizer then the organizer will report the result on [their website.](https://crddc2022.sekilab.global/submissions/). The results in this thesis will be report based on evaluation datasets. The partition of the dataset can be seen in dataset_partition.

RDDC2020 have some problems with truncated data and annotations. The problems have been addressed and fixed in RDDC2022. The dataset is available [here](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547). The dataset is open to the public. More detail about the dataset can be found in [the link](https://arxiv.org/abs/2209.08538). The reason why this thesis started at RDDC2020 is because the size of the dataset is smaller and easier to work with. In the future, I will work with RDDC2022.

2. Sleeper crack dataset acquired from diagnostic locomotives came from the Swiss Federal Railways (SBB), and acquired from UAVs came from the Matterhorn Gotthard Railway (MGB). Both are private data.
#
## Road damage detection
### 1. Understanding object detection algorithm.
To have a better understanding of the object detection algorithm, I have implemented a simple object detection algorithm using Faster R-CNN techniques from [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) following [Pham et al. (2020)](https://ieeexplore.ieee.org/document/9378027). It is common to use pre-trained model as a feature extractor part of the network. Detectron2 provides a lot of pre-trained models. I have used Faster R-CNN with ResNet-101 backbone (R101-FPN) and ResNeXt-101 backbone (X101-FPN), both pre-trained on [COCO dataset](https://cocodataset.org/#home). The model is trained on the road damage dataset. 

### 2. Data augmentation
Data augmentation is a technique to increase the size of the dataset by applying some transformations to the original images. The purpose is to make the model more robust to the variation of the data. From Pham et al. (2020), horizontal flipping, resizing, and rotation are used. The images look like this: 

<img src="images/aug_problem1.png" height="250" />
<img src="images/aug_problem2.png" height="250" />
<img src="images/aug_problem4.png" height="250" />

The problem is that this augmentation technique is lack realistic, the placing is too random and the model is not robust to the variation of the data. Resulting in slightly worse performance than the original model.

 Therefore, I have tried to implement other data augmentation techniques to improve the performance of the original model.
 #### <ins>Perspective-awareness</ins> 
The technique is introduced by [Lis (2020a)](https://arxiv.org/abs/2210.01779). The idea is to consider the apparent size of the obstacles decreases as their distance to the vehicle increases in road obstacle detection applications. To inject synthetic damage into the image in a perspective-aware manner, the augmentation is applied to the image in the following steps: 
1. Use any pre-trained segmentation model to segment the road surface from the image. For this project, Panoptic FPN with ResNet-101 backbone pre-trained on the COCO dataset is used.

<img src="images/perfect1.png" height="250" />

2. Determine the perspective map by using the segmentation mask from the previous step. The perspective map is a 2D array with the same size as the image. Each element of the array is the distance from the pixel to the camera.

<img src="images/perfect2.png"  height="250" />

Now if we plot a ruler onto the original image, the ruler will look like the image below. Note that the interval between the ticks is 3.5 m.

<img src="images/perfect3.png"  height="250" />

3. After determining  the perspective maps for damages and background image, the damages are randomly placed on the background image. The damages will be placed into the background that has a similar perspective. The place of the damage is random but the size of the damage is according to the scale calculated from the perspective maps of damage and background image. Additionally, to make the injected damage looks more realistic [Poisson blending](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf) is used to blend the damage into the background image. The result is shown below.

<img src="images/perspective_aug1.png"  height="250" />
<img src="images/perspective_aug2.png"  height="250" />
<img src="images/perspective_aug3.png"  height="250" />


 #### <ins>Erasing damage</ins>
This technique is purposed by [Lis (2020b)](https://arxiv.org/abs/2012.13633) who used this technique to make to model recognize the drivable road path for autonomous  vehicles and [F. Kluger et al (2018)](https://ieeexplore.ieee.org/document/8622318) use the CycleGAN model to train on removed-damage images and added-generated-damage images. However, in this study, I will only focus on erasing damage, either erase it completely or partially. The erasing process can be used by any inpainting model, in this study I used [MAT: Mask-Aware Transformer for Large Hole Image Inpainting
](https://arxiv.org/pdf/2203.15270.pdf). Please clone this [MAT's Github](https://github.com/fenglinglwb/MAT.git).

<img src="images/Erase_before.png"  height="250" />
<img src="images/Erase_after.png"  height="250" />
<img src="images/Erase_completely.jpg"  height="250" />

#
###  3. Object Detection Model 
Up to this point, I only played around with [Faster R-CNN with ResNet+FPN](https://arxiv.org/pdf/1612.03144.pdf) backbone. However, there are other object detection architectures that reported outperform Faster R-CNN. For example [Cascade R-CNN](https://arxiv.org/pdf/1712.00726.pdf)

#
### 4. Results
| Model | Precision | Recall | F1 score | Converge iteration | Score threshold | 
| --- | --- | --- | --- | --- | --- | 
| R101-FPN | 0.55 | 0.51 | 0.53 | 115 000 | 0.57 |
| X101-FPN | 0.60 | 0.50 | 0.55 | 95 000 | 0.59 | 
| X101-FPN + Augmentation | 0.61 | 0.48 | 0.54 | 125 000 | 0.62 |
| X101-FPN + Road Segment | 0.61 | 0.46 | 0.53 | 120 000 | 0.56 |
| X101-FPN + Perspective-awareness | 0.51 | 0.49 | 0.50 | 130 000 | 0.53 |
| X101-FPN + Erasing damage | 0.59 | 0.49 | 0.54 | 120 000 | 0.59 |



 



## Citation
Pham, V., Pham, C., & Dang, T. (2020). Road Damage Detection and Classification with Detectron2 and Faster R-CNN. https://doi.org/10.1109/bigdata50022.2020.9378027

Lis, K. (2022, October 4). Perspective Aware Road Obstacle Detection. arXiv.org. https://arxiv.org/abs/2210.01779

Lis, K. (2020, December 25). Detecting Road Obstacles by Erasing Them. arXiv.org. https://arxiv.org/abs/2012.13633

F. Kluger et al., "Region-based Cycle-Consistent Data Augmentation for Object Detection," 2018 IEEE International Conference on Big Data (Big Data), Seattle, WA, USA, 2018, pp. 5205-5211, doi: 10.1109/BigData.2018.8622318.