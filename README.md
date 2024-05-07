<div align="center">

# Deep Learning-Based Object Pose Estimation: A Comprehensive Survey

[Jian Liu](https://cnjliu.github.io/), Wei Sun, Hui Yang, Zhiwen Zeng, Chongpei Liu,

Jin Zeng, [Xingyu Liu](https://lliu-xingyu.github.io/), [Hossein Rahmani](https://sites.google.com/view/rahmaniatlu), [Nicu Sebe](https://scholar.google.com.hk/citations?user=stFCYOAAAAAJ&hl=zh-CN&oi=ao), [Ajmal Mian](https://ajmalsaeed.net/)

### [Introduction](#introduction) | [Datasets](#datasets) | [Instance-Level](#instance-level) | [Category-Level](#category-level) | [Unseen](#unseen) | [Applications](#applications)

</div>

## Introduction

This is the official repository of [''Deep Learning-Based Object Pose Estimation: A Comprehensive Survey''](https://arxiv.org/pdf/2202.02980.pdf). Specifically, we first introduce the [datasets](#datasets) used for object pose estimation. Then, we review the [instance-level](#instance-level), [category-level](#category-level), and [unseen](#unseen) methods, respectively. Finally, we summarize the common [applications](#applications) of this task. The taxonomy of this survey is shown as follows
<p align="center"> <img src="./resources/taxonomy.png" width="100%"> </p>

## Datasets
Chronological overview of the datasets for object pose estimation evaluation. Notably, the pink arrows represent the BOP datasets, which can be used to evaluate both instance-level and unseen object methods. The red references represent the datasets of articulated objects.

<p align="center"> <img src="./resources/datasets.png" width="100%"> </p>

### Datasets for Instance-Level Methods

- BOP Challenge Datasets [[Paper]](https://arxiv.org/abs/2403.09799) [[Data]](https://bop.felk.cvut.cz/challenges/bop-challenge-2023/)

### Datasets for Category-Level Methods

### Datasets for Unseen Methods



## Instance-Level



## Category-Level
### 2019
Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Normalized_Object_Coordinate_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2019_paper.pdf) [[Code]](https://github.com/hughw19/NOCS_CVPR2019)
### 2020
Learning Canonical Shape Space for Category-Level 6D Object Pose and Size Estimation [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Learning_Canonical_Shape_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2020_paper.pdf)


## Unseen
<details>
<summary>CAD Model-Based</summary>

#### 2019
CorNet: Generic 3D Corners for 6D Pose Estimation of New Objects without Retraining [[Paper]](https://openaccess.thecvf.com/content_ICCVW_2019/papers/R6D/Pitteri_CorNet_Generic_3D_Corners_for_6D_Pose_Estimation_of_New_ICCVW_2019_paper.pdf)
#### 2020
3D Object Detection and Pose Estimation of Unseen Objects in Color Images with Local Surface Embeddings [[Paper]](https://openaccess.thecvf.com/content/ACCV2020/papers/Pitteri_3D_Object_Detection_and_Pose_Estimation_of_Unseen_Objects_in_ACCV_2020_paper.pdf)

Multi-path Learning for Object Pose Estimation Across Domains [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sundermeyer_Multi-Path_Learning_for_Object_Pose_Estimation_Across_Domains_CVPR_2020_paper.pdf) [[Code]](https://github.com/DLR-RM/AugmentedAutoencoder/tree/multipath)

#### 2021
ZePHyR: Zero-shot Pose Hypothesis Rating [[Paper]](https://arxiv.org/pdf/2104.13526) [[Code]](https://github.com/r-pad/zephyr)
#### 2022
Unseen Object 6D Pose Estimation: A Benchmark and Baselines [[Paper]](https://arxiv.org/pdf/2206.11808) [[Code]](https://graspnet.net/unseen6d)
I Like to Move It: 6D Pose Estimation as an Action Decision Process [[Paper]](https://arxiv.org/pdf/2009.12678)

DISP6D: Disentangled Implicit Shape and Pose Learning for Scalable 6D Pose Estimation [[Paper]](https://arxiv.org/pdf/2107.12549) [[Code]](https://github.com/fylwen/DISP-6D)

Templates for 3D Object Pose Estimation Revisited: Generalization to New objects and Robustness to Occlusions[[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Nguyen_Templates_for_3D_Object_Pose_Estimation_Revisited_Generalization_to_New_CVPR_2022_paper.pdf) [[Code]](https://github.com/nv-nguyen/template-pose)

Fusing Local Similarities for Retrieval-based 3D Orientation Estimation of Unseen Objects [[Paper]](https://arxiv.org/pdf/2203.08472) [[Code]](https://github.com/sailor-z/Unseen_Object_Pose)

Self-supervised Vision Transformers for 3D pose estimation of novel objects [[Paper]](https://www.sciencedirect.com/science/article/pii/S0262885623001907)

OVE6D: Object Viewpoint Encoding for Depth-based 6D Object Pose Estimation [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Cai_OVE6D_Object_Viewpoint_Encoding_for_Depth-Based_6D_Object_Pose_Estimation_CVPR_2022_paper.pdf) [[Code]](https://github.com/dingdingcai/OVE6D-pose)

OSOP: A Multi-Stage One Shot Object Pose Estimation Framework [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Shugurov_OSOP_A_Multi-Stage_One_Shot_Object_Pose_Estimation_Framework_CVPR_2022_paper.pdf) 

MegaPose: 6D Pose Estimation of Novel Objects via Render & Compare [[Paper]](https://arxiv.org/pdf/2212.06870) [[Code]](https://github.com/megapose6d/megapose6d)
#### 2023
Learning Symmetry-Aware Geometry Correspondences for 6D Object Pose Estimation [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Learning_Symmetry-Aware_Geometry_Correspondences_for_6D_Object_Pose_Estimation_ICCV_2023_paper.pdf) [[Code]](https://github.com/hikvision-research/GCPose)

KeyMatchNet: Zero-Shot Pose Estimation in 3D Point Clouds by Generalized Keypoint Matching [[Paper]](https://arxiv.org/pdf/2303.16102)

ZeroPose: CAD-Model-based Zero-Shot Pose Estimation [[Paper]](https://arxiv.org/pdf/2305.17934)

ZS6D: Zero-shot 6D Object Pose Estimation using Vision Transformers [[Paper]](https://arxiv.org/pdf/2309.11986)

FoundPose: Unseen Object Pose Estimation with Foundation Features [[Paper]](https://arxiv.org/pdf/2311.18809) [[Code]](https://evinpinar.github.io/foundpose/)

#### 2024
FreeZe: Training-free zero-shot 6D pose estimation with geometric and vision foundation models [[Paper]](https://arxiv.org/pdf/2312.00947) [[Code]](https://andreacaraffa.github.io/freeze/)

MatchU: Matching Unseen Objects for 6D Pose Estimation from RGB-D Images [[Paper]](https://arxiv.org/pdf/2403.01517)

SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation [[Paper]](https://arxiv.org/pdf/2311.15707) [[Code]](https://github.com/JiehongLin/SAM-6D)

FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects [[Paper]](https://arxiv.org/pdf/2312.08344) [[Code]](https://github.com/NVlabs/FoundationPose)

Object Pose Estimation via the Aggregation of Diffusion Features [[Paper]](https://arxiv.org/pdf/2403.18791) [[Code]](https://github.com/Tianfu18/diff-feats-pose)

GigaPose: Fast and Robust Novel Object Pose Estimation via One Correspondence [[Paper]](https://arxiv.org/pdf/2311.14155) [[Code]](https://github.com/nv-nguyen/gigaPose)

</details>

## Applications
### Robotic Manipulation
#### Instance-Level Manipulation
#### Category-Level Manipulation
#### Unseen Object Manipulation

### Augmented Reality/Virtual Reality

### Aerospace

### Hand-Object Interaction

### Autonomous Driving
