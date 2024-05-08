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
Chronological overview of the datasets for object pose estimation evaluation. Notably, the pink arrows represent the BOP Challenge datasets, which can be used to evaluate both instance-level and unseen object methods. The red references represent the datasets of articulated objects.

<p align="center"> <img src="./resources/datasets.png" width="100%"> </p>

### Datasets for Instance-Level Methods
<details>
<summary>All Datasets</summary>
  
- BOP Challenge Datasets [[Paper]](https://arxiv.org/abs/2403.09799) [[Data]](https://bop.felk.cvut.cz/challenges/bop-challenge-2023/)
- YCBInEOAT Dataset [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9341314) [[Data]](https://github.com/wenbowen123/iros20-6d-pose-tracking)
</details>

### Datasets for Category-Level Methods
<details>
<summary>Rigid Objects Datasets</summary>
  
- CAMERA25 Dataset [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Normalized_Object_Coordinate_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2019_paper.pdf) [[Data]](https://github.com/hughw19/NOCS_CVPR2019)
- REAL275 Dataset [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Normalized_Object_Coordinate_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2019_paper.pdf) [[Data]](https://github.com/hughw19/NOCS_CVPR2019)
- kPAM Dataset [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-95459-8_9) [[Data]](https://sites.google.com/view/kpam)
- TOD Dataset [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_KeyPose_Multi-View_3D_Labeling_and_Keypoint_Estimation_for_Transparent_Objects_CVPR_2020_paper.pdf) [[Data]](https://sites.google.com/view/keypose)
- Objectron Dataset [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ahmadyan_Objectron_A_Large_Scale_Dataset_of_Object-Centric_Videos_in_the_CVPR_2021_paper.pdf) [[Data]](https://github.com/google-research-datasets/Objectron)
- Wild6D Dataset [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/afe99e55be23b3523818da1fefa33494-Abstract-Conference.html) [[Data]](https://oasisyang.github.io/semi-pose/)
- PhoCaL Dataset [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_PhoCaL_A_Multi-Modal_Dataset_for_Category-Level_Object_Pose_Estimation_With_CVPR_2022_paper.pdf) [[Data]](https://www.campar.in.tum.de/public_datasets/2022_cvpr_wang/)
- HouseCat6D Dataset [[Paper]](https://arxiv.org/pdf/2212.10428) [[Data]](https://sites.google.com/view/housecat6d)
</details>

<details>
<summary>Articulated Objects Datasets</summary>
  
- BMVC Dataset [[Paper]](https://www.researchgate.net/profile/Frank-Michel/publication/292033202_Pose_Estimation_of_Kinematic_Chain_Instances_via_Object_Coordinate_Regression/links/56caf40b08aee3cee5414a70/Pose-Estimation-of-Kinematic-Chain-Instances-via-Object-Coordinate-Regression.pdf) [[Data]](https://gaminghardware-guide.de/gaminghardware-guide-de-hat-cvlab-dresden-de-uebernommen/research/scene-understanding/pose-estimation/#BMVC15)
- RBO Dataset [[Paper]](https://journals.sagepub.com/doi/full/10.1177/0278364919844314) [[Data]](https://zenodo.org/records/1036660)
- HOI4D Dataset [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_HOI4D_A_4D_Egocentric_Dataset_for_Category-Level_Human-Object_Interaction_CVPR_2022_paper.pdf) [[Data]](http://www.hoi4d.top/)
- ReArtMix/ReArtVal Dataset [[Paper]](https://ieeexplore.ieee.org/abstract/document/9670684)
- ContactArt Dataset [[Paper]](https://arxiv.org/pdf/2305.01618) [[Data]](https://zehaozhu.github.io/ContactArt/)
</details>

### Datasets for Unseen Methods
<details>
<summary>All Datasets</summary>
  
- MOPED Dataset [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Park_LatentFusion_End-to-End_Differentiable_Reconstruction_and_Rendering_for_Unseen_Object_Pose_CVPR_2020_paper.pdf) [[Data]](https://latentfusion.github.io/)
- GenMOP Dataset [[Paper]](https://link.springer.com/content/pdf/10.1007/978-3-031-19824-3_18) [[Data]](https://liuyuan-pal.github.io/Gen6D/)
- OnePose Dataset [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_OnePose_One-Shot_Object_Pose_Estimation_Without_CAD_Models_CVPR_2022_paper.pdf) [[Data]](https://zju3dv.github.io/onepose/)
- OnePose-LowTexture Dataset [[Paper]](https://papers.nips.cc/paper_files/paper/2022/file/e43f900f571de6c96a70d5724a0fb565-Paper-Conference.pdf) [[Data]](https://zju3dv.github.io/onepose_plus_plus/)
</details>




## Instance-Level
### Correspondence-Based Methods
<details>
<summary>Sparse Correspondence Methods</summary>
  
#### 2019

#### 2020

</details>
<details>
<summary>Dense Correspondence Methods</summary>
  
#### 2019

#### 2020

</details>

### Template-Based Methods
<details>
<summary>RGB-Based Template Methods</summary>
  
#### 2019

#### 2020

</details>
<details>
<summary>Point Cloud-Based Template Methods</summary>
  
#### 2019

#### 2020

</details>

### Voting-Based Methods
<details>
<summary>Indirect Voting Methods</summary>
  
#### 2019

#### 2020

</details>
<details>
<summary>Direct Voting Methods</summary>
  
#### 2019

#### 2020

</details>

### Regression-Based Methods
<details>
<summary>Geometry-Guided Regression Methods</summary>
  
#### 2019

#### 2020

</details>
<details>
<summary>Direct Regression Methods</summary>
  
#### 2019

#### 2020

</details>



## Category-Level
### Shape Prior-Based Methods
<details>
<summary>NOCS Shape Alignment Methods</summary>
  
#### 2020
- Learning Canonical Shape Space for Category-Level 6D Object Pose and Size Estimation [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Learning_Canonical_Shape_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2020_paper.pdf)
</details>

<details>
<summary>Direct Regress Pose Methods</summary>

#### 2020
- Learning Canonical Shape Space for Category-Level 6D Object Pose and Size Estimation [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Learning_Canonical_Shape_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2020_paper.pdf)
</details>

### Shape Prior-Free Methods
<details>
<summary>Depth-Guided Geometry-Aware Methods</summary>

#### 2020
- Learning Canonical Shape Space for Category-Level 6D Object Pose and Size Estimation [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Learning_Canonical_Shape_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2020_paper.pdf)
</details>

<details>
<summary>RGBD-Guided Semantic and Geometry Fusion Methods</summary>
  
#### 2019
- Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Normalized_Object_Coordinate_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2019_paper.pdf) [[Code]](https://github.com/hughw19/NOCS_CVPR2019)
#### 2020
- Learning Canonical Shape Space for Category-Level 6D Object Pose and Size Estimation [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Learning_Canonical_Shape_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2020_paper.pdf)
</details>

<details>
<summary>Others</summary>

#### 2020
- Learning Canonical Shape Space for Category-Level 6D Object Pose and Size Estimation [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Learning_Canonical_Shape_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2020_paper.pdf)
</details>

## Unseen
### CAD Model-Based Methods
<details>
<summary>CAD Model-Based</summary>
  
#### 2019
- CorNet: Generic 3D Corners for 6D Pose Estimation of New Objects without Retraining [[Paper]](https://openaccess.thecvf.com/content_ICCVW_2019/papers/R6D/Pitteri_CorNet_Generic_3D_Corners_for_6D_Pose_Estimation_of_New_ICCVW_2019_paper.pdf)
#### 2020
- 3D Object Detection and Pose Estimation of Unseen Objects in Color Images with Local Surface Embeddings [[Paper]](https://openaccess.thecvf.com/content/ACCV2020/papers/Pitteri_3D_Object_Detection_and_Pose_Estimation_of_Unseen_Objects_in_ACCV_2020_paper.pdf)
- Multi-path Learning for Object Pose Estimation Across Domains [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sundermeyer_Multi-Path_Learning_for_Object_Pose_Estimation_Across_Domains_CVPR_2020_paper.pdf) [[Code]](https://github.com/DLR-RM/AugmentedAutoencoder/tree/multipath)
#### 2021
- ZePHyR: Zero-shot Pose Hypothesis Rating [[Paper]](https://arxiv.org/pdf/2104.13526) [[Code]](https://github.com/r-pad/zephyr)
#### 2022
- Unseen Object 6D Pose Estimation: A Benchmark and Baselines [[Paper]](https://arxiv.org/pdf/2206.11808) [[Code]](https://graspnet.net/unseen6d)
- I Like to Move It: 6D Pose Estimation as an Action Decision Process [[Paper]](https://arxiv.org/pdf/2009.12678)
- DISP6D: Disentangled Implicit Shape and Pose Learning for Scalable 6D Pose Estimation [[Paper]](https://arxiv.org/pdf/2107.12549) [[Code]](https://github.com/fylwen/DISP-6D)
- Templates for 3D Object Pose Estimation Revisited: Generalization to New objects and Robustness to Occlusions [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Nguyen_Templates_for_3D_Object_Pose_Estimation_Revisited_Generalization_to_New_CVPR_2022_paper.pdf) [[Code]](https://github.com/nv-nguyen/template-pose)
- Fusing Local Similarities for Retrieval-based 3D Orientation Estimation of Unseen Objects [[Paper]](https://arxiv.org/pdf/2203.08472) [[Code]](https://github.com/sailor-z/Unseen_Object_Pose)
- Self-supervised Vision Transformers for 3D pose estimation of novel objects [[Paper]](https://www.sciencedirect.com/science/article/pii/S0262885623001907)
- OVE6D: Object Viewpoint Encoding for Depth-based 6D Object Pose Estimation [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Cai_OVE6D_Object_Viewpoint_Encoding_for_Depth-Based_6D_Object_Pose_Estimation_CVPR_2022_paper.pdf) [[Code]](https://github.com/dingdingcai/OVE6D-pose)
- OSOP: A Multi-Stage One Shot Object Pose Estimation Framework [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Shugurov_OSOP_A_Multi-Stage_One_Shot_Object_Pose_Estimation_Framework_CVPR_2022_paper.pdf) 
- MegaPose: 6D Pose Estimation of Novel Objects via Render & Compare [[Paper]](https://arxiv.org/pdf/2212.06870) [[Code]](https://github.com/megapose6d/megapose6d)
#### 2023
- Learning Symmetry-Aware Geometry Correspondences for 6D Object Pose Estimation [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Learning_Symmetry-Aware_Geometry_Correspondences_for_6D_Object_Pose_Estimation_ICCV_2023_paper.pdf) [[Code]](https://github.com/hikvision-research/GCPose)
- KeyMatchNet: Zero-Shot Pose Estimation in 3D Point Clouds by Generalized Keypoint Matching [[Paper]](https://arxiv.org/pdf/2303.16102)
- ZeroPose: CAD-Model-based Zero-Shot Pose Estimation [[Paper]](https://arxiv.org/pdf/2305.17934)
- Diff-DOPE: Differentiable Deep Object Pose Estimation [[Paper]](https://arxiv.org/pdf/2310.00463) [[Code]](https://github.com/NVlabs/diff-dope)
- ZS6D: Zero-shot 6D Object Pose Estimation using Vision Transformers [[Paper]](https://arxiv.org/pdf/2309.11986)
- FoundPose: Unseen Object Pose Estimation with Foundation Features [[Paper]](https://arxiv.org/pdf/2311.18809) [[Code]](https://evinpinar.github.io/foundpose/)
#### 2024
- FreeZe: Training-free zero-shot 6D pose estimation with geometric and vision foundation models [[Paper]](https://arxiv.org/pdf/2312.00947) [[Code]](https://andreacaraffa.github.io/freeze/)
- MatchU: Matching Unseen Objects for 6D Pose Estimation from RGB-D Images [[Paper]](https://arxiv.org/pdf/2403.01517)
- SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation [[Paper]](https://arxiv.org/pdf/2311.15707) [[Code]](https://github.com/JiehongLin/SAM-6D)
- FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects [[Paper]](https://arxiv.org/pdf/2312.08344) [[Code]](https://github.com/NVlabs/FoundationPose)
- Object Pose Estimation via the Aggregation of Diffusion Features [[Paper]](https://arxiv.org/pdf/2403.18791) [[Code]](https://github.com/Tianfu18/diff-feats-pose)
- GigaPose: Fast and Robust Novel Object Pose Estimation via One Correspondence [[Paper]](https://arxiv.org/pdf/2311.14155) [[Code]](https://github.com/nv-nguyen/gigaPose)
- GenFlow: Generalizable Recurrent Flow for 6D Pose Refinement of Novel Objects [[Paper]](https://arxiv.org/pdf/2403.11510) 
</details>

<details>
<summary>Mannual Reference View-Based</summary>
  
#### 2020 
- LatentFusion: End-to-End Differentiable Reconstruction and Rendering for Unseen Object Pose Estimation[[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Park_LatentFusion_End-to-End_Differentiable_Reconstruction_and_Rendering_for_Unseen_Object_Pose_CVPR_2020_paper.pdf) [[Code]](https://github.com/NVlabs/latentfusion)
#### 2021
- Unseen Object Pose Estimation via Registration [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9517491) 
#### 2022
- PIZZA: A Powerful Image-only Zero-Shot Zero-CAD Approach to 6 DoF Tracking [[Paper]](https://arxiv.org/pdf/2209.07589) [[Code]](https://github.com/nv-nguyen/pizza)
- FS6D: Few-Shot 6D Pose Estimation of Novel Objects [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/He_FS6D_Few-Shot_6D_Pose_Estimation_of_Novel_Objects_CVPR_2022_paper.pdf) [[Code]](https://github.com/ethnhe/FS6D-PyTorch)
- Gen6D: Generalizable Model-Free 6-DoF Object Pose Estimation from RGB Images [[Paper]](https://link.springer.com/content/pdf/10.1007/978-3-031-19824-3_18) [[Code]](https://github.com/liuyuan-pal/Gen6D)
- OnePose: One-Shot Object Pose Estimation without CAD Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_OnePose_One-Shot_Object_Pose_Estimation_Without_CAD_Models_CVPR_2022_paper.pdf) [[Code]](https://github.com/zju3dv/OnePose)
- OnePose++: Keypoint-Free One-Shot Object Pose Estimation without CAD Models [[Paper]](https://papers.nips.cc/paper_files/paper/2022/file/e43f900f571de6c96a70d5724a0fb565-Paper-Conference.pdf) [[Code]](https://github.com/zju3dv/OnePose_Plus_Plus)
#### 2023
- POPE: 6-DoF Promptable Pose Estimation of Any Object, in Any Scene, with One Reference [[Paper]](https://arxiv.org/pdf/2305.15727) [[Code]](https://github.com/paulpanwang/POPE)
- SA6D: Self-Adaptive Few-Shot 6D Pose Estimator for Novel and Occluded Objects [[Paper]](https://arxiv.org/pdf/2308.16528) 
- PoseMatcher: One-shot 6D Object Pose Estimation by Deep Feature Matching [[Paper]](https://openaccess.thecvf.com/content/ICCV2023W/R6D/papers/Castro_PoseMatcher_One-Shot_6D_Object_Pose_Estimation_by_Deep_Feature_Matching_ICCVW_2023_paper.pdf) [[Code]](https://github.com/PedroCastro/PoseMatcher)
#### 2024
- NOPE: Novel Object Pose Estimation from a Single Image [[Paper]](https://arxiv.org/pdf/2303.13612) [[Code]](https://github.com/nv-nguyen/nope)
- LocPoseNet: Robust Location Prior for Unseen Object Pose Estimation [[Paper]](https://arxiv.org/pdf/2211.16290) [[Code]](https://github.com/sailor-z/LocPoseNet)
- Open-vocabulary object 6D pose estimation [[Paper]](https://arxiv.org/pdf/2312.00690) [[Code]](https://github.com/jcorsetti/oryon)
- Learning to Estimate 6DoF Pose from Limited Data: A Few-Shot, Generalizable Approach using RGB Images [[Paper]](https://arxiv.org/pdf/2306.07598) [[Code]](https://github.com/paulpanwang/Cas6D)
- MFOS: Model-Free & One-Shot Object Pose Estimation [[Paper]](https://arxiv.org/pdf/2310.01897) 
- GS-Pose: Cascaded Framework for Generalizable Segmentation-based 6D Object Pose Estimation [[Paper]](https://arxiv.org/pdf/2403.10683) [[Code]](https://github.com/dingdingcai/GSPose)
- FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects [[Paper]](https://arxiv.org/pdf/2312.08344) [[Code]](https://github.com/NVlabs/FoundationPose)
</details>

### Manual Reference View-Based Methods
<details>
<summary>Feature Matching-Based Methods</summary>
  
#### 2019

#### 2020

</details>
<details>
<summary>Template Matching-Based Methods</summary>
  
#### 2019

#### 2020

</details>


## Applications
### Robotic Manipulation
<details>
<summary>Instance-Level Manipulation</summary>


</details>
<details>
<summary>Category-Level Manipulation</summary>


</details>
<details>
<summary>Unseen Object Manipulation</summary>

- ZePHyR: Zero-shot Pose Hypothesis Rating [[Paper]](https://arxiv.org/pdf/2104.13526) [[Code]](https://github.com/r-pad/zephyr)
- MegaPose: 6D Pose Estimation of Novel Objects via Render & Compare [[Paper]](https://arxiv.org/pdf/2212.06870) [[Code]](https://github.com/megapose6d/megapose6d)
- FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects [[Paper]](https://arxiv.org/pdf/2312.08344) [[Code]](https://github.com/NVlabs/FoundationPose)
</details>

### Augmented Reality/Virtual Reality
<details>
<summary>All Methods</summary>

- Gen6D: Generalizable Model-Free 6-DoF Object Pose Estimation from RGB Images [[Paper]](https://link.springer.com/content/pdf/10.1007/978-3-031-19824-3_18) [[Code]](https://github.com/liuyuan-pal/Gen6D)
- OnePose++: Keypoint-Free One-Shot Object Pose Estimation without CAD Models [[Paper]](https://papers.nips.cc/paper_files/paper/2022/file/e43f900f571de6c96a70d5724a0fb565-Paper-Conference.pdf) [[Code]](https://github.com/zju3dv/OnePose_Plus_Plus)
- FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects [[Paper]](https://arxiv.org/pdf/2312.08344) [[Code]](https://github.com/NVlabs/FoundationPose)
</details>

### Aerospace
<details>
<summary>All Methods</summary>

- 
</details>

### Hand-Object Interaction
<details>
<summary>All Methods</summary>

- 
</details>

### Autonomous Driving
<details>
<summary>All Methods</summary>

- 
</details>
