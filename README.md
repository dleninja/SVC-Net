# SVC-Net
Deep learning for Spatial Vascular Connectivity Network (SVC-Net).

## Overview
Optical coherence tomography angiography (OCTA) provides unrivaled capability for depth-resolved visualization of retinal vasculature at the microcapillary level resolution. For OCTA image construction, repeated OCT scans from one location are required to identify blood vessels with active blood flow. The requirement for multi-scan-volumetric OCT can reduce OCTA imaging speed, which will induce eye movements and limit the image field-of-view. In principle, the blood flow should also affect the reflectance brightness profile along the vessel direction in a single-scan-volumetric OCT.

![Spatial vascular connectivity (SVC) between adjacent OCT scans contains both spatial and temporal information that can be used as inference for deep learning OCTA prediction.](/misc/project_overview.png)
*Fig. 1. a) An illustration of the deep learning pipeline for OCTA construction using SVC network (SVC-Net). The input is derived from a singular OCT volume and is comprised of three neighboring OCT B-scans. The output is a single OCTA B-scan. In the training process, the ground truth is derived from conventional OCTA construction, i.e., speckle variance. b) An illustration of the vascular spatial connectivity information found in the OCT B-scans. The retinal vessels are connected via the neighboring OCT B-scans. c) A representation of the neighboring B-scans in a single OCT volume.*

Here we report a spatial vascular connectivity network (SVC-Net) for deep learning OCTA construction from single-scan-volumetric OCT. We quantitatively determine the optimal number of neighboring B-scans as image input, we compare the effects of neighboring B-scans to single B-scan input models, and we explore different loss functions for optimization of SVC-Net. This approach can improve the clinical implementation of OCTA by improving transverse image resolution or increasing the field-of-view.

## Model Architecture

Our model, SVC-Net, follows the design of an encoder-decoder network. For the encoder, the EfficientNetB0 neural network was employed. Whereas the decoder was custom designed using the Keras library. Briefly, we used a convolutional neural network to predict vessels in an image regression manner. The input into the CNN was a multichannel input comprised of OCT B-scans and the output was a grayscale image.

![SVC-Net is an encoder-decoder network. ](/misc/network_architecture.png)
*Fig. 2. a) The network architecture of SVC-Net, with representative input images. The input is a three-channel image of size 𝑚×𝑛×3, and the output is a single-channel image of size 𝑚×𝑛×1. The network follows a U-shaped architecture with cross connections from encoder to decoder network represented by dashed arrows. For each layer, the information between each sub-block and operation flows from top to bottom. b) The components of the network. The network is composed of modules, M1 to M6. Each module contains a set of operations. The modules further make up the block components, which are represented as squares in the individual layers of a).*

## Dependencies

- tensorflow >= 1.31.1
- keras >= 2.2.4
- python >= 3.7.1

For more information related to the dataset and codes, please feel free to contact the corresponding author via email (Prof. Xincheng Yao, xcy@uic.edu), and tell us about your study. We can provide you with the necessary information and resources.

## Citations
Le, David, Taeyoon Son, Tae-Hoon Kim, Tobiloba Adejumo, Mansour Abtahi, Shaiban Ahmed, Alfa Rossi, Behrouz Ebrahimi, Albert Dadzie, and Xincheng Yao. "SVC-Net: A spatially vascular connectivity network for deep learning construction of microcapillary angiography from single-scan-volumetric OCT." Communications Engineering (2023). https://dx.doi.org/10.21203/rs.3.rs-2387074/v1

[![DOI](https://zenodo.org/badge/736834011.svg)](https://zenodo.org/doi/10.5281/zenodo.10443044)
