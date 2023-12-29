# SVC-Net
## Overview
Optical coherence tomography angiography (OCTA) provides unrivaled capability for depth-resolved visualization of retinal vasculature at the microcapillary level resolution. For OCTA image construction, repeated OCT scans from one location are required to identify blood vessels with active blood flow. The requirement for multi-scan-volumetric OCT can reduce OCTA imaging speed, which will induce eye movements and limit the image field-of-view. In principle, the blood flow should also affect the reflectance brightness profile along the vessel direction in a single-scan-volumetric OCT. Here we report a spatial vascular connectivity network (SVC-Net) for deep learning OCTA construction from single-scan-volumetric OCT. We quantitatively determine the optimal number of neighboring B-scans as image input, we compare the effects of neighboring B-scans to single B-scan input models, and we explore different loss functions for optimization of SVC-Net. This approach can improve the clinical implementation of OCTA by improving transverse image resolution or increasing the field-of-view.

## Model Architecture

Our model, SVC-Net, follows the design of an encoder-decoder network. For the encoder, the EfficientNetB0 neural network was employed. Whereas the decoder was custom designed using the Keras library. Briefly, we used a convolutional neural network to predict vessels in an image regression manner. The input into the CNN was a multichannel input comprised of OCT B-scans and the output was a grayscale image. 

## Dependencies

- tensorflow >= 1.31.1
- keras >= 2.2.4
- python >= 3.7.1

## Citations
Le, David, Taeyoon Son, Tae-Hoon Kim, Tobiloba Adejumo, Mansour Abtahi, Shaiban Ahmed, Alfa Rossi, Behrouz Ebrahimi, Albert Dadzie, and Xincheng Yao. "SVC-Net: A spatially vascular connectivity network for deep learning construction of microcapillary angiography from single-scan-volumetric OCT." Communications Engineering (2023). https://dx.doi.org/10.21203/rs.3.rs-2387074/v1
