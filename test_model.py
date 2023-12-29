"""
Spatial Variance Connectivity Network (SVC-Net)
This file contains the code to train SVC-Net. For this example, a 3N model will be demonstrated.
Minor modifications is needed for the other types of input models.
@author: dleninja
"""
#
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from model import *
from custom_utils import *
#
model = model = svcnet_model(height=im_shape[0], width=im_shape[1], n_channels=3)
#
model_name = "construction_model.100.hdf5"
#
export_dir = Path("Results")
#
main_export_name = "Results/Prediction"
export_pred_dir = Path(main_export_name)
if not os.path.exists(export_pred_dir):
    os.makedirs(export_pred_dir)
#
load_weights_name = os.path.join(export_dir, model_name)
print(load_weights_name)
model.load_weights(load_weights_name)
#
path1 = Path("dataset/test/input")
#
im_shape = (800, 320) # (height, width)
#
test_datagen = ImageDataGenerator(rescale = 1.)
#
test_generator = test_datagen.flow_from_directory(
    directory = path1,
    class_mode = None,
    color_mode = "rgb",
    shuffle = False,
    target_size = im_shape,
    batch_size = 1)
#
file_names = test_generator.filenames
n_test = len(file_names)
#
X_pred = model.predict_generator(test_generator,n_test)
print(X_pred.shape)

# Save each b-scan image into corresponding eye folder
for i in range(0, n_test):
    image_pred = X_pred[i,:,:,0]*255
    #
    temp_filename = file_names[i]
    im_name = temp_filename.split("/")
    #
    temp_path_name = main_export_name + "/" + im_name[0]
    export_pred_dir = Path(temp_path_name)
    #
    if not os.path.exists(export_pred_dir):
        os.makedirs(export_pred_dir)
    #
    save_name = os.path.join(export_pred_dir,"pred_" + im_name[1])
    cv2.imwrite(save_name,image_pred)
