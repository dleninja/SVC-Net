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
batch_size = 32
epochs = 100
#
path1 = Path("dataset/train/input")
path2 = Path("dataset/train/GT")
#
path1_valid = Path("dataset/valid/input")
path2_valid = Path("dataset/valid/gt")
#
export_dir = Path("Results")
if not os.path.exists(export_dir):
    os.makedirs(export_dir)
#
model_file_format = os.path.join(
    export_dir, 
    "construction_model.{epoch:03d}.hdf5"
)
#
checkpointer = ModelCheckpoint(
    model_file_format,
    period = 100,
    save_best_only=False,
    save_weights_only=True
)
#
im_shape = (800, 320) # (height, width)
#
data_gen_args = dict(
    zoom_range = 0.8,
    horizontal_flip = True,
    fill_mode = 'reflect'
    )
#
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
valid_datagen = ImageDataGenerator(rescale = 1.)
#
seed = 1
#
image_generator = image_datagen.flow_from_directory(
    directory = path1,
    class_mode = None,
    color_mode = "rgb",
    target_size = im_shape,
    seed = seed)
#
mask_generator = mask_datagen.flow_from_directory(
    directory = path2,
    class_mode = None,
    color_mode = "grayscale",
    target_size = im_shape,
    seed = seed)
#
# Combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)
#
val_image_generator = valid_datagen.flow_from_directory(
    directory = path1_valid,
    class_mode = None,
    color_mode = "rgb",
    target_size = im_shape,
    seed = seed)
#
val_mask_generator = valid_datagen.flow_from_directory(
    directory = path2_valid,
    class_mode = None,
    color_mode = "grayscale",
    target_size = im_shape,
    seed = seed)
#
# Combine generators into one which yields image and masks
valid_generator = zip(val_image_generator, val_mask_generator)
#
n_train = len(image_generator.filenames)
n_valid = len(val_image_generator.filenames)
#
# Load the model
model = svcnet_model(height=im_shape[0], width=im_shape[1], n_channels=3)
#
model.compile(
        optimizer = Adam(learning_rate=0.0001),
        loss = SSIMLoss,
        metrics = ["acc"]
    )
#
# Train the model
history = model.fit(
    x = train_generator,
    batch_size = batch_size,
    steps_per_epoch = n_train // batch_size,
    validation_data = valid_generator,
    validation_steps = n_valid // batch_size,
    callbacks = checkpointer,
    epochs = epochs)