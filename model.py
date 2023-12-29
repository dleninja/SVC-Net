"""
Spatial Variance Connectivity Network (SVC-Net)
This file contains the code to train SVC-Net. For this example, a 3N model will be demonstrated.
Minor modifications is needed for the other types of input models.
@author: dleninja
"""
#
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate
from tensorflow.keras import Model
from tensorflow.keras import backend
import sys

def svcnet_model(height=None, width=None, n_channels=3):
  """
  SVC-Net model is an encoder-decoder network.
  - The encoder utilizes the EfficientNetB0 architecture.
  - The decoder is a custom decoder.

  Args:
    height: height of image.
    width: width of image.
    n_channels: number of adjacent scans.

  Returns:
    Model of network.
  """
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
  #
  input_layer = Input(shape=(height, width, n_channels))
  encoder_model = EfficientNetB0(
      include_top=False,
      weights=None,
      input_tensor=input_layer,
      input_shape=(height, width, n_channels)
  )
  #
  skip1 = encoder_model.get_layer('block6a_expand_activation').output
  skip2 = encoder_model.get_layer('block4a_expand_activation').output
  skip3 = encoder_model.get_layer('block3a_expand_activation').output
  skip4 = encoder_model.get_layer('block2a_expand_activation').output
  encoder_output = encoder_model.get_layer('top_activation').output
  #
  up1 = UpSampling2D(size=(2,2))(encoder_output)
  cat1 = Concatenate(axis=bn_axis, name="concat1")([up1, skip1])
  decoder1 = decoder_block(cat1, 256, 'decoder_block1')
  #
  up2 = UpSampling2D(size=(2,2))(decoder1)
  cat2 = Concatenate(axis=bn_axis, name="concat2")([up2, skip2])
  decoder2 = decoder_block(cat2, 128, 'decoder_block2')
  #
  up3 = UpSampling2D(size=(2,2))(decoder2)
  cat3 = Concatenate(axis=bn_axis, name="concat3")([up3, skip3])
  decoder3 = decoder_block(cat3, 64, 'decoder_block3')
  #
  up4 = UpSampling2D(size=(2,2))(decoder3)
  cat4 = Concatenate(axis=bn_axis, name="concat4")([up4, skip4])
  decoder4 = decoder_block(cat4, 32, 'decoder_block4')
  #
  up5 = UpSampling2D(size=(2,2))(decoder4)
  decoder5 = decoder_block(up5, 16, 'decoder_block5')
  #
  final_conv = Conv2D(filters=1, kernel_size=3, use_bias=True, padding="same", name='final_conv')(decoder5)
  final_act = Activation('linear')(final_conv)
  #
  svcnet = Model(input_layer, final_act)
  #
  return svcnet
#
def decoder_block(x, num_filter, name):
  """
  Decoder Block

  Args:
    x: input tensor.
    num_filter: number of filters for 2D Convolution.
    name: string, block name.

  Returns:
    x: Output tensor for the block.
  """
  #
  x = Conv2D(filters=num_filter, kernel_size=3, use_bias=False, padding="same", name=name + '_0_conv')(x)
  x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn")(x)
  x = Activation("relu", name=name + '_0_relu')(x)
  #
  x = Conv2D(filters=num_filter, kernel_size=3,  use_bias=False, padding="same", name=name + '_1_conv')(x)
  x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(x)
  x = Activation("relu", name=name + '_1_relu')(x)
  #
  return x