"""
Spatial Variance Connectivity Network (SVC-Net)
@author: dleninja
"""
#
import tensorflow as tf
#
def SSIMLoss(y_true, y_pred):
    """
    Structural similiarity index measure (SSIM) loss function

    Args:
        y_true: ground truth tensor
        y_pred: prediction tensor by the model

    Returns:
        loss
    """
    return (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0)))