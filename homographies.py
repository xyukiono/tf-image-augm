#
# Homographies
# Some codes come from  https://github.com/rpautrat/SuperPoint
# input image is supposed to be 3D tensor [H,W,C] and floating 0~255 values

import tensorflow as tf
from tensorflow.contrib.image import transform as H_transform
from math import pi
import cv2 as cv

def sample_homography(
        shape, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
        perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=pi/2,
        allow_artifacts=False, translation_overflow=0.):
    """Sample a random valid homography.

    Computes the homography transformation between a random patch in the original image
    and a warped projection with the same image size.
    As in `tf.contrib.image.transform`, it maps the output point (warped patch) to a
    transformed input point (original patch).
    The original patch, which is initialized with a simple half-size centered crop, is
    iteratively projected, scaled, rotated and translated.

    Arguments:
        shape: A rank-2 `Tensor` specifying the height and width of the original image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography.
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.

    Returns:
        A `Tensor` of shape `[1, 8]` corresponding to the flattened homography transform.
    """

    # Corners of the output image
    pts1 = tf.stack([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], axis=0)
    # Corners of the input patch
    margin = (1 - patch_ratio) / 2
    pts2 = margin + tf.constant([[0, 0], [0, patch_ratio],
                                 [patch_ratio, patch_ratio], [patch_ratio, 0]],
                                tf.float32)

    # Random perspective and affine perturbations
    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        perspective_displacement = tf.truncated_normal([1], 0., perspective_amplitude_y/2)
        h_displacement_left = tf.truncated_normal([1], 0., perspective_amplitude_x/2)
        h_displacement_right = tf.truncated_normal([1], 0., perspective_amplitude_x/2)
        pts2 += tf.stack([tf.concat([h_displacement_left, perspective_displacement], 0),
                          tf.concat([h_displacement_left, -perspective_displacement], 0),
                          tf.concat([h_displacement_right, perspective_displacement], 0),
                          tf.concat([h_displacement_right, -perspective_displacement],
                                    0)])

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = tf.concat(
                [[1.], tf.truncated_normal([n_scales], 1, scaling_amplitude/2)], 0)
        center = tf.reduce_mean(pts2, axis=0, keepdims=True)
        scaled = tf.expand_dims(pts2 - center, axis=0) * tf.expand_dims(
                tf.expand_dims(scales, 1), 1) + center
        if allow_artifacts:
            valid = tf.range(n_scales)  # all scales are valid except scale=1
        else:
            valid = tf.where(tf.reduce_all((scaled >= 0.) & (scaled < 1.), [1, 2]))[:, 0]
        idx = valid[tf.random_uniform((), maxval=tf.shape(valid)[0], dtype=tf.int32)]
        pts2 = scaled[idx]

    # Random translation
    if translation:
        t_min, t_max = tf.reduce_min(pts2, axis=0), tf.reduce_min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += tf.expand_dims(tf.stack([tf.random_uniform((), -t_min[0], t_max[0]),
                                         tf.random_uniform((), -t_min[1], t_max[1])]),
                               axis=0)

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = tf.lin_space(tf.constant(-max_angle), tf.constant(max_angle), n_angles)
        angles = tf.concat([angles, [0.]], axis=0)  # in case no rotation is valid
        center = tf.reduce_mean(pts2, axis=0, keepdims=True)
        rot_mat = tf.reshape(tf.stack([tf.cos(angles), -tf.sin(angles), tf.sin(angles),
                                       tf.cos(angles)], axis=1), [-1, 2, 2])
        rotated = tf.matmul(
                tf.tile(tf.expand_dims(pts2 - center, axis=0), [n_angles+1, 1, 1]),
                rot_mat) + center
        if allow_artifacts:
            valid = tf.range(n_angles)  # all angles are valid, except angle=0
        else:
            valid = tf.where(tf.reduce_all((rotated >= 0.) & (rotated < 1.),
                                           axis=[1, 2]))[:, 0]
        idx = valid[tf.random_uniform((), maxval=tf.shape(valid)[0], dtype=tf.int32)]
        pts2 = rotated[idx]

    # Rescale to actual size
    shape = tf.to_float(shape[::-1])  # different convention [y, x]
    pts1 *= tf.expand_dims(shape, axis=0)
    pts2 *= tf.expand_dims(shape, axis=0)

    # getPerspectiveTransform
    # See opencv/modules/imgproc/src/imgwarp.cpp
    # p[2]=(src.x,src.y), q[2]=(dst.x,dst.y)
    # ax() computes a[i][0...7], ay() computes a[i+4][0...7]
    def ax(p, q): return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q): return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = tf.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = tf.transpose(tf.stack(
        [[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))
    homography = tf.transpose(tf.matrix_solve_ls(a_mat, p_mat, fast=True))
    return homography

def invert_homography(H):
    """
    Computes the inverse transformation for a flattened homography transformation.
    H.shape = [B,8]
    """
    return mat2flat(tf.matrix_inverse(flat2mat(H)))


def flat2mat(H):
    """
    Converts a flattened homography transformation with shape `[1, 8]` to its
    corresponding homography matrix with shape `[1, 3, 3]`.
    """
    return tf.reshape(tf.concat([H, tf.ones([tf.shape(H)[0], 1])], axis=1), [-1, 3, 3])


def mat2flat(H):
    """
    Converts an homography matrix with shape `[1, 3, 3]` to its corresponding flattened
    homography transformation with shape `[1, 8]`.
    """
    H = tf.reshape(H, [-1, 9])
    return (H / H[:, 8:9])[:, :8]

def compute_valid_mask(image_shape, homography, erosion_radius=0):
    """
    Compute a boolean mask of the valid pixels resulting from an homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.

    Arguments:
        input_shape: Tensor of rank 2 representing the image shape, i.e. `[H, W]`.
        homography: Tensor of shape (B, 8) or (8,), where B is the batch size.
        erosion_radius: radius of the margin to be discarded.

    Returns: a Tensor of type `tf.int32` and shape (H, W).
    """
    mask = H_transform(tf.ones(image_shape), homography, interpolation='NEAREST')
    if erosion_radius > 0:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erosion_radius*2,)*2)
        mask = tf.nn.erosion2d(
                mask[tf.newaxis, ..., tf.newaxis],
                tf.to_float(tf.constant(kernel)[..., tf.newaxis]),
                [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')[0, ..., 0] + 1.
    return tf.to_int32(mask)