import cv2 as cv
import numpy as np
import tensorflow as tf
import homographies as hmg
import improcess as imp

def homographic_augmentation(image, valid_border_margin=3, **hmg_params):
    # image is 3D tensor

    with tf.name_scope('homographic_augmentation'):
        image_shape = tf.shape(image)[:2]
        homography = hmg.sample_homography(image_shape, **hmg_params)[0]
        warped_image = tf.contrib.image.transform(
                image, homography, interpolation='BILINEAR')
        valid_mask = hmg.compute_valid_mask(image_shape, homography,
                                        erosion_radius=valid_border_margin)

    ret = {'image': warped_image, 
           'valid_mask': valid_mask,
           'homography': homography,
          }
    return ret

def elastic_deformation(image, alpha, sigma, alpha_affine):

    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    # image is 3D tensor and sigma needs to be scalar (not tf.Tensor)
    # e.g.
    #     alpha = image.shape[1] * 2
    #     sigma = image.shape[1] * 0.08
    #     alpha_affine = image.shape[1] * 0.08

    shape = tf.shape(image)
    shape_size = shape[:2]
    src_height, src_width = tf.unstack(shape_size)
    
    center_square = tf.to_float(shape_size)// 2
    square_size = tf.to_float(tf.reduce_min(shape_size)) // 3

    pts1 = tf.to_float([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + tf.random_uniform(tf.shape(pts1), minval=-alpha_affine, maxval=alpha_affine, dtype=tf.float32)

    def ax(p, q): return [p[0], p[1], 1, 0, 0, 0]
    def ay(p, q): return [0, 0, 0, p[0], p[1], 1]

    a_mat = tf.stack([f(pts1[i], pts2[i]) for i in range(3) for f in (ax, ay)], axis=0)
    p_mat = tf.transpose(tf.stack(
                [[pts2[i][j] for i in range(3) for j in range(2)]], axis=0))  

    M = tf.transpose(tf.matrix_solve_ls(a_mat, p_mat, fast=True)) # [1,6]
    H = tf.concat([M, tf.zeros([tf.shape(M)[0], 2])], axis=1) # affine to homography
    H_inv = hmg.invert_homography(H) # [1,8]

    # add padding to eliminate out-of-frame regions
    pad_size = tf.to_int32(tf.to_float(tf.maximum(src_height, src_width)) *  (np.sqrt(2)-1.0) / 2 + 0.5)
    image_pad = tf.pad(image, [[pad_size]*2, [pad_size]*2, [0,0]], 'REFLECT')
    warped_image_pad = tf.contrib.image.transform(
                        image_pad, H_inv[0], interpolation='BILINEAR')

    tmp_height, tmp_width = tf.unstack(tf.shape(warped_image_pad))[:2]

    truncate = 4 # default param of scipy.ndimage.gaussian_filter
    ksize = 2 * int(truncate * sigma + 0.5) + 1

    # Super-slow due to large keernel size (about 100~200)
    dx = imp.gauss_blur(tf.random_uniform([tmp_height, tmp_width, 1]) * 2 - 1, ksize=ksize, sigma=sigma) * alpha
    dy = imp.gauss_blur(tf.random_uniform([tmp_height, tmp_width, 1]) * 2 - 1, ksize=ksize, sigma=sigma) * alpha

    x_t, y_t = tf.meshgrid(tf.range(0, tmp_width), tf.range(0, tmp_height))
    x_t = tf.to_float(x_t[...,None])
    y_t = tf.to_float(y_t[...,None])

    coords = tf.concat([x_t+dx, y_t+dy], axis=-1)
    deform_image_pad = imp.bilinear_sampling(warped_image_pad[None], coords[None])[0]
    deform_image = tf.slice(deform_image_pad, [pad_size, pad_size, 0], [src_height, src_width, -1])

    return deform_image

def random_distortion(images, num_anchors=10, perturb_sigma=5.0, disable_border=True):
    # Similar results to elastic deformation (a bit complex transformation)
    # However, the transformation is much faster that elastic deformation and have a straightforward arguments
    # TODO: Need to adapt reflect padding and eliminate out-of-frame
    # images is 4D tensor [B,H,W,C]
    # num_anchors : the number of base position to make distortion, total anchors in a image = num_anchors**2
    # perturb_sigma : the displacement sigma of each anchor

    src_shp_list = images.get_shape().as_list()
    batch_size, src_height, src_width = tf.unstack(tf.shape(images))[:3]

    if disable_border:
        pad_size = tf.to_int32(tf.to_float(tf.maximum(src_height, src_width)) *  (np.sqrt(2)-1.0) / 2 + 0.5)
        images = tf.pad(images, [[0,0], [pad_size]*2, [pad_size]*2, [0,0]], 'REFLECT')
    height, width = tf.unstack(tf.shape(images))[1:3]

    mapx_base = tf.matmul(tf.ones(shape=tf.stack([num_anchors, 1])),
                    tf.transpose(tf.expand_dims(tf.linspace(0., tf.to_float(width), num_anchors), 1), [1, 0]))
    mapy_base = tf.matmul(tf.expand_dims(tf.linspace(0., tf.to_float(height), num_anchors), 1),
                    tf.ones(shape=tf.stack([1, num_anchors])))

    mapx_base = tf.tile(mapx_base[None,...,None], [batch_size,1,1,1]) # [batch_size, N, N, 1]
    mapy_base = tf.tile(mapy_base[None,...,None], [batch_size,1,1,1])
    distortion_x = tf.random_normal((batch_size,num_anchors,num_anchors,1), stddev=perturb_sigma)
    distortion_y = tf.random_normal((batch_size,num_anchors,num_anchors,1), stddev=perturb_sigma)
    mapx = mapx_base + distortion_x
    mapy = mapy_base + distortion_y
    mapx_inv = mapx_base - distortion_x
    mapy_inv = mapy_base - distortion_y

    interp_mapx_base = tf.image.resize_images(mapx_base, size=(height, width), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    interp_mapy_base = tf.image.resize_images(mapy_base, size=(height, width), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    coord_maps_base = tf.concat([interp_mapx_base, interp_mapy_base], axis=-1)

    interp_mapx = tf.image.resize_images(mapx, size=(height, width), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    interp_mapy = tf.image.resize_images(mapy, size=(height, width), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    coord_maps = tf.concat([interp_mapx, interp_mapy], axis=-1) # [batch_size, height, width, 2]

    # interp_mapx_inv = tf.image.resize_images(mapx_inv, size=(height, width), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    # interp_mapy_inv = tf.image.resize_images(mapy_inv, size=(height, width), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    # coord_maps_inv = tf.concat([interp_mapx_inv, interp_mapy_inv], axis=-1) # [batch_size, height, width, 2]
    coord_maps_inv = coord_maps_base + (coord_maps_base-coord_maps)

    warp_images = imp.bilinear_sampling(images, coord_maps)

    if disable_border:
        warp_images = tf.slice(warp_images, [0, pad_size, pad_size, 0], [-1, src_height, src_width, -1])

    warp_images.set_shape(src_shp_list)
    # shp_list[-1] = 2
    # coord_maps.set_shape(shp_list)
    # coord_maps_inv.set_shape(shp_list)

    return warp_images
    # return warp_images, coord_maps, coord_maps_inv

def get_transform_matrix(shape, scales=None, oris=None, shifts=None):
    # scales: [B,1] output_size = s * input_shape
    # oris: [B,1] output_ori = clockwise rotation
    # shifts: [B,2] the same direction as image coordinate (right/bottom is positive)

    batch_size, height, width = tf.unstack(shape)[:3]
    height = tf.to_float(height)
    width = tf.to_float(width)
    T = tf.tile(tf.eye(3)[None], [batch_size, 1, 1])
    zeros = tf.zeros([batch_size, 1], dtype=tf.float32)
    ones = tf.ones([batch_size, 1], dtype=tf.float32)

    C1 = tf.concat([ones, zeros, -ones*(width-1)/2,
                    zeros, ones, -ones*(height-1)/2,
                    zeros, zeros, ones,
                   ], axis=1)
    C1 = tf.reshape(C1, [-1,3,3])
    C2 = tf.concat([ones, zeros, ones*(width-1)/2,
                    zeros, ones, ones*(height-1)/2,
                    zeros, zeros, ones,
                   ], axis=1)
    C2 = tf.reshape(C2, [-1,3,3])

    # T = C2

    if oris is not None:
        oris = -oris
        sins = tf.sin(oris)
        coss = tf.cos(oris)
        R = tf.concat([coss, -sins, zeros,
                      sins,  coss, zeros,
                      zeros, zeros, ones,
                     ], axis=1)
        R = tf.reshape(R, [-1,3,3])
        T = tf.matmul(T, R)
    if scales is not None:
        scales = 1.0 / scales
        S = tf.concat([scales, zeros, zeros,
                      zeros, scales, zeros,
                      zeros, zeros, ones,
                     ], axis=1)
        S = tf.reshape(S, [-1,3,3])
        T = tf.matmul(T, S)
    if shifts is not None:
        shifts = -shifts
        us = tf.slice(shifts, [0,0], [-1,1])
        vs = tf.slice(shifts, [0,1], [-1,1])
        D = tf.concat([ones,  zeros, us,
                       zeros, ones,  vs,
                       zeros, zeros, ones,
                      ], axis=1)
        D = tf.reshape(D, [-1,3,3])
        T = tf.matmul(T, D)
    # T = tf.matmul(T, C1)
    return T

def centerized_transform(T, width, height, name='centerized_transform'):
    # T : [batch_size, 3, 3]
    with tf.name_scope('centerized_transform'):
        batch_size = tf.shape(T)[0]
        height = tf.to_float(height)
        width = tf.to_float(width)

        zeros = tf.zeros([batch_size, 1], dtype=tf.float32)
        ones = tf.ones([batch_size, 1], dtype=tf.float32)

        C1 = tf.concat([ones, zeros, -ones*(width-1)/2,
                        zeros, ones, -ones*(height-1)/2,
                        zeros, zeros, ones,
                       ], axis=1)
        C1 = tf.reshape(C1, [-1,3,3])
        C2 = tf.concat([ones, zeros, ones*(width-1)/2,
                        zeros, ones, ones*(height-1)/2,
                        zeros, zeros, ones,
                       ], axis=1)
        C2 = tf.reshape(C2, [-1,3,3])

        T = tf.matmul(T, C1)
        T = tf.matmul(C2, T)
        return T

def euclid_augmentation(images, max_rad=np.pi, max_scale=1.0, max_shift=0.05, disable_border=True):
    # images : 4D tensor [B,H,W,C]
    # max_rad : [-np.pi, np.pi]
    # max_scale : linear scale
    # max_shift : normalized coodinate [0,1] 1.0=image size 

    im_shape = tf.shape(images)
    batch_size = im_shape[0]
    src_height, src_width = tf.unstack(im_shape)[1:3]

    if disable_border:
        pad_size = tf.to_int32(tf.to_float(tf.maximum(src_height, src_width)) *  (2.0-1.0) / 2 + 0.5) # larger than usual (sqrt(2))
        images = tf.pad(images, [[0,0], [pad_size]*2, [pad_size]*2, [0,0]], 'REFLECT')
    im_shape = tf.shape(images)
    height, width = tf.unstack(tf.to_float(im_shape))[1:3]

    if max_rad == 0:
        # no-rotation
        oris = tf.zeros([batch_size,1], dtype=tf.float32)
    else:
        oris = -max_rad + 2 * max_rad * tf.random_uniform([batch_size,1])

    if max_scale == 1:
        # no-scale
        scales = tf.ones([batch_size,1], dtype=tf.float32)
    else:
        log_scale = tf.to_float(tf.log(max_scale)) # max_scale may be tf.float64
        scales = tf.exp(-log_scale+2*log_scale*tf.random_uniform([batch_size, 1]))

    if max_shift == 0:
        shifts = tf.zeros([batch_size,2], dtype=tf.float32)
    else:
        max_x = max_shift * tf.to_float(width)
        max_y = max_shift * tf.to_float(height)
        shifts_x = -max_x + 2 * max_x * tf.random_uniform([batch_size,1])
        shifts_y = -max_y + 2 * max_y * tf.random_uniform([batch_size,1])
        shifts = tf.concat([shifts_x, shifts_y], axis=1)

    transforms = get_transform_matrix(im_shape, scales=scales, oris=oris, shifts=shifts) # [batch_size,3,3]
    transforms_raw = tf.identity(transforms)
    transforms = centerized_transform(transforms, width, height)
    transform_params = tf.slice(tf.reshape(transforms, [-1,9]), [0,0], [-1,8]) # remove the last elements (always 1)

    images = tf.contrib.image.transform(images, transform_params)

    if disable_border:
        images = tf.slice(images, [0, pad_size, pad_size, 0], [-1, src_height, src_width, -1])

    return images
