#
# Image processing
# Some codes come from  https://github.com/rpautrat/SuperPoint
# input image is supposed to be 3D tensor [H,W,C] and floating 0~255 values
# 
import cv2 as cv
import numpy as np
import tensorflow as tf
from misc import get_rank

def ratio_preserving_resize(image, size, method=tf.image.ResizeMethod.BILINEAR):
    # image is 4D or 3D tensor
    # size = tuple of (height, width)

    src_size = tf.shape(image)[:2] if get_rank(image) == 3 else tf.shape(image)[1:3]

    target_size = tf.convert_to_tensor(size)
    scales = tf.to_float(tf.divide(target_size, src_size)) # safe cast
    new_size = tf.to_float(src_size) * tf.reduce_max(scales)
    image = tf.image.resize_images(image, tf.to_int32(new_size),
                                   method=method)
    return tf.image.resize_image_with_crop_or_pad(image, target_size[0], target_size[1]), scales

def center_crop(images, crop_size, name='center_crop'):
    # crop_size = int or (width, height)
    # images is 4D or 3D tensor
    # If you treat intrinsic matrix, you need to modify it as well
    with tf.name_scope(name):
        if isinstance(crop_size, int):
            crop_width = crop_height = crop_size
        else:
            crop_width, crop_height = crop_size

        shp = images.get_shape().as_list()
        ndim = len(shp)
        xaxis, yaxis = get_xy_axis(ndim)
        height = shp[yaxis]
        width = shp[xaxis]

        ofst_y = (height - crop_height) // 2
        ofst_x = (width - crop_width) // 2

        begins = [0] * ndim
        begins[yaxis] = ofst_y
        begins[xaxis] = ofst_x
        ends = [-1] * ndim
        ends[yaxis] = crop_height
        ends[xaxis] = crop_width

        images = tf.slice(images, begins, ends)
        new_shp = shp
        new_shp[yaxis] = crop_height
        new_shp[xaxis] = crop_width
        images.set_shape(new_shp)

        return images

def gauss_blur(image, ksize, sigma=0):
    # images is 4D tensor [B,H,W,C] or 3D tensor [H,W,C]
    # if sigma is non-positive, it is computed automatically followed by https://docs.opencv.org/3.2.0/d4/d86/group__imgproc__filter.html

    is_4D = get_rank(image) == 4
    channel = tf.shape(image)[-1]

    kernel = cv.getGaussianKernel(ksize, sigma)[:, 0]
    kernel = np.outer(kernel, kernel).astype(np.float32)
    kernel = tf.reshape(tf.convert_to_tensor(kernel), [ksize]*2+[1, 1])
    kernel = tf.tile(kernel,[1,1,channel,1])
    pad_size = int(ksize/2)
    if is_4D:
        image = tf.pad(image, [[0, 0], [pad_size]*2, [pad_size]*2, [0, 0]], 'REFLECT')
        image = tf.nn.depthwise_conv2d(image, kernel, [1, 1, 1, 1], 'VALID')
    else:
        image = tf.pad(image, [[pad_size]*2, [pad_size]*2, [0, 0]], 'REFLECT')
        image = tf.expand_dims(image, axis=0)  # add batch dim
        image = tf.nn.depthwise_conv2d(image, kernel, [1, 1, 1, 1], 'VALID')[0]

    return image
    
def bilinear_sampling(photos, coords):
    """Construct a new image by bilinear sampling from the input image.

    Points falling outside the source image boundary have value 0.

    Args:
        photos: source image to be sampled from [batch, height_s, width_s, channels]
        coords: coordinates of source pixels to sample from [batch, height_t,
          width_t, 2]. height_t/width_t correspond to the dimensions of the output
          image (don't need to be the same as height_s/width_s). The two channels
          correspond to x and y coordinates respectively.
    Returns:
        A new sampled image [batch, height_t, width_t, channels]
    """ 
    # photos: [batch_size, height2, width2, C]
    # coords: [batch_size, height1, width1, C]
    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
            ])), 1), [1, 0])
        rep = tf.cast(rep, 'float32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    with tf.name_scope('image_sampling'):
        coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
        inp_size = tf.shape(photos)
        coord_size = tf.shape(coords)

        out_size = tf.stack([coord_size[0], 
                             coord_size[1],
                             coord_size[2],
                             inp_size[3],
                            ]) 

        coords_x = tf.cast(coords_x, 'float32')
        coords_y = tf.cast(coords_y, 'float32')

        x0 = tf.floor(coords_x)
        x1 = x0 + 1
        y0 = tf.floor(coords_y)
        y1 = y0 + 1

        y_max = tf.cast(tf.shape(photos)[1] - 1, 'float32')
        x_max = tf.cast(tf.shape(photos)[2] - 1, 'float32')
        zero = tf.zeros([1], dtype='float32')

        x0_safe = tf.clip_by_value(x0, zero, x_max)
        y0_safe = tf.clip_by_value(y0, zero, y_max)
        x1_safe = tf.clip_by_value(x1, zero, x_max)
        y1_safe = tf.clip_by_value(y1, zero, y_max)

        ## bilinear interp weights, with points outside the grid having weight 0
        # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
        # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
        # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
        # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

        wt_x0 = x1_safe - coords_x
        wt_x1 = coords_x - x0_safe
        wt_y0 = y1_safe - coords_y
        wt_y1 = coords_y - y0_safe

        ## indices in the flat image to sample from
        dim2 = tf.cast(inp_size[2], 'float32')
        dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
        base = tf.reshape(
            _repeat(
                tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
                coord_size[1] * coord_size[2]),
            [out_size[0], out_size[1], out_size[2], 1])

        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = tf.reshape(x0_safe + base_y0, [-1])
        idx01 = x0_safe + base_y1
        idx10 = x1_safe + base_y0
        idx11 = x1_safe + base_y1

        
        ## sample from photos
        photos_flat = tf.reshape(photos, tf.stack([-1, inp_size[3]]))
        photos_flat = tf.cast(photos_flat, 'float32')

        im00 = tf.reshape(tf.gather(photos_flat, tf.cast(idx00, 'int32')), out_size)
        im01 = tf.reshape(tf.gather(photos_flat, tf.cast(idx01, 'int32')), out_size)
        im10 = tf.reshape(tf.gather(photos_flat, tf.cast(idx10, 'int32')), out_size)
        im11 = tf.reshape(tf.gather(photos_flat, tf.cast(idx11, 'int32')), out_size)

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        out_photos = tf.add_n([
            w00 * im00, w01 * im01,
            w10 * im10, w11 * im11
        ])
        
        return out_photos