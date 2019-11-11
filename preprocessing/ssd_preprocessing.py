# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images.

The preprocessing steps for VGG were introduced in the following technical
report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import matplotlib.pyplot as plt


slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def _ImageDimensions(image, rank=3):
    """Returns the dimensions of an image tensor.

    Args:
      image: A rank-D Tensor. For 3-D  of shape: `[height, width, channels]`.
      rank: The expected rank of the image

    Returns:
      A list of corresponding to the dimensions of the
      input image.  Dimensions that are statically known are python integers,
      otherwise they are integer scalar tensors.
    """
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(image), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


# x 为归一化后的图像 convert_image，num_cases = 4
def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:
      x: input Tensor.
      func: Python function to apply.
      num_cases: Python int32, number of cases to sample sel from.

    Returns:
      The result of func(x, sel), where func receives the value of the
      selector as a python integer, but sel is sampled dynamically.
    """

    # 在　[0,maxval) 范围内产生一个服从均匀分布的随机数
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)

    # Pass the real x only to one of the func calls.
    # control_flow_ops.switch(data,pred)，当 pred=False 时，将 data 输出给 x[0]；当 pred=True 时，将 data 输出给 x[1]
    # cotrol_flow_ops.merge 返回输出的张量和对应的索引
    # control_flow_ops.merge 会在列表中输出一个可用的结果，达到随机进行一种数据增强的效果
    return control_flow_ops.merge([func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
                                   for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
      image: 3-D Tensor containing single image in [0, 1].
      color_ordering: Python int, a type of distortion (valid values: 0-3).
      fast_mode: Avoids slower ops (random_hue and random_contrast)
      scope: Optional scope for name_scope.
    Returns:
      3-D Tensor color-distorted image on range [0, 1]
    Raises:
      ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                # 在 [-max_delta,max_delta) 范围内随机调整图像亮度
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                # 在 [lower,upper) 范围内随机调整图像饱和度
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


def ssd_random_sample_patch(image, labels, bboxes, ratio_list=[0.1, 0.3, 0.5, 0.7, 0.9, 1.], name=None):
    '''ssd_random_sample_patch.
    select one min_iou
    sample _width and _height from [0-width] and [0-height]
    check if the aspect ratio between 0.5-2.
    select left_top point from (width - _width, height - _height)
    check if this bbox has a min_iou with all ground_truth bboxes
    keep ground_truth those center is in this sampled patch, if none then try again
    '''

    # 获取随机抽取图像块的宽度及高度
    def sample_width_height(width, height):
        with tf.name_scope('sample_width_height'):
            index = 0
            max_attempt = 10
            sampled_width, sampled_height = width, height

            # 抽取的图像块宽高比应在 [0.5,2] 范围内
            def condition(index, sampled_width, sampled_height, width, height):
                return tf.logical_or(tf.logical_and(tf.logical_or(tf.greater(sampled_width, sampled_height * 2),
                                                                  tf.greater(sampled_height, sampled_width * 2)),
                                                    tf.less(index, max_attempt)),
                                     tf.less(index, 1))

            def body(index, sampled_width, sampled_height, width, height):
                # 产生图像随机块宽度与高度随机值，使随机块占原图比例为 [0.1,1]
                sampled_width = tf.random_uniform([1], minval=0.3, maxval=0.999, dtype=tf.float32)[0] * width
                sampled_height = tf.random_uniform([1], minval=0.3, maxval=0.999, dtype=tf.float32)[0] * height

                return index + 1, sampled_width, sampled_height, width, height

            [index, sampled_width, sampled_height, _, _] = tf.while_loop(condition, body,
                                                                         [index, sampled_width, sampled_height, width,
                                                                          height], parallel_iterations=4,
                                                                         back_prop=False, swap_memory=True)

            return tf.cast(sampled_width, tf.int32), tf.cast(sampled_height, tf.int32)


    # 利用 Python 的广播机制，获取 roi 与每个标注框的 iou 值
    def jaccard_with_anchors(roi, bboxes):
        with tf.name_scope('jaccard_with_anchors'):
            int_ymin = tf.maximum(roi[0], bboxes[:, 0])
            int_xmin = tf.maximum(roi[1], bboxes[:, 1])
            int_ymax = tf.minimum(roi[2], bboxes[:, 2])
            int_xmax = tf.minimum(roi[3], bboxes[:, 3])
            h = tf.maximum(int_ymax - int_ymin, 0.)
            w = tf.maximum(int_xmax - int_xmin, 0.)
            inter_vol = h * w
            union_vol = (roi[3] - roi[1]) * (roi[2] - roi[0]) + (
                    (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]) - inter_vol)
            jaccard = tf.div(inter_vol, union_vol)
            return jaccard


    def areas(bboxes):
        with tf.name_scope('bboxes_areas'):
            vol = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            return vol

    def check_roi_center(width, height, labels, bboxes):
        with tf.name_scope('check_roi_center'):
            index = 0
            max_attempt = 20
            roi = [0., 0., 0., 0.]
            float_width = tf.cast(width, tf.float32)
            float_height = tf.cast(height, tf.float32)
            mask = tf.cast(tf.zeros_like(labels, dtype=tf.uint8), tf.bool)

            # 获取标注框中心 x,y 坐标
            center_x, center_y = (bboxes[:, 1] + bboxes[:, 3]) / 2, (bboxes[:, 0] + bboxes[:, 2]) / 2

            def condition(index, roi, mask):
                return tf.logical_or(tf.logical_and(tf.reduce_sum(tf.cast(mask, tf.int32)) < 1,
                                                    tf.less(index, max_attempt)),
                                     tf.less(index, 1))

            def body(index, roi, mask):
                # 获取抽取图像块高度及宽度
                sampled_width, sampled_height = sample_width_height(float_width, float_height)

                # 以 x,y 为左上角坐标抽取一个图像块
                x = tf.random_uniform([], minval=0, maxval=width - sampled_width, dtype=tf.int32)
                y = tf.random_uniform([], minval=0, maxval=height - sampled_height, dtype=tf.int32)

                # 抽取图像块左上角及右下角归一化后的顶点坐标
                roi = [tf.cast(y, tf.float32) / float_height,
                       tf.cast(x, tf.float32) / float_width,
                       tf.cast(y + sampled_height, tf.float32) / float_height,
                       tf.cast(x + sampled_width, tf.float32) / float_width]

                # 当 mask=True 时，标注框中心位于抽取的图像块内
                # 标注框中心坐标应位于抽取图像块右下方
                mask_min = tf.logical_and(tf.greater(center_y, roi[0]), tf.greater(center_x, roi[1]))
                # 标注框中心坐标应位于抽取图像块左上方
                mask_max = tf.logical_and(tf.less(center_y, roi[2]), tf.less(center_x, roi[3]))

                # mask 反映标注框中心是否在抽取的图像块内部
                mask = tf.logical_and(mask_min, mask_max)

                return index + 1, roi, mask

            [index, roi, mask] = tf.while_loop(condition, body, [index, roi, mask], parallel_iterations=10,
                                               back_prop=False, swap_memory=True)

            # 返回 mask 为 True (标注框中心位于抽取图像块中)位置处的 labels 及 bboxes 值
            mask_labels = tf.boolean_mask(labels, mask)
            mask_bboxes = tf.boolean_mask(bboxes, mask)

            return roi, mask_labels, mask_bboxes


    # min_iou 随机从 [0.1,0.3,0.5,0.7,0.9] 中获取
    def check_roi_overlap(width, height, labels, bboxes, min_iou):
        with tf.name_scope('check_roi_overlap'):
            index = 0
            max_attempt = 50
            roi = [0., 0., 1., 1.]
            mask_labels = labels
            mask_bboxes = bboxes

            def condition(index, roi, mask_labels, mask_bboxes):
                return tf.logical_or(tf.logical_or(tf.logical_and(
                    # 需要随机抽取的图像块与 mask_bboxes 的重叠度大于一个随机抽取的 min_iou
                    # jaccard_with_anchors 获取 roi 与每个 bboxes 的 IoU 值
                    tf.reduce_sum(tf.cast(jaccard_with_anchors(roi, mask_bboxes) < min_iou, tf.int32)) > 0,
                    tf.less(index, max_attempt)),
                    tf.less(index, 1)),
                    tf.less(tf.shape(mask_labels)[0], 1))

            def body(index, roi, mask_labels, mask_bboxes):
                # roi：标注框中心位于抽取图像块内时的抽取图像块左上角及右下角归一化顶点坐标
                # mask_bboxes: 标注框中心位于抽取的图像块内部的标注框坐标
                roi, mask_labels, mask_bboxes = check_roi_center(width, height, labels, bboxes)
                return index + 1, roi, mask_labels, mask_bboxes

            [index, roi, mask_labels, mask_bboxes] = tf.while_loop(condition, body,
                                                                   [index, roi, mask_labels, mask_bboxes],
                                                                   parallel_iterations=16, back_prop=False,
                                                                   swap_memory=True)

            # 当抽取的图像块中不包含标注框的中心时直接返回原图，不进行图像块抽取
            # 否则返回抽取的图像块的左上角坐标及高度与宽度，以及原图中标注框中心位于抽取的图像块内部的标注框坐标
            return tf.cond(tf.greater(tf.shape(mask_labels)[0], 0),
                           lambda: (tf.cast([roi[0] * tf.cast(height, tf.float32),
                                             roi[1] * tf.cast(width, tf.float32),
                                             (roi[2] - roi[0]) * tf.cast(height, tf.float32),
                                             (roi[3] - roi[1]) * tf.cast(width, tf.float32)], tf.int32),
                                    mask_labels, mask_bboxes),
                           lambda: (tf.cast([0, 0, height, width], tf.int32), labels, bboxes))

    def sample_patch(image, labels, bboxes, min_iou):
        with tf.name_scope('sample_patch'):
            height, width, depth = _ImageDimensions(image, rank=3)

            # 从原图中获取符合条件的随机抽取的图像块，或者直接返回原图
            # 这些条件包括：图像块与标注框 IoU 值大于给定值、图像块宽高比在 [0.5,2] 之间、图像块占原图大小比例为 [0.1,1]
            # roi_slice_range：抽取的图像块左上角坐标及抽取的图像块高度及宽度（如果没有抽取返回的就是原图高度及宽度）
            roi_slice_range, mask_labels, mask_bboxes = check_roi_overlap(width, height, labels, bboxes, min_iou)

            scale = tf.cast(tf.stack([height, width, height, width]), mask_bboxes.dtype)

            # 广播，对应元素相乘
            mask_bboxes = mask_bboxes * scale

            # Add offset.
            offset = tf.cast(tf.stack([roi_slice_range[0], roi_slice_range[1], roi_slice_range[0], roi_slice_range[1]]),
                             mask_bboxes.dtype)
            mask_bboxes = mask_bboxes - offset

            cliped_ymin = tf.maximum(0., mask_bboxes[:, 0])
            cliped_xmin = tf.maximum(0., mask_bboxes[:, 1])
            cliped_ymax = tf.minimum(tf.cast(roi_slice_range[2], tf.float32), mask_bboxes[:, 2])
            cliped_xmax = tf.minimum(tf.cast(roi_slice_range[3], tf.float32), mask_bboxes[:, 3])

            mask_bboxes = tf.stack([cliped_ymin, cliped_xmin, cliped_ymax, cliped_xmax], axis=-1)
            # Rescale to target dimension.
            scale = tf.cast(tf.stack([roi_slice_range[2], roi_slice_range[3],
                                      roi_slice_range[2], roi_slice_range[3]]), mask_bboxes.dtype)

            return tf.cond(tf.logical_or(tf.less(roi_slice_range[2], 1), tf.less(roi_slice_range[3], 1)),
                           lambda: (image, labels, bboxes),
                           lambda: (tf.slice(image, [roi_slice_range[0], roi_slice_range[1], 0],
                                             [roi_slice_range[2], roi_slice_range[3], -1]),
                                    mask_labels, mask_bboxes / scale))

    with tf.name_scope('ssd_random_sample_patch'):
        image = tf.convert_to_tensor(image, name='image')

        min_iou_list = tf.convert_to_tensor(ratio_list)

        # 以均匀分布产生范围在 [0,5] 之内的随机数
        samples_min_iou = tf.multinomial(tf.log([[1. / len(ratio_list)] * len(ratio_list)]), 1)

        # 根据产生的随机数获取一个最小交并比值
        sampled_min_iou = min_iou_list[tf.cast(samples_min_iou[0][0], tf.int32)]

        # 当获得的交并比为 1 时，表示不进行 patch 抽取，否则需要抽取一个 patch
        return tf.cond(tf.less(sampled_min_iou, 1.),
                       lambda: sample_patch(image, labels, bboxes, sampled_min_iou),
                       lambda: (image, labels, bboxes))


# ratio 取 [1.1,4) 范围内均匀分布的随机数
def ssd_random_expand(image, bboxes, ratio=2., name=None):
    with tf.name_scope('ssd_random_expand'):
        image = tf.convert_to_tensor(image, name='image')
        if image.get_shape().ndims != 3:
            raise ValueError('\'image\' must have 3 dimensions.')

        height, width, depth = _ImageDimensions(image, rank=3)

        float_height, float_width = tf.to_float(height), tf.to_float(width)

        # ratio范围: (1.1~4)
        canvas_width, canvas_height = tf.to_int32(float_width * ratio), tf.to_int32(float_height * ratio)

        mean_color_of_image = [_R_MEAN / 255., _G_MEAN / 255.,
                               _B_MEAN / 255.]  # tf.reduce_mean(tf.reshape(image, [-1, 3]), 0)

        # maxval:[0.1,3)*width
        x = tf.random_uniform([], minval=0, maxval=canvas_width - width, dtype=tf.int32)
        y = tf.random_uniform([], minval=0, maxval=canvas_height - height, dtype=tf.int32)

        paddings = tf.convert_to_tensor([[y, canvas_height - height - y], [x, canvas_width - width - x]])

        big_canvas = tf.stack([tf.pad(image[:, :, 0], paddings, "CONSTANT", constant_values=mean_color_of_image[0]),
                               tf.pad(image[:, :, 1], paddings, "CONSTANT", constant_values=mean_color_of_image[1]),
                               tf.pad(image[:, :, 2], paddings, "CONSTANT", constant_values=mean_color_of_image[2])],
                              axis=-1)

        scale = tf.cast(tf.stack([height, width, height, width]), bboxes.dtype)
        absolute_bboxes = bboxes * scale + tf.cast(tf.stack([y, x, y, x]), bboxes.dtype)

        return big_canvas, absolute_bboxes / tf.cast(
            tf.stack([canvas_height, canvas_width, canvas_height, canvas_width]), bboxes.dtype)


# def ssd_random_sample_patch_wrapper(image, labels, bboxes):
#   with tf.name_scope('ssd_random_sample_patch_wrapper'):
#     orgi_image, orgi_labels, orgi_bboxes = image, labels, bboxes
#     def check_bboxes(bboxes):
#       areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
#       return tf.logical_and(tf.logical_and(areas < 0.9, areas > 0.001),
#                             tf.logical_and((bboxes[:, 3] - bboxes[:, 1]) > 0.025, (bboxes[:, 2] - bboxes[:, 0]) > 0.025))

#     index = 0
#     max_attempt = 3
#     def condition(index, image, labels, bboxes):
#       return tf.logical_or(tf.logical_and(tf.reduce_sum(tf.cast(check_bboxes(bboxes), tf.int64)) < 1, tf.less(index, max_attempt)), tf.less(index, 1))

#     def body(index, image, labels, bboxes):
#       image, bboxes = tf.cond(tf.random_uniform([], minval=0., maxval=1., dtype=tf.float32) < 0.5,
#                       lambda: (image, bboxes),
#                       lambda: ssd_random_expand(image, bboxes, tf.random_uniform([1], minval=1.1, maxval=4., dtype=tf.float32)[0]))
#       # Distort image and bounding boxes.
#       random_sample_image, labels, bboxes = ssd_random_sample_patch(image, labels, bboxes, ratio_list=[-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.])
#       random_sample_image.set_shape([None, None, 3])
#       return index+1, random_sample_image, labels, bboxes

#     [index, image, labels, bboxes] = tf.while_loop(condition, body, [index, orgi_image, orgi_labels, orgi_bboxes], parallel_iterations=4, back_prop=False, swap_memory=True)

#     valid_mask = check_bboxes(bboxes)
#     labels, bboxes = tf.boolean_mask(labels, valid_mask), tf.boolean_mask(bboxes, valid_mask)
#     return tf.cond(tf.less(index, max_attempt),
#                 lambda : (image, labels, bboxes),
#                 lambda : (orgi_image, orgi_labels, orgi_bboxes))

def ssd_random_sample_patch_wrapper(image, labels, bboxes):
    with tf.name_scope('ssd_random_sample_patch_wrapper'):
        orgi_image, orgi_labels, orgi_bboxes = image, labels, bboxes

        def check_bboxes(bboxes):
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            return tf.logical_and(tf.logical_and(areas < 0.9, areas > 0.001),
                                  tf.logical_and((bboxes[:, 3] - bboxes[:, 1]) > 0.025,
                                                 (bboxes[:, 2] - bboxes[:, 0]) > 0.025))

        index = 0
        max_attempt = 3

        def condition(index, image, labels, bboxes, orgi_image, orgi_labels, orgi_bboxes):
            return tf.logical_or(tf.logical_and(tf.reduce_sum(tf.cast(check_bboxes(bboxes), tf.int64)) < 1,
                                                tf.less(index, max_attempt)), tf.less(index, 1))

        # 使用均匀分布，按照 0.5 的概率，返回不经处理的图像或者经 random_expand 的图像
        def body(index, image, labels, bboxes, orgi_image, orgi_labels, orgi_bboxes):
            image, bboxes = tf.cond(tf.random_uniform([], minval=0., maxval=1., dtype=tf.float32) < 0.5,
                                    lambda: (orgi_image, orgi_bboxes),
                                    lambda: ssd_random_expand(orgi_image, orgi_bboxes,
                                                              tf.random_uniform([1], minval=1.1, maxval=4.,
                                                                                dtype=tf.float32)[0]))
            # Distort image and bounding boxes.
            # 在图像中抽取一个小块
            random_sample_image, labels, bboxes = ssd_random_sample_patch(image, orgi_labels, bboxes,
                                                                          ratio_list=[-0.1, 0.1, 0.3, 0.5, 0.7, 0.9,
                                                                                      1.])
            random_sample_image.set_shape([None, None, 3])
            return index + 1, random_sample_image, labels, bboxes, orgi_image, orgi_labels, orgi_bboxes

        # while 循环。condition判断条件，body 循环体，初始条件
        [index, image, labels, bboxes, orgi_image, orgi_labels, orgi_bboxes] = \
            tf.while_loop(condition, body, [index, image, labels, bboxes, orgi_image, orgi_labels, orgi_bboxes],
                          parallel_iterations=4, back_prop=False, swap_memory=True)

        valid_mask = check_bboxes(bboxes)
        labels, bboxes = tf.boolean_mask(labels, valid_mask), tf.boolean_mask(bboxes, valid_mask)
        return tf.cond(tf.less(index, max_attempt),
                       lambda: (image, labels, bboxes),
                       lambda: (orgi_image, orgi_labels, orgi_bboxes))


def _mean_image_subtraction(image, means):
    """Subtracts the given means from each image channel.

    For example:
      means = [123.68, 116.779, 103.939]
      image = _mean_image_subtraction(image, means)

    Note that the rank of `image` must be known.

    Args:
      image: a tensor of size [height, width, C].
      means: a C-vector of values to subtract from each channel.

    Returns:
      the centered image.

    Raises:
      ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


def unwhiten_image(image):
    means = [_R_MEAN, _G_MEAN, _B_MEAN]
    num_channels = image.get_shape().as_list()[-1]
    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] += means[i]
    return tf.concat(axis=2, values=channels)


def random_flip_left_right(image, bboxes):
    with tf.name_scope('random_flip_left_right'):
        uniform_random = tf.random_uniform([], 0, 1.0)
        mirror_cond = tf.less(uniform_random, .5)
        # Flip image.
        result = tf.cond(mirror_cond, lambda: tf.image.flip_left_right(image), lambda: image)
        # Flip bboxes.
        mirror_bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3],
                                  bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
        bboxes = tf.cond(mirror_cond, lambda: mirror_bboxes, lambda: bboxes)
        return result, bboxes


# 调试函数，查看图像预处理结果
# image:(?,?,3)
def preprocess_visualize(image):
    fig = plt.figure()
    plt.imshow(image)
    plt.show()

    return True


def preprocess_for_train(image, labels, bboxes, out_shape, data_format='channels_first',
                         scope='ssd_preprocessing_train', output_rgb=True):
    """Preprocesses the given image for training.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      labels: A `Tensor` containing all labels for all bboxes of this image.
      bboxes: A `Tensor` containing all bboxes of this image, in range [0., 1.] with shape [num_bboxes, 4].
      out_shape: The height and width of the image after preprocessing.
      data_format: The data_format of the desired output image.
    Returns:
      A preprocessed image.
    """

    # 在某个 tf.name_scope 指定的区域中定义的对象，name 属性上都会增加该命名区的区域名
    # 将不同的对象及操作放在由 tf.name_scope 指定的区域中，便于在 tensorboard 中展示清晰的
    # 逻辑关系图，方便参数命名管理
    with tf.name_scope(scope, 'ssd_preprocessing_train', [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        # Convert to float scaled [0, 1].
        # 转为浮点型并归一化到 [0,1] 范围
        # 当输入是整形，输出是浮点型时，tf.image.convert_image_type 将对图像进行归一化
        # image:(?,?,3)
        orig_dtype = image.dtype
        if orig_dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)   # default_name: convert_image

        debug_op = tf.py_func(preprocess_visualize, [image], [tf.bool])
        summary_op=tf.summary.image('image_norm',tf.expand_dims(image,axis=0))
        with tf.control_dependencies([summary_op]):   # 在 tensorboard 的 graph 中有虚线表示该操作
            image = tf.identity(image,name='image')

        # image = tf.Print(image, [image], message='image after normailzed')

        # Randomly distort the colors. There are 4 ways to do it.
        # 随机对图像进行亮度调整、对比度调整
        distort_image = apply_with_random_selector(image,
                                                   lambda x, ordering: distort_color(x, ordering, True),
                                                   num_cases=4)

        #
        random_sample_image, labels, bboxes = ssd_random_sample_patch_wrapper(distort_image, labels, bboxes)
        # image, bboxes = tf.cond(tf.random_uniform([1], minval=0., maxval=1., dtype=tf.float32)[0] < 0.25,
        #                     lambda: (image, bboxes),
        #                     lambda: ssd_random_expand(image, bboxes, tf.random_uniform([1], minval=2, maxval=4, dtype=tf.int32)[0]))

        # # Distort image and bounding boxes.
        # random_sample_image, labels, bboxes = ssd_random_sample_patch(image, labels, bboxes, ratio_list=[0.1, 0.3, 0.5, 0.7, 0.9, 1.])

        # Randomly flip the image horizontally.
        # 按照 0.5 的概率水平翻转图像
        random_sample_flip_image, bboxes = random_flip_left_right(random_sample_image, bboxes)

        # Rescale to VGG input scale.
        # 图像缩放到 300*300 大小
        # 对于目标检测，设置 align=False，对边角不友好，但能带来整数倍的上下采样，方便了坐标值的计算
        random_sample_flip_resized_image = tf.image.resize_images(random_sample_flip_image,
                                                                  out_shape, method=tf.image.ResizeMethod.BILINEAR,
                                                                  align_corners=False)
        random_sample_flip_resized_image.set_shape([None, None, 3])

        final_image = tf.to_float(
            tf.image.convert_image_dtype(random_sample_flip_resized_image, orig_dtype, saturate=True))

        # 最后减去图像均值
        # mean:[123.68,116.78,103.94]
        final_image = _mean_image_subtraction(final_image, [_R_MEAN, _G_MEAN, _B_MEAN])

        # 当图像大小不能从 graph 中推断出来时，使用 set_shape 可以帮助了解图像维度
        final_image.set_shape(out_shape + [3])

        # 图像 rgb 转 bgr
        if not output_rgb:
            image_channels = tf.unstack(final_image, axis=-1, name='split_rgb')
            final_image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=-1, name='merge_bgr')
        if data_format == 'channels_first':
            final_image = tf.transpose(final_image, perm=(2, 0, 1))
        return final_image, labels, bboxes


def preprocess_for_eval(image, out_shape, data_format='channels_first', scope='ssd_preprocessing_eval',
                        output_rgb=True):
    """Preprocesses the given image for evaluation.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      out_shape: The height and width of the image after preprocessing.
      data_format: The data_format of the desired output image.
    Returns:
      A preprocessed image.
    """
    with tf.name_scope(scope, 'ssd_preprocessing_eval', [image]):
        image = tf.to_float(image)

        # 关于 align_corners 的设置，对于目标检测，设置为 false 对边角情况不友好，但能带来整数倍的上下采样，方便了坐标值的计算
        # 对于语义分割，由于边角像素也要纳入 mIOU 的计算，设置为 false 会对精度造成明显的影响
        image = tf.image.resize_images(image, out_shape, method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
        image.set_shape(out_shape + [3])  # tf.set_shape 可以用来增加维度信息

        image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])

        # rgb 转 bgr
        if not output_rgb:
            image_channels = tf.unstack(image, axis=-1, name='split_rgb')
            image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=-1, name='merge_bgr')
        # Image data format.
        if data_format == 'channels_first':
            image = tf.transpose(image, perm=(2, 0, 1))
        return image


def preprocess_image(image, labels, bboxes, out_shape, is_training=False, data_format='channels_first',
                     output_rgb=True):
    """Preprocesses the given image.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      labels: A `Tensor` containing all labels for all bboxes of this image.
      bboxes: A `Tensor` containing all bboxes of this image, in range [0., 1.] with shape [num_bboxes, 4].
      out_shape: The height and width of the image after preprocessing.
      is_training: Wether we are in training phase.
      data_format: The data_format of the desired output image.

    Returns:
      A preprocessed image.
    """
    if is_training:
        return preprocess_for_train(image, labels, bboxes, out_shape, data_format=data_format, output_rgb=output_rgb)
    else:
        return preprocess_for_eval(image, out_shape, data_format=data_format, output_rgb=output_rgb)
