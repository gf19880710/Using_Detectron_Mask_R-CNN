#!/usr/bin/env python

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder. Allows for using a combination of multiple models.
For example, one model may be used for RPN, another model for Fast R-CNN style
box detection, yet another model to predict masks, and yet another model to
predict keypoints.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import os
import sys
import yaml

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import load_cfg
from detectron.core.config import merge_cfg_from_cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
import detectron.core.rpn_generator as rpn_engine
import detectron.core.test_engine as model_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# 权重文件路径
rpn_pkl = "/home/gengfeng/Desktop/projects/DETECTRON/out_dir/train/coco_labelme_train/generalized_rcnn/model_final.pkl"
# 配置文件路径
rpn_cfg = "/home/gengfeng/Desktop/projects/DETECTRON/configs/used/e2e_mask_rcnn_R-50-FPN_1x.yaml"


def get_rpn_box_proposals(im):
    """
    功能: 获取传入图像的region proposal

    输入参数列表:
        im: BGR格式图片
    返回参数列表:
        boxes: 建议boxes
        scores: 对应的分数
    """
    cfg.immutable(False)
    merge_cfg_from_file(rpn_cfg)
    cfg.NUM_GPUS = 1
    cfg.MODEL.RPN_ONLY = True
    cfg.TEST.RPN_PRE_NMS_TOP_N = 10000
    cfg.TEST.RPN_POST_NMS_TOP_N = 2000
    assert_and_infer_cfg(cache_urls=False)

    model = model_engine.initialize_model_from_cfg(rpn_pkl)
    with c2_utils.NamedCudaScope(0):
        boxes, scores = rpn_engine.im_proposals(model, im)
    return boxes, scores


def get_detectron_result(im):
    """
    功能: 获取传入图像的检测结果

    输入参数列表:
        im: BGR格式图片
    返回参数列表:
        img: 检测结果图片
        detectron_result_info: 检测结果字典
    """
    setup_logging(__name__)
    logger = logging.getLogger(__name__)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    cfg_orig = load_cfg(yaml.dump(cfg))

    if rpn_pkl is not None:
        proposal_boxes, _proposal_scores = get_rpn_box_proposals(im)
        workspace.ResetWorkspace()
    else:
        proposal_boxes = None

    cls_boxes, cls_segms, cls_keyps = None, None, None
    pkl = rpn_pkl
    yml = rpn_cfg
    cfg.immutable(False)
    merge_cfg_from_cfg(cfg_orig)
    merge_cfg_from_file(yml)
    if len(pkl) > 0:
        weights_file = pkl
    else:
        weights_file = cfg.TEST.WEIGHTS
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg(cache_urls=False)
    model = model_engine.initialize_model_from_cfg(weights_file)
    with c2_utils.NamedCudaScope(0):
        cls_boxes_, cls_segms_, cls_keyps_ = \
            model_engine.im_detect_all(model, im, proposal_boxes)
    cls_boxes = cls_boxes_ if cls_boxes_ is not None else cls_boxes
    cls_segms = cls_segms_ if cls_segms_ is not None else cls_segms
    cls_keyps = cls_keyps_ if cls_keyps_ is not None else cls_keyps
    workspace.ResetWorkspace()

    img, detectron_result_info = vis_utils.vis_one_image_opencv(
        im,
        cls_boxes,
        cls_segms,
        cls_keyps,
        dataset=dummy_coco_dataset,
        show_box=True,
        show_class=True
    )

    return img, detectron_result_info
