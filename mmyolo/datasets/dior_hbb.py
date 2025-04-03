# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets import DIORHBBDataset

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from ..registry import DATASETS


@DATASETS.register_module()
class YOLOv5DIORHBBDataset(BatchShapePolicyDataset, DIORHBBDataset):
    """Dataset for YOLOv5 VOC Dataset.

    We only add `BatchShapePolicy` function compared with VOCDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    pass