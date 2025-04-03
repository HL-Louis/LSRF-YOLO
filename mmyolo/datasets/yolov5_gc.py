# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets import NEUDataset

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from ..registry import DATASETS


@DATASETS.register_module()
class YOLOv5GCDataset(BatchShapePolicyDataset, NEUDataset):
    """Dataset for YOLOv5 VOC Dataset.

    We only add `BatchShapePolicy` function compared with VOCDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    pass
