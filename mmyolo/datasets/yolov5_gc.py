# Copyright (c) OpenMMLab. All rights reserved.
####  cLass registered in MMDET
# from mmdet.registry import DATASETS
# from .xml_style import XMLDataset
# @DATASETS.register_module()
# class NEUDataset(XMLDataset):
#     # CLASSES = ('crackles',)
#     # PALETTE = [(106, 0, 228)]
#     # CLASSES = ('crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches')
#     # PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
#     #            (106, 0, 228), (0, 60, 100)]
#     # CLASSES = (
#     #     "1_chongkong", "2_hanfeng", "3_yueyawan", "4_shuiban", "5_youban", "6_siban", "7_yiwu", "8_yahen", "9_zhehen",
#     #     "10_yaozhe")
#     # PALETTE = [(106, 0, 228), (119, 11, 32), (165, 42, 42), (109, 63, 54),
#     #            (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
#     #            (153, 69, 1), (120, 166, 157)]
#
#     METAINFO = {
#         'classes':
#             ('1_chongkong', '2_hanfeng', '3_yueyawan', '4_shuiban', '5_youban', '6_siban', '7_yiwu', '8_yahen',
#              '9_zhehen', '10_yaozhe'),
#         # palette is a list of color tuples, which is used for visualization.
#         'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (109, 63, 54),
#                (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
#                (153, 69, 1), (120, 166, 157)]
#     }
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
