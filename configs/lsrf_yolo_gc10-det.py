# _base_ = ['../_base_/default_runtime.py']
_base_ = ['../_base_/default_runtime.py', '../_base_/det_p5_tta.py']

# ========================Frequently modified parameters======================
# -----data related-----
data_root = 'data/New_GC-DET/'
# data_root = 'data/APDDD'
# data_root = '../mmdetection2.0/data/NEU-DET/'
dataset_type = 'YOLOv5NEUDataset'


num_classes = 10 # Number of classes for classification
# Batch size of a single GPU during training
train_batch_size_per_gpu = 16
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 8
# persistent_workers must be False if num_workers is 0
persistent_workers = True

# -----model related-----
# Basic size of multi-scale prior box
# anchors = [
#     [(10, 13), (16, 30), (33, 23)],  # P3/8
#     [(30, 61), (62, 45), (59, 119)],  # P4/16
#     [(116, 90), (156, 198), (373, 326)]  # P5/32
# ]
# anchors = [
#     [(24, 16), (37, 50), (283, 20)],  # P3/8
#     [(71, 71), (58, 131), (501, 18)],  # P4/16
#     [(79, 206), (148, 193), (359, 188)]  # P5/32
# ]
# [[(12, 13), (27, 16), (40, 40)], [(61, 73), (252, 20), (508, 19)], [(68, 146), (124, 205), (232, 210)]]
# [[(12, 12), (27, 16), (40, 40)], [(59, 72), (253, 20), (509, 18)], [(73, 136), (74, 219), (155, 206)]]

# GC10-DEt
anchors = [
    [(12, 13), (27, 16), (40, 40)],  # P3/8
    [(61, 73), (252, 20), (508, 19)],  # P4/16
    [(68, 146), (124, 205), (232, 210)]]  # P5/32

#NEU-DET
# anchors = [
#     [(12, 16), (19, 36), (40, 28)],  # P3/8
#     [(36, 75), (76, 55), (72, 146)],  # P4/16
#     [(142, 110), (192, 243), (459, 401)]]  # P5/32


# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=128 bs

max_epochs = 300  # Maximum training epochs
num_epochs_stage2 = 20
model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.5),  # NMS type and threshold
    max_per_img=100)  # Max number of detections of each image

# ========================Possible modified parameters========================
# -----data related-----
img_scale = (512,512)  # width, height
# Dataset type, this will be used to define the dataset

# Batch size of a single GPU during validation
val_batch_size_per_gpu = 16
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2

# Config of batch shapes. Only on val.
# It means not used if batch_shapes_cfg is None.
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    # The image scale of padding should be divided by pad_size_divisor
    size_divisor=32,
    # Additional paddings for pixel scale
    extra_pad_ratio=0.5)

# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.5

# deepen_factor = 0.167
# widen_factor = 0.375
# Strides of multi-scale prior box
strides = [8, 16, 32]
num_det_layers = 3  # The number of model output scales
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config

# -----train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
loss_cls_weight = 0.5
loss_bbox_weight = 0.05
loss_obj_weight = 1.0
prior_match_thr = 4.  # Priori box matching threshold
# The obj loss weights of the three output layers
obj_level_weights = [4., 1., 0.4]
lr_factor = 0.01  # Learning rate scaling factor

# Save model checkpoint and validation intervals
save_checkpoint_intervals = 10
# The maximum checkpoints to keep.
max_keep_ckpts = 3
# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# ===============================Unmodified in most cases====================
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        # mean=[0., 0., 0.],
        # std=[255., 255., 255.],
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
# mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
        bgr_to_rgb=True),
# type='YOLODetector',
#     data_preprocessor=dict(
#         type='YOLOv5DetDataPreprocessor',
#         mean=[103.53, 116.28, 123.675],
#         std=[57.375, 57.12, 58.395],
#         bgr_to_rgb=False),
#     backbone=dict(
#         type='YOLOv5CSPDarknet',
#         deepen_factor=deepen_factor,
#         widen_factor=widen_factor,
#         norm_cfg=norm_cfg,
#         act_cfg=dict(type='SiLU', inplace=True)),
    # neck=dict(
    #     type='YOLOv6RepPAFPN',
    #     deepen_factor=deepen_factor,
    #     widen_factor=widen_factor,
    #     in_channels=[256, 512, 1024],
    #     out_channels=[128, 256, 512],
    #     num_csp_blocks=12,
    #     norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
    #     act_cfg=dict(type='ReLU', inplace=True),
    # ),
    # neck=dict(
    #     type='YOLOv5PAFPN',
    #     deepen_factor=deepen_factor,
    #     widen_factor=widen_factor,
    #     in_channels=[256, 512, 1024],
    #     out_channels=[256, 512, 1024],
    #     num_csp_blocks=3,
    #     norm_cfg=norm_cfg,
    #     act_cfg=dict(type='SiLU', inplace=True)),
backbone=dict(
        type='YOLO_TR',
        arch='P5',
        expand_ratio=0.5,
        out_indices=(1,2,3,4),
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        channel_attention=True,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
# backbone=dict(
#         type='YOLOv7Backbone',
#         arch='Tiny',
#         norm_cfg=norm_cfg,
#         act_cfg=dict(type='SiLU', inplace=True)),
#     neck=dict(
# is_tiny_version=True,
#         type='YOLOv7PAFPN',
#         block_cfg=dict(
#             type='TinyDownSampleBlock', middle_ratio=0.25),
#         upsample_feats_cat_first=False,
#         in_channels=[128, 256, 512],
#         # The real output channel will be multiplied by 2
#         out_channels=[64, 128, 256],
#         norm_cfg=norm_cfg,
#         act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
#
#
#         use_repconv_outs=False),
#     bbox_head=dict(
#         type='YOLOv7Head',
#         head_module=dict(
#             type='YOLOv7HeadModule',
#             num_classes=num_classes,
#             in_channels=[128, 256, 512],
#             featmap_strides=strides,
#             num_base_priors=3),
#         prior_generator=dict(
#             type='mmdet.YOLOAnchorGenerator',
#             base_sizes=anchors,
#             strides=strides),
#         # scaled based on number of detection layers
#         loss_cls=dict(
#             type='mmdet.CrossEntropyLoss',
#             use_sigmoid=True,
#             reduction='mean',
#             loss_weight=loss_cls_weight *
#             (num_classes / 80 * 3 / num_det_layers)),
#         loss_bbox=dict(
#             type='IoULoss',
#             iou_mode='ciou',
#             bbox_format='xywh',
#             reduction='mean',
#             loss_weight=loss_bbox_weight * (3 / num_det_layers),
#             return_iou=True),
#         loss_obj=dict(
#             type='mmdet.CrossEntropyLoss',
#             use_sigmoid=True,
#             reduction='mean',
#             loss_weight=loss_obj_weight *
#             ((img_scale[0] / 640)**2 * 3 / num_det_layers)),
#         prior_match_thr=prior_match_thr,
#         obj_level_weights=obj_level_weights,
#         # BatchYOLOv7Assigner params
#         simota_candidate_topk=10,
#         simota_iou_weight=3.0,
#         simota_cls_weight=1.0),
#     test_cfg=model_test_cfg)

# backbone=dict(
#         type='YOLOv6EfficientRep',
#         deepen_factor=deepen_factor,
#         widen_factor=widen_factor,
#         norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
#         act_cfg=dict(type='ReLU', inplace=True)),
#     neck=dict(
#         type='YOLOv6RepPAFPN',
#         deepen_factor=deepen_factor,
#         widen_factor=widen_factor,
#         in_channels=[256, 512, 1024],
#         out_channels=[128, 256, 512],
#         num_csp_blocks=12,
#         norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
#         act_cfg=dict(type='ReLU', inplace=True),
#     ),
#     bbox_head=dict(
#         type='YOLOv6Head',
#         head_module=dict(
#             type='YOLOv6HeadModule',
#             num_classes=num_classes,
#             in_channels=[128, 256, 512],
#             widen_factor=widen_factor,
#             norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
#             act_cfg=dict(type='SiLU', inplace=True),
#             featmap_strides=[8, 16, 32]),
#         loss_bbox=dict(
#             type='IoULoss',
#             iou_mode='giou',
#             bbox_format='xyxy',
#             reduction='mean',
#             loss_weight=2.5,
#             return_iou=False)),
#     train_cfg=dict(
#         initial_epoch=4,
#         initial_assigner=dict(
#             type='BatchATSSAssigner',
#             num_classes=num_classes,
#             topk=9,
#             iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
#         assigner=dict(
#             type='BatchTaskAlignedAssigner',
#             num_classes=num_classes,
#             topk=13,
#             alpha=1,
#             beta=6),
#     ),
#     test_cfg=dict(
#         multi_label=True,
#         nms_pre=30000,
#         score_thr=0.001,
#         nms=dict(type='nms', iou_threshold=0.65),
#         max_per_img=300))
#
# backbone=dict(
#         type='YOLOv8CSPDarknet',
#         arch='P5',
#         last_stage_out_channels=1024,
#         deepen_factor=deepen_factor,
#         widen_factor=widen_factor,
#         norm_cfg=norm_cfg,
#         act_cfg=dict(type='SiLU', inplace=True)),
#     neck=dict(
#         type='YOLOv8PAFPN',
#         deepen_factor=deepen_factor,
#         widen_factor=widen_factor,
#         in_channels=[256, 512, 1024],
#         out_channels=[256, 512, 1024],
#         num_csp_blocks=3,
#         norm_cfg=norm_cfg,
#         act_cfg=dict(type='SiLU', inplace=True)),
#     bbox_head=dict(
#         type='YOLOv8Head',
#         head_module=dict(
#             type='YOLOv8HeadModule',
#             num_classes=num_classes,
#             in_channels=[256, 512, 1024],
#             widen_factor=widen_factor,
#             reg_max=16,
#             norm_cfg=norm_cfg,
#             act_cfg=dict(type='SiLU', inplace=True),
#             featmap_strides=strides),
#         prior_generator=dict(
#             type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
#         bbox_coder=dict(type='DistancePointBBoxCoder'),
#         # scaled based on number of detection layers
#         loss_cls=dict(
#             type='mmdet.CrossEntropyLoss',
#             use_sigmoid=True,
#             reduction='none',
#             loss_weight=0.5),
#         loss_bbox=dict(
#             type='IoULoss',
#             iou_mode='ciou',
#             bbox_format='xyxy',
#             reduction='sum',
#             loss_weight=7.5,
#             return_iou=False),
#         loss_dfl=dict(
#             type='mmdet.DistributionFocalLoss',
#             reduction='mean',
#             loss_weight=1.5 / 4)),
#     train_cfg=dict(
#         assigner=dict(
#             type='BatchTaskAlignedAssigner',
#             num_classes=num_classes,
#             use_ciou=True,
#             topk=10,
#             alpha=0.5,
#             beta=6.0,
#             eps=1e-9)),
# backbone=dict(
#         type='CSPNeXt',
#         arch='P5',
#         expand_ratio=0.5,
# # out_indices = (1, 2, 3, 4),
#         deepen_factor=deepen_factor,
#         widen_factor=widen_factor,
#         channel_attention=True,
#         norm_cfg=norm_cfg,
#         act_cfg=dict(type='SiLU', inplace=True)),
#     neck=dict(
#         type='CSPNeXtPAFPN',
#         deepen_factor=deepen_factor,
#         widen_factor=widen_factor,
#         in_channels=[256, 512, 1024],
#
#         out_channels=256,
#         num_csp_blocks=3,
#         expand_ratio=0.5,
#         norm_cfg=norm_cfg,
#         act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='RSNeck',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[128,256, 512, 1024],
        # in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
# # #
    bbox_head=dict(
        type='RTMDetHead',
        head_module=dict(
            type='RTMDetSepBNHeadModule',
            widen_factor=widen_factor,
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            share_conv=True,
            pred_kernel_size=1,
            featmap_strides=strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=2.0)),
    # bbox_head=dict(
    #     type='YOLOv5Head',
    #     head_module=dict(
    #         type='YOLOv5HeadModule',
    #         num_classes=num_classes,
    #         in_channels=[256, 512, 1024],
    #         widen_factor=widen_factor,
    #         featmap_strides=strides,
    #         num_base_priors=3),
    #     prior_generator=dict(
    #         type='mmdet.YOLOAnchorGenerator',
    #         base_sizes=anchors,
    #         strides=strides),
    #     # scaled based on number of detection layers
    #     loss_cls=dict(
    #         type='mmdet.CrossEntropyLoss',
    #         use_sigmoid=True,
    #         reduction='mean',
    #         loss_weight=loss_cls_weight *
    #         (num_classes / 80 * 3 / num_det_layers)),
    #     loss_bbox=dict(
    #         type='IoULoss',
    #         iou_mode='ciou',
    #         bbox_format='xywh',
    #         eps=1e-7,
    #         reduction='mean',
    #         loss_weight=loss_bbox_weight * (3 / num_det_layers),
    #         return_iou=True),
    #     loss_obj=dict(
    #         type='mmdet.CrossEntropyLoss',
    #         use_sigmoid=True,
    #         reduction='mean',
    #         loss_weight=loss_obj_weight *
    #         ((img_scale[0] / 640)**2 * 3 / num_det_layers)),
    #     prior_match_thr=prior_match_thr,
    #     obj_level_weights=obj_level_weights),
train_cfg=dict(
        assigner=dict(
            type='BatchDynamicSoftLabelAssigner',
            num_classes=num_classes,
            topk=13,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=model_test_cfg)

# albu_train_transforms = [
#     dict(
#         type='ShiftScaleRotate',
#         shift_limit=0.0625,
#         scale_limit=0.0,
#         rotate_limit=0,
#         interpolation=1,
#         p=0.5),
#     dict(
#         type='RandomBrightnessContrast',
#         brightness_limit=[0.1, 0.3],
#         contrast_limit=[0.1, 0.3],
#         p=0.2),
#     dict(
#         type='OneOf',
#         transforms=[
#             dict(
#                 type='RGBShift',
#                 r_shift_limit=10,
#                 g_shift_limit=10,
#                 b_shift_limit=10,
#                 p=1.0),
#             dict(
#                 type='HueSaturationValue',
#                 hue_shift_limit=20,
#                 sat_shift_limit=30,
#                 val_shift_limit=20,
#                 p=1.0)
#         ],
#         p=0.1),
#     dict(type='ImageCompression', quality_lower=85, quality_upper=95, p=0.2),
#     dict(type='ChannelShuffle', p=0.1),
#     dict(
#         type='OneOf',
#         transforms=[
#             dict(type='Blur', blur_limit=3, p=1.0),
#             dict(type='MedianBlur', blur_limit=3, p=1.0)
#         ],
#         p=0.1),
# ]
# pre_transform = [
#     dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
#     dict(type='LoadAnnotations', with_bbox=True)
# ]
#
# train_pipeline = [
#     *pre_transform,
#     dict(
#         type='Mosaic',
#         img_scale=img_scale,
#         use_cached=True,
#         max_cached_images=40,
#         pad_val=114.0),
# dict(
#         type='mmdet.RandomResize',
#         # img_scale is (width, height)
#         scale=(img_scale[0] * 2, img_scale[1] * 2),
#         ratio_range=(0.1, 2.0),
#         resize_type='mmdet.Resize',
#         keep_ratio=True),
#     # dict(
#     #     type='YOLOv5RandomAffine',
#     #     max_rotate_degree=0.0,
#     #     max_shear_degree=0.0,
#     #     scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
#     #     # img_scale is (width, height)
#     #     border=(-img_scale[0] // 2, -img_scale[1] // 2),
#     #     border_val=(114, 114, 114)),
#
#     # dict(type='YOLOv5HSVRandomAug'),
#     # dict(type='mmdet.RandomFlip', prob=0.5),
#     dict(type='mmdet.RandomCrop', crop_size=img_scale),
#     dict(type='mmdet.YOLOXHSVRandomAug'),
#     dict(type='mmdet.RandomFlip', prob=0.5),
#     dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
#     dict(
#             type='YOLOv5MixUp',
#             use_cached=True,
#             max_cached_images=20),
# # dict(
# #         type='mmdet.Albu',
# #         transforms=albu_train_transforms,
# #         bbox_params=dict(
# #             type='BboxParams',
# #             format='pascal_voc',
# #             label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
# #         keymap={
# #             'img': 'image',
# #             'gt_bboxes': 'bboxes'
# #         }),
#     dict(
#         type='mmdet.PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
#                    'flip_direction'))
# ]


train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=40,
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        # img_scale is (width, height)
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=(0.1, 2.0),
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    # dict(
    #     type='YOLOv5MixUp',
    #     use_cached=True,
    #     max_cached_images=20),
    dict(type='mmdet.PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.RandomResize',
        scale=img_scale,
        ratio_range=(0.1, 2.0),
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]
# train_pipeline_stage2 = [
#     dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='mmdet.RandomResize',
#         # img_scale is (width, height)
#         scale=(img_scale[0] * 2, img_scale[1] * 2),
#         ratio_range=(0.1, 2.0),
#         resize_type='mmdet.Resize',
#         keep_ratio=True),
#     dict(type='mmdet.RandomCrop', crop_size=img_scale),
#     dict(type='mmdet.YOLOXHSVRandomAug'),
#     dict(type='mmdet.RandomFlip', prob=0.5),
#     dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
#     dict(
#         type='YOLOv5MixUp',
#         use_cached=True,
#         max_cached_images=20),
#     dict(type='mmdet.PackDetInputs')
# ]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/Main/train.txt',
        data_prefix=dict(sub_data_root='images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

    # dataset=dict(
    #     type='RepeatDataset',
    #     times = 3,
    #     dataset=(
    #
    #         dict(
    #             type=dataset_type,
    #             data_root=data_root,
    #             ann_file='main/train.txt',
    #             data_prefix=dict(sub_data_root='images/'),
    #             filter_cfg=dict(filter_empty_gt=False, min_size=32),
    #             pipeline=train_pipeline)
    #     )
    #
    # ))


test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

# test_pipeline = [
#     dict(
#         type='LoadImageFromFile', file_client_args=_base_.file_client_args
#         ),
#     dict(type='mmdet.Resize', scale=img_scale, keep_ratio=False), # 这里将 LetterResize 修改成 mmdet.Resize
#     dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
#     dict(
#         type='mmdet.PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        ann_file='ImageSets/Main/test.txt',
        data_prefix=dict(sub_data_root='images/'),
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = val_dataloader
base_lr = 0.01
weight_decay = 0.0005
param_scheduler = None
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOv5OptimizerConstructor')
# base_lr = 0.004
# weight_decay = 0.05
# lr_start_factor = 1.0e-5
# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=lr_start_factor,
#         by_epoch=False,
#         begin=0,
#         end=1000),
#     dict(
#         # use cosine lr from 150 to 300 epoch
#         type='CosineAnnealingLR',
#         eta_min=base_lr * 0.05,
#         begin=max_epochs // 2,
#         end=max_epochs,
#         T_max=max_epochs // 2,
#         by_epoch=True,
#         convert_to_iter_based=True),
# ]
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=base_lr, weight_decay=weight_decay),
#     paramwise_cfg=dict(
#         norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# default_hooks = dict(
#     param_scheduler=dict(
#         type='YOLOv5ParamSchedulerHook',
#         scheduler_type='linear',
#         lr_factor=lr_factor,
#         max_epochs=max_epochs),
#     checkpoint=dict(
#         type='CheckpointHook',
#         interval=save_checkpoint_intervals,
#         save_best='auto',
#         max_keep_ckpts=max_keep_ckpts))
default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='cosine',
        lr_factor=lr_factor,
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_checkpoint_intervals,
        max_keep_ckpts=max_keep_ckpts,
        save_best='auto'))

# custom_hooks = [
#     dict(
#         type='EMAHook',
#         ema_type='ExpMomentumEMA',
#         momentum=0.0001,
#         update_buffers=True,
#         strict_load=False,
#         priority=49)
# ]
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - num_epochs_stage2,
        switch_pipeline=train_pipeline_stage2)]

val_evaluator = dict(
    type='mmdet.VOCMetric', metric='mAP', eval_mode='area')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_checkpoint_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# set_random_seed= 42
# load_from = 'smgc10.pth'