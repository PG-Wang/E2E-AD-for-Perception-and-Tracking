point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian'
]
dataset_type = 'NuScenesTrackDataset'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFilesCustom'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='InstanceRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='Normalize3D',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad3D', size_divisor=32)
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesCustom'),
    dict(
        type='Normalize3D',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad3D', size_divisor=32)
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='NuScenesTrackDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/ada_track_infos_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle',
            'pedestrian'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        num_frames_per_sample=3,
        pipeline_single=[
            dict(type='LoadMultiViewImageFromFilesCustom'),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='InstanceRangeFilter',
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(
                type='Normalize3D',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad3D', size_divisor=32)
        ],
        pipeline_post=[
            dict(type='FormatBundle3DTrack'),
            dict(
                type='Collect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 'img',
                    'timestamp', 'l2g_r_mat', 'l2g_t'
                ],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                           'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
                           'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                           'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                           'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                           'pcd_rotation', 'pts_filename',
                           'transformation_3d_flow', 'scene_token'))
        ],
        use_valid_flag=True),
    val=dict(
        type='NuScenesTrackDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/ada_track_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[-50, -50, -5, 50, 50, 3]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'trailer', 'bus',
                            'construction_vehicle', 'bicycle', 'motorcycle',
                            'pedestrian', 'traffic_cone', 'barrier'
                        ],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        classes=[
            'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle',
            'pedestrian'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        pipeline_single=[
            dict(type='LoadMultiViewImageFromFilesCustom'),
            dict(
                type='Normalize3D',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad3D', size_divisor=32)
        ],
        pipeline_post=[
            dict(type='FormatBundle3DTrack'),
            dict(
                type='Collect3D',
                keys=['img', 'timestamp', 'l2g_r_mat', 'l2g_t'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                           'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
                           'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                           'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                           'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                           'pcd_rotation', 'pts_filename',
                           'transformation_3d_flow', 'scene_token'))
        ],
        num_frames_per_sample=1),
    test=dict(
        type='NuScenesTrackDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/ada_track_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[-50, -50, -5, 50, 50, 3]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'trailer', 'bus',
                            'construction_vehicle', 'bicycle', 'motorcycle',
                            'pedestrian', 'traffic_cone', 'barrier'
                        ],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        classes=[
            'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle',
            'pedestrian'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        pipeline_single=[
            dict(type='LoadMultiViewImageFromFilesCustom'),
            dict(
                type='Normalize3D',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad3D', size_divisor=32)
        ],
        pipeline_post=[
            dict(type='FormatBundle3DTrack'),
            dict(
                type='Collect3D',
                keys=['img', 'timestamp', 'l2g_r_mat', 'l2g_t'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                           'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
                           'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                           'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                           'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                           'pcd_rotation', 'pts_filename',
                           'transformation_3d_flow', 'scene_token'))
        ],
        num_frames_per_sample=1))
evaluation = dict(interval=2)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')],
    tags=dict(samples_per_gpu=1, workers_per_gpu=4, total_epochs=24))
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs_detr3d'
load_from = '/home/w/ADA-Track/pretrained/detr3d_resnet101.pth'
resume_from = None
workflow = [('train', 1)]
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
plugin = True
plugin_dir = 'plugin/track/'
_samples_per_gpu = 1
_workers_per_gpu = 4
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
model = dict(
    type='AlternatingDetAssoCamTracker',
    with_data_aug=True,
    feature_update_rate=0.0,
    use_grid_mask=True,
    num_classes=7,
    num_query=300,
    bbox_coder=dict(
        type='DETRTrack3DCoder',
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_num=100,
        num_classes=7),
    fix_feats=True,
    fix_neck=True,
    score_thresh=0.4,
    filter_score_thresh=0.35,
    miss_tolerance=5,
    affinity_thresh=0.3,
    hungarian=False,
    qim_args=dict(
        qim_type='QIMBase',
        merger_dropout=0,
        update_query_pos=True,
        fp_ratio=0.05,
        random_drop=0.1),
    img_backbone=dict(
        type='ResNet',
        with_cp=False,
        pretrained='open-mmlab://detectron2/resnet101_caffe',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=4,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    loss_cfg=dict(
        type='ParallelMatcher',
        num_classes=7,
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='HungarianAssigner3DTrack',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_asso=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.0,
            alpha=-1,
            loss_weight=10.0)),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='DETR3DAssoTrackingHead',
        num_classes=7,
        in_channels=256,
        num_cams=6,
        num_feature_levels=4,
        with_box_refine=True,
        transformer=dict(
            type='DETR3DAssoTrackingTransformer',
            decoder=dict(
                type='DETR3DAssoTrackingTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                association_network=dict(
                    type='EdgeAugmentedTransformerAssociationLayer',
                    embed_dims=256,
                    ffn_dims=256,
                    num_heads=8,
                    dropout=0.1,
                    norm_first=False,
                    cross_attn_value_gate=False),
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='Detr3DCrossAtten',
                            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                            num_points=1,
                            embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5)),
    train_cfg=dict(
        grid_size=[512, 512, 1],
        voxel_size=[0.2, 0.2, 8],
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        out_size_factor=4,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='GIoU3DCost', weight=0.0),
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])),
    test_cfg=dict(
        post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=[-51.2, -51.2],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        score_threshold=0.1,
        out_size_factor=4,
        voxel_size=[0.2, 0.2, 8],
        nms_type='rotate',
        pre_max_size=1000,
        post_max_size=83,
        nms_thr=0.2))
train_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(
        type='Collect3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 'img',
            'timestamp', 'l2g_r_mat', 'l2g_t'
        ],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                   'transformation_3d_flow', 'scene_token'))
]
test_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(
        type='Collect3D',
        keys=['img', 'timestamp', 'l2g_r_mat', 'l2g_t'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                   'transformation_3d_flow', 'scene_token'))
]
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=24)
find_unused_parameters = True
fp16 = dict(loss_scale='dynamic')
gpu_ids = range(0, 1)
