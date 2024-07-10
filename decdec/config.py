# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_dec2dec_config(cfg):
    """
    Add config for DECDEC.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME ="mask_former_semantic"

    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # DECDEC model config
    cfg.MODEL.DECDEC = CN()

    # loss
    cfg.MODEL.DECDEC.DEEP_SUPERVISION = True
    cfg.MODEL.DECDEC.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.DECDEC.CLASS_WEIGHT = 1.0
    cfg.MODEL.DECDEC.DICE_WEIGHT = 1.0
    cfg.MODEL.DECDEC.MASK_WEIGHT = 20.0

    cfg.MODEL.DECDEC.NO_OBJECT_WEIGHT = 1e-5
    cfg.MODEL.DECDEC.CLASS_WEIGHT = 3.0
    cfg.MODEL.DECDEC.DICE_WEIGHT = 3.0
    cfg.MODEL.DECDEC.MASK_WEIGHT = 0.3
    cfg.MODEL.DECDEC.INSDIS_WEIGHT = 1.0
    cfg.MODEL.DECDEC.AUX_SEMANTIC_WEIGHT = 1.0
    cfg.MODEL.DECDEC.USE_AUX_SEMANTIC_DECODER = False

    cfg.MODEL.DECDEC.PIXEL_INSDIS_TEMPERATURE = 1.5
    cfg.MODEL.DECDEC.PIXEL_INSDIS_SAMPLE_K = 4096
    cfg.MODEL.DECDEC.AUX_SEMANTIC_TEMPERATURE = 2.0
    cfg.MODEL.DECDEC.AUX_SEMANTIC_SAMPLE_K = 4096
    cfg.MODEL.DECDEC.MASKING_VOID_PIXEL = True
    
    # transformer config
    cfg.MODEL.DECDEC.NHEADS = 8
    cfg.MODEL.DECDEC.DROPOUT = 0.1
    cfg.MODEL.DECDEC.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DECDEC.ENC_LAYERS = 0
    cfg.MODEL.DECDEC.DEC_LAYERS = 6
    cfg.MODEL.DECDEC.PRE_NORM = False

    cfg.MODEL.DECDEC.HIDDEN_DIM = 256
    cfg.MODEL.DECDEC.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.DECDEC.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.DECDEC.ENFORCE_INPUT_PROJ = False

    # DECDEC inference config
    cfg.MODEL.DECDEC.TEST = CN()
    cfg.MODEL.DECDEC.TEST.SEMANTIC_ON = True
    cfg.MODEL.DECDEC.TEST.INSTANCE_ON = False
    cfg.MODEL.DECDEC.TEST.PANOPTIC_ON = False
    cfg.MODEL.DECDEC.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.DECDEC.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.DECDEC.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    cfg.MODEL.DECDEC.SHARE_FINAL_MATCHING = True
    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.DECDEC.SIZE_DIVISIBILITY = 32

    # NOTE: maskformer2 extra configs

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 256
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    cfg.MODEL.NUM_CLASSES = 133
    cfg.MODEL.DECDEC.BACKBONE_NAME= 'resnet50'
    cfg.MODEL.DECDEC.FREEZE_BACKBONE= False
    cfg.MODEL.DECDEC.FEATURE_DIM= [2048,1024,512,256,256]
    cfg.MODEL.DECDEC.PIXEL_DECODER_LAYERS= [1,5,1,1]
    cfg.MODEL.DECDEC.TRANSFORMER_DECODER_LAYERS= [2,2,2,0]
    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.DECDEC.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.DECDEC.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.DECDEC.IMPORTANCE_SAMPLE_RATIO = 0.75

    cfg.OUTPUT_DIR = './output'