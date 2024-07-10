from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modeling import (
    Backbone, 
    PixelDecoderLayerNoCA,
    PixelDecoderLayer,
    TransformerDecoderLayer,
    Predictor,
    ConvBN,
)
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, ImageList, Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling.postprocessing import sem_seg_postprocess

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher


@META_ARCH_REGISTRY.register()
class DecDec(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        criterion: nn.Module,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,

        input_size = 256,
        # model parameters
        backbone_name = 'resnet50',
        freeze_backbone = False,
        num_queries = 100,
        num_classes = 133,
        feature_dim=[2048,1024,512,256,256],
        pixel_decoder_layers=[1,5,1,1],
        transformer_decoder_layers=[2,2,2,0],
        d_model = 256,
        ffn_dim = 2048,
        n_head = 8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference


        self.pixel_decoder_layers = pixel_decoder_layers
        self.transformer_decoder_layers = transformer_decoder_layers
        self.object_query = nn.Parameter(torch.zeros(1,num_queries,d_model,requires_grad=True))
        self.backbone = Backbone(backbone_name,freeze_backbone)

        self.pixel_decoder = nn.ModuleList()
        self.transformer_decoder = nn.ModuleList()
        input_size = input_size//32
        for i in range(3):
            if i==0:
                layer = nn.ModuleList([PixelDecoderLayerNoCA(input_size,feature_dim[i],d_model,n_head,ffn_dim,dropout=0.1, activation_fn = 'gelu',interpolate = False)]*(pixel_decoder_layers[i]-1))
                layer.append(PixelDecoderLayerNoCA(input_size,feature_dim[i],d_model,n_head,ffn_dim,dropout=0.1, activation_fn = 'gelu',interpolate = True))
                self.pixel_decoder.append(layer)
            else:
                layer = nn.ModuleList([PixelDecoderLayer(input_size,feature_dim[i],d_model,n_head,ffn_dim,dropout=0.1, activation_fn = 'gelu',interpolate = False)]*(pixel_decoder_layers[i]-1))
                layer.append(PixelDecoderLayer(input_size,feature_dim[i],d_model,n_head,ffn_dim,dropout=0.1, activation_fn = 'gelu',interpolate = True))
                self.pixel_decoder.append(layer)
            input_size*=2
            self.transformer_decoder.append(nn.ModuleList([TransformerDecoderLayer(feature_dim[i+1],d_model,n_head,ffn_dim,dropout=0.1,activation_fn='gelu')]*transformer_decoder_layers[i]))
        layer = nn.ModuleList([PixelDecoderLayer(input_size,feature_dim[-1],d_model,n_head,ffn_dim,dropout=0.1, activation_fn = 'gelu',interpolate = False)]*pixel_decoder_layers[-1])
        self.pixel_decoder.append(layer)

        self.final_feature_conv = ConvBN(feature_dim[-1],feature_dim[-1],kernel_size=1, bias=False, norm='syncbn', act=None)

        self.aux_seg_head = nn.ModuleList([Predictor(in_channel_pixel=feature_dim[i+1],in_channel_query=d_model,num_classes=num_classes+1) for i in range(4)])
        self.seg_head = Predictor(in_channel_pixel=feature_dim[-1],in_channel_query=d_model,num_classes=num_classes+1)
        self.class_proj = ConvBN(256, 256, kernel_size=1, bias=False, norm='syncbn', act='gelu',conv_type='1d')
        self.mask_proj = ConvBN(256, 256, kernel_size=1, bias=False, norm='syncbn', act='gelu',conv_type='1d')

    @classmethod
    def from_config(cls, cfg):
        # Loss parameters:
        deep_supervision = cfg.MODEL.DECDEC.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.DECDEC.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.DECDEC.CLASS_WEIGHT
        dice_weight = cfg.MODEL.DECDEC.DICE_WEIGHT
        mask_weight = cfg.MODEL.DECDEC.MASK_WEIGHT

        # building criterion
        # matcher = HungarianMatcher(
        #     cost_class=class_weight,
        #     cost_mask=mask_weight,
        #     cost_dice=dice_weight,
        #     num_points=cfg.MODEL.DECDEC.TRAIN_NUM_POINTS,
        # )
        matcher = HungarianMatcher(masking_void_pixel=cfg.MODEL.DECDEC.MASKING_VOID_PIXEL)

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.DECDEC.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        # criterion = SetCriterion(
        #     cfg.MODEL.NUM_CLASSES,
        #     matcher=matcher,
        #     weight_dict=weight_dict,
        #     eos_coef=no_object_weight,
        #     losses=losses,
        #     num_points=cfg.MODEL.DECDEC.TRAIN_NUM_POINTS,
        #     oversample_ratio=cfg.MODEL.DECDEC.OVERSAMPLE_RATIO,
        #     importance_sample_ratio=cfg.MODEL.DECDEC.IMPORTANCE_SAMPLE_RATIO,
        # )
        criterion = SetCriterion(
            cfg.MODEL.NUM_CLASSES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            share_final_matching=cfg.MODEL.DECDEC.SHARE_FINAL_MATCHING,
            pixel_insdis_temperature=cfg.MODEL.DECDEC.PIXEL_INSDIS_TEMPERATURE,
            pixel_insdis_sample_k=cfg.MODEL.DECDEC.PIXEL_INSDIS_SAMPLE_K,
            aux_semantic_temperature=cfg.MODEL.DECDEC.AUX_SEMANTIC_TEMPERATURE,
            aux_semantic_sample_k=cfg.MODEL.DECDEC.AUX_SEMANTIC_SAMPLE_K,
            masking_void_pixel=cfg.MODEL.DECDEC.MASKING_VOID_PIXEL,
        )

        return {
            "criterion": criterion,
            "object_mask_threshold": cfg.MODEL.DECDEC.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.DECDEC.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.DECDEC.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.DECDEC.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.DECDEC.TEST.PANOPTIC_ON
                or cfg.MODEL.DECDEC.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.DECDEC.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.DECDEC.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.DECDEC.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,

            'input_size' : cfg.INPUT.IMAGE_SIZE,
            'backbone_name' : cfg.MODEL.DECDEC.BACKBONE_NAME,
            'freeze_backbone' : cfg.MODEL.DECDEC.FREEZE_BACKBONE,
            'num_queries' : cfg.MODEL.DECDEC.NUM_OBJECT_QUERIES,
            'num_classes' : cfg.MODEL.NUM_CLASSES,
            'feature_dim':cfg.MODEL.DECDEC.FEATURE_DIM,
            'pixel_decoder_layers':cfg.MODEL.DECDEC.PIXEL_DECODER_LAYERS,
            'transformer_decoder_layers':cfg.MODEL.DECDEC.TRANSFORMER_DECODER_LAYERS,
            'd_model' : cfg.MODEL.DECDEC.HIDDEN_DIM,
            'ffn_dim' : cfg.MODEL.DECDEC.DIM_FEEDFORWARD,
            'n_head' : cfg.MODEL.DECDEC.NHEADS,
        }
    @property   
    def device(self):
        return self.pixel_mean.device
    
    def forward(self,batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        bsz, c,h,w = images.tensor.shape
        feature = self.backbone(images.tensor)
        object_query = self.object_query.repeat(bsz,1,1)
        aux_outputs = []
        for idx,pix_layers in enumerate(self.pixel_decoder_layers):
            for i in range(pix_layers):
                feature = self.pixel_decoder[idx][i](feature,object_query)
            for i in range(self.transformer_decoder_layers[idx]):
                object_query = self.transformer_decoder[idx][i](feature,object_query)

            object_query = object_query.transpose(1,2)
            class_emb = self.class_proj(object_query)
            mask_emb = self.mask_proj(object_query)
            aux_outputs.append(self.aux_seg_head[idx](class_emb,mask_emb,feature))
            object_query = object_query.transpose(1,2)
        aux_outputs.reverse()
        feature = F.interpolate(feature,scale_factor=(4,4))
        feature = self.final_feature_conv(feature)

        object_query = object_query.transpose(1,2)
        class_emb = self.class_proj(object_query)
        mask_emb = self.mask_proj(object_query)
        
        # predict
        prediction_result = self.seg_head(class_emb,mask_emb,feature)

        # predictions_class.append(prediction_result['class_logits'])
        # predictions_mask.append(prediction_result['mask_logits'])
        # predictions_pixel_feature.append(prediction_result['pixel_feature'])
        outputs = {
            'pred_logits': prediction_result['pred_logits'],
            'pred_masks': prediction_result['pred_masks'],
            'pixel_feature': prediction_result['pixel_feature'],
            'aux_outputs': aux_outputs
        }

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results
        # return out
    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result


    
# x = torch.randn(1,3,256,256)
# model = DecDec()
# out = model(x)
# print(out['pred_logits'].shape,out['pred_masks'].shape,out['pixel_feature'].shape)
# for i in out['aux_results']:
#     print(i.shape)