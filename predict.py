import sys
sys.path.insert(0, "decdec")
import tempfile
from pathlib import Path
import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

# import decdec project
from decdec import add_dec2dec_config

def setup():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_dec2dec_config(cfg)
    cfg.merge_from_file("decdec/configs/coco/panoptic-segmentation/dec2dec_R50_bs16_50ep.yaml")
    cfg.MODEL.WEIGHTS = 'model_final.pth'
    cfg.MODEL.DECDEC.TEST.SEMANTIC_ON = True
    cfg.MODEL.DECDEC.TEST.INSTANCE_ON = True
    cfg.MODEL.DECDEC.TEST.PANOPTIC_ON = True
    cfg.INPUT.MIN_SIZE_TEST = 256
    cfg.INPUT.MAX_SIZE_TEST = 256
    cfg.INPUT.IMAGE_SIZE=256
    predictor = DefaultPredictor(cfg)
    coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")
    return predictor, coco_metadata

def predict(predictor, coco_metadata, path):
    im = cv2.imread(str(path))
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"),
                                            outputs["panoptic_seg"][1]).get_image()
    v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
    v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
    result = np.concatenate((panoptic_result, instance_result, semantic_result), axis=0)[:, :, ::-1]
    out_path = Path(tempfile.mkdtemp()) / "out.png"
    cv2.imwrite(str(out_path), result)
    return out_path

predictor, coco_metadata = setup()
path = input()
print(predict(predictor, coco_metadata, path))