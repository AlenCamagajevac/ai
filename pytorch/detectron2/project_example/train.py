# import some common libraries
import cv2

# import some common detectron2 utilities
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import os
import random

# import other libraries
from detectron2.data.datasets.coco import load_coco_json


path_ds = os.path.join("detectron2", "project_example", "ds")  # Update project name
path_ds_meta = os.path.join("detectron2", "project_example", "ds_metadata")  # Update project name
ds_file_base = "_coco.json"
ds_train = "train"
ds_validation = "val"
path_output = os.path.join("detectron2", "project_example", "output")  # Update project name

# Change model by selecting a new one from the model zoo and updating following 2 lines
# https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
path_config = os.path.join("detectron2", "detectron2_repo", "configs", "COCO-Detection",
                           "faster_rcnn_R_101_FPN_3x.yaml")
model_weights = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"


def main():
    setup_logger()

    for d in [ds_train, ds_validation]:
        DatasetCatalog.register(os.path.join(path_ds, d), lambda d=d:
                                load_coco_json(os.path.join(path_ds_meta, ds_train + ds_file_base),
                                               os.path.join(path_ds, ds_train)))
        # Update class names
        MetadataCatalog.get(os.path.join(path_ds, d)).set(thing_classes=["apple_green", "apple_red", "clementine",
                                                                         "cucumber", "garlic", "lemon", "onion"])
    project_metadata = MetadataCatalog.get(os.path.join(path_ds, ds_train))

    cfg = get_cfg()
    cfg.merge_from_file(path_config)
    cfg.DATASETS.TRAIN = (os.path.join(path_ds, ds_train), )
    cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_weights  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR (original: 0.00025)
    cfg.SOLVER.MAX_ITER = 1000  # you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this small dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # Update number of classes
    cfg.OUTPUT_DIR = path_output

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # Set the testing threshold for this model
    cfg.DATASETS.TEST = (os.path.join(path_ds + ds_validation), )
    predictor = DefaultPredictor(cfg)

    dataset_dicts = load_coco_json(os.path.join(path_ds_meta, ds_validation + ds_file_base),
                                   os.path.join(path_ds, ds_validation))
    for d in random.sample(dataset_dicts, 3):  # Show n validation images
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=project_metadata, scale=1)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow('object_detection', v.get_image()[:, :, ::-1])
        cv2.waitKey()


if __name__ == "__main__":
    main()
