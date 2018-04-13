import Mask_RCNN.utils as utils
import Mask_RCNN.model as modellib

from Mask_RCNN.bowl_config import bowl_config
from Mask_RCNN.bowl_dataset import BowlDataset
from Mask_RCNN.model import log

from glob import glob

import os

import train_config

init_with = train_config.init_with
CHECKPOINT_PATH = train_config.CHECKPOINT_PATH

if not os.path.exists(train_config.TRAIN_DIR):
    print ("No valid trainig directory found ...")
    sys.exit(1)

if init_with == "checkpoint"  and not os.path.exists(CHECKPOINT_PATH):
    print ("No valid trainig checkpoint file found ...")
    sys.exit(1)

# Create directory for saving checkpoints and logs:
if not os.path.exists(train_config.MODEL_DIR):
    os.mkdir(train_config.MODEL_DIR)

# Download COCO trained weights from Releases if needed
if not os.path.exists(train_config.COCO_MODEL_PATH):
    utils.download_trained_weights(train_config.COCO_MODEL_PATH)

model = modellib.MaskRCNN(mode="training", 
                          config=bowl_config,
                          model_dir=train_config.MODEL_DIR)

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), 
                       by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(train_config.COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", 
                                "mrcnn_bbox_fc", 
                                "mrcnn_bbox", 
                                "mrcnn_mask"])
elif init_with == "checkpoint":
    model.load_weights(train_config.CHECKPOINT_PATH, 
                       by_name=False)
    print ("Loaded model weights from checkpoint ...")
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)
    print ("Loaded model weights from last training epoch ...")

dataset_train = BowlDataset()
dataset_train.load_bowl(train_config.TRAIN_DIR)
dataset_train.prepare()

dataset_val = BowlDataset()
dataset_val.load_bowl(train_config.TRAIN_DIR)
dataset_val.prepare()

model.train(dataset_train, dataset_val, 
            learning_rate=train_config.LEARNING_RATE / 10,
            epochs=2, 
            layers="all")