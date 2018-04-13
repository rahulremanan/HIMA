import os

use_config_file = True

if use_config_file:
    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "model/checkpoint")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "model/coco/mask_rcnn_coco.h5")

    # Local path to the training data
    TRAIN_DIR = './fixed_data/train'

    # What model initial model weights to use
    init_with = "checkpoint"  # imagenet, coco, checkpoint or last

    # Specify full path to the checkpoint file
    CHECKPOINT_PATH = './model/checkpoint/RCNN_checkpoint.h5'
    
    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 1e-4