import Mask_RCNN.utils as utils
import Mask_RCNN.model as modellib
import Mask_RCNN.functions as f

from Mask_RCNN.bowl_config import bowl_config
from Mask_RCNN.bowl_dataset import BowlDataset
from Mask_RCNN.model import log
from Mask_RCNN.inference_config import inference_config
from Mask_RCNN.utils import rle_encode, rle_decode, rle_to_string

from glob import glob

import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

import os
import time

ROOT_DIR = os.getcwd()

MODEL_DIR = os.path.join(ROOT_DIR, "model/checkpoint")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "RCNN_checkpoint.h5")
SUBMISSION_FILE = "./stage2_sample_submission_final.csv"
FILE_PREFIX = "./sub-dsbowl2018-RCNN_"
TEST_DIR = "./test_2/"
TRAIN_DIR = "./test_2/"
init_with = "checkpoint"

gen_predictions = True
train_model = False

model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Test dataset
dataset_test = BowlDataset()
dataset_test.load_bowl(TEST_DIR)
dataset_test.prepare()

if init_with == "checkpoint":
    model.load_weights(CHECKPOINT_PATH, by_name=True)
else:
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)
    model_path = model.find_last()[1]
    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", 
          model_path)
    model.load_weights(model_path, 
                       by_name=True)

def generate_RCNN_prediction(model = None,                              
                             SUBMISSION_FILE = None,                              
                             TEST_DIR = None,                              
                             OUTPUT_FILE = None):
    
    output = []
    sample_submission = pd.read_csv(SUBMISSION_FILE)
    
    ImageId = []
    EncodedPixels = []
    
    for image_id in tqdm(sample_submission.ImageId):
        image_path = os.path.join(TEST_DIR, 
                                  image_id, "images", 
                                  image_id + ".png")
    
        original_image = cv2.imread(image_path)
        results = model.detect([original_image], 
                               verbose=0)
        r = results[0]
    
        masks = r["masks"]
        
        ImageId_batch, EncodedPixels_batch = f.numpy2encoding_no_overlap2(masks, 
                                                                          image_id, 
                                                                          r["scores"])
        ImageId += ImageId_batch
        EncodedPixels += EncodedPixels_batch
    
    f.write2csv(OUTPUT_FILE, 
                ImageId, 
                EncodedPixels)

def generate_timestamp():
    timestring = time.strftime("%Y_%m_%d-%H_%M_%S")
    print ("Time stamp generated: " + timestring)
    return timestring


timestr = generate_timestamp()

generate_RCNN_prediction(model = model,
                         SUBMISSION_FILE = SUBMISSION_FILE,
                         TEST_DIR = TEST_DIR,
                         OUTPUT_FILE = os.path.join(FILE_PREFIX + timestr + ".csv"))