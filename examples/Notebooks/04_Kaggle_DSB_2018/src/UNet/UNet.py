import os
import sys
import random
import warnings
import types
import time
import gc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

try:
    import warnings
    warnings.filterwarnings('ignore')
    from keras.models import Model, load_model
    from keras.layers import Input
    from keras.layers.core import Dropout, Lambda
    from keras.layers.convolutional import Conv2D, Conv2DTranspose
    from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
    from keras.layers.merge import concatenate
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.optimizers import SGD, RMSprop, Adagrad, Adam
    from keras import backend as K
    from keras.metrics import binary_crossentropy
    from keras.models import model_from_json
except:
    print ("Install Keras 2 (cmd: $sudo pip3 install keras) to run this notebook.")

import tensorflow as tf

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

DEFAULT_UNIT_SIZE = 128
DEFAULT_DROPOUT = 0.55

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 1024
random.seed = seed
np.random.seed = seed

args = types.SimpleNamespace()
args.data_path = ['./data/']
args.config_file = ['./model/trained_2018_03_21-13_26_55_config_UNet.json']
args.weights_file = ['./model/trained_2018_03_21-13_26_55_weights_UNet.model']
args.output_dir = ['./model/']

checkpointer_savepath = os.path.join(args.output_dir[0]     +       
                                     'checkpoint/UNet_I' +       
                                     str(IMG_WIDTH)  + '_'  + 
                                     str(IMG_HEIGHT) + '_'  +  
                                     'U' + str(DEFAULT_UNIT_SIZE)+ 
                                     '.h5')

TEST_PATH = os.path.join(args.data_path[0]+'/test/')
print (TEST_PATH)

TRAIN_PATH = os.path.join(args.data_path[0]+'/train_aug/')
print (TRAIN_PATH)

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

use_pre_proc_images = True
merge_pre_proc_images = True

if use_pre_proc_images == True and merge_pre_proc_images == False:
    train_data = np.load(os.path.join(args.data_path[0]+'/train_pre_proc_256.npz'))
elif use_pre_proc_images == False and merge_pre_proc_images == False:
    train_data = np.load(os.path.join(args.data_path[0]+'/train_aug_256.npz'))
elif use_pre_proc_images == True and merge_pre_proc_images == True:
    train_data_pre_proc = np.load(os.path.join(args.data_path[0]+'/train_aug_256.npz'))
    train_data_aug = np.load(os.path.join(args.data_path[0]+'/train_pre_proc_256.npz'))
else:
    train_data = np.load(os.path.join(args.data_path[0]+'/train_aug_256.npz'))

if merge_pre_proc_images == True:
    X_train_aug = train_data_aug['xtrain']
    Y_train_aug = train_data_aug['ytrain']
    
    X_train_pre_proc = train_data_pre_proc['xtrain']
    Y_train_pre_proc = train_data_pre_proc['ytrain']
    
    X_train = np.concatenate((X_train_aug , X_train_pre_proc), axis =0)
    Y_train = np.concatenate((Y_train_aug , Y_train_pre_proc), axis =0)
    
    del train_data_aug
    del train_data_pre_proc
    del X_train_aug
    del Y_train_aug
    del X_train_pre_proc
    del Y_train_pre_proc
    gc.collect()
else:    
    X_train = train_data['xtrain']
    Y_train = train_data['ytrain']
    del train_data
    gc.collect()
    
if len(X_train) != len(Y_train):
    print ("Mismatched images and prediction masks for training data ...")
    sys.exit(1)
    
split_data = False
split_factor = 4

if split_data == True:
    sample_size = len(X_train)
    split_size = sample_size//split_factor
    n =(randint(0, split_size))
    try:
        X_train = X_train[n:((sample_size - split_size) + n)]
        Y_train = Y_train[n:((sample_size - split_size) + n)]
    except:
        print ("Failed to split training data ...")
        X_train = X_train
        Y_train = Y_train
else:
    X_train = X_train
    Y_train = Y_train


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

NUM_CLASSES = 2

def mean_iou_tf(y_true, y_pred):
   score, up_opt = tf.metrics.mean_iou(y_true, y_pred, NUM_CLASSES)
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
   return score

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)-K.log(dice_coef(y_true, y_pred))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

use_dice = True

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
DEFAULT_ACTIVATION = 'elu' # 'relu', 'elu'
unit_size = DEFAULT_UNIT_SIZE
dropout = DEFAULT_DROPOUT
final_max_pooling = True

def build_UNet(unit_size = None,
                 final_max_pooling = None):
    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(unit_size, (3, 3), 
                activation=DEFAULT_ACTIVATION, 
                kernel_initializer='he_normal', 
                padding='same') (s)
    c1 = Dropout(dropout) (c1)
    c1 = Conv2D(unit_size, (3, 3), 
                activation=DEFAULT_ACTIVATION, 
                kernel_initializer='he_normal', 
                padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(unit_size*2, (3, 3), 
                activation=DEFAULT_ACTIVATION, 
                kernel_initializer='he_normal', 
                padding='same') (p1)
    c2 = Dropout(dropout) (c2)
    c2 = Conv2D(unit_size*2, (3, 3), 
                activation=DEFAULT_ACTIVATION, 
                kernel_initializer='he_normal', 
                padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(unit_size*4, (3, 3), 
                activation=DEFAULT_ACTIVATION, 
                kernel_initializer='he_normal', 
                padding='same') (p2)
    c3 = Dropout(dropout) (c3)
    c3 = Conv2D(unit_size*4, (3, 3), 
                activation=DEFAULT_ACTIVATION, 
                kernel_initializer='he_normal', 
                padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(unit_size*8, (3, 3), 
                activation=DEFAULT_ACTIVATION, 
                kernel_initializer='he_normal', 
                padding='same') (p3)
    c4 = Dropout(dropout) (c4)
    c4 = Conv2D(unit_size*8, (3, 3), 
                activation=DEFAULT_ACTIVATION, 
                kernel_initializer='he_normal', 
                padding='same') (c4)
    p4 = MaxPooling2D((2, 2)) (c4)

    c5 = Conv2D(unit_size*16, (3, 3), 
                activation=DEFAULT_ACTIVATION, 
                kernel_initializer='he_normal', 
                padding='same') (p4)
    c5 = Dropout(dropout) (c5)
    c5 = Conv2D(unit_size*16, (3, 3), 
                activation=DEFAULT_ACTIVATION, 
                kernel_initializer='he_normal', 
                padding='same') (c5)
    c5 = Dropout(dropout) (c5)
    c5 = Conv2D(unit_size*16, (3, 3), 
                activation=DEFAULT_ACTIVATION, 
                kernel_initializer='he_normal', 
                padding='same') (c5)

    u6 = Conv2DTranspose(unit_size*8, (2, 2), 
                         strides=(2, 2), 
                         padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(unit_size*8, (3, 3), 
                activation=DEFAULT_ACTIVATION, 
                kernel_initializer='he_normal', 
                padding='same') (u6)
    c6 = Dropout(dropout) (c6)
    c6 = Conv2D(unit_size*8, (3, 3), 
                activation=DEFAULT_ACTIVATION, 
                kernel_initializer='he_normal', 
                padding='same') (c6)

    u7 = Conv2DTranspose(unit_size*4, (2, 2), 
                         strides=(2, 2), 
                         padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(unit_size*4, (3, 3), 
                activation=DEFAULT_ACTIVATION, 
                kernel_initializer='he_normal', 
                padding='same') (u7)
    c7 = Dropout(dropout) (c7)
    c7 = Conv2D(unit_size*4, (3, 3), 
                activation=DEFAULT_ACTIVATION, 
                kernel_initializer='he_normal', 
                padding='same') (c7)

    u8 = Conv2DTranspose(unit_size*2, (2, 2), 
                         strides=(2, 2), 
                         padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(unit_size*2, (3, 3), 
                activation=DEFAULT_ACTIVATION, 
                kernel_initializer='he_normal', 
                padding='same') (u8)
    c8 = Dropout(dropout) (c8)
    c8 = Conv2D(unit_size*2, (3, 3), 
                activation=DEFAULT_ACTIVATION, 
                kernel_initializer='he_normal', 
                padding='same') (c8)

    u9 = Conv2DTranspose(unit_size, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(unit_size, (3, 3), 
                activation=DEFAULT_ACTIVATION, 
                kernel_initializer='he_normal', 
                padding='same') (u9)
    c9 = Dropout(dropout) (c9)
    c9 = Conv2D(unit_size, (3, 3), 
                activation=DEFAULT_ACTIVATION, 
                kernel_initializer='he_normal', 
                padding='same') (c9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

model = build_UNet(unit_size = unit_size,
              final_max_pooling = final_max_pooling)

def load_prediction_model(args):
    try:
        print (args.config_file[0]) 
        with open(args.config_file[0]) as json_file:
              model_json = json_file.read()
        model = model_from_json(model_json)
        return model
    except:
        print ("Please specify a model configuration file ...")
        sys.exit(1)

def load_prediction_model_weights(args):
    try:
        model.load_weights(args.weights_file[0])
        print ("Loaded model weights from: " 
               + str(args.weights_file[0]))
        return model        
    except:
        print ("Error loading model weights ...")
        sys.exit(1)

load_from_checkpoint = True
load_from_config = False
load_model_weights = False

def load_saved_model(checkpointer_savepath = None, 
                     args = None,
                     mean_iou = None,
                     mean_iou_tf = None,
                     dice_coef = None,
                     bce_dice = None,
                     dice_coef_loss = None,
                     load_from_checkpoint = None,
                     load_from_config = None,
                     load_model_weights = None):
    if load_from_checkpoint == True:
        if use_dice == True:
            model = load_model(checkpointer_savepath,                  \
                               custom_objects={'mean_iou': mean_iou,   \
                                               'mean_iou_tf': mean_iou_tf,   \
                                               'dice_coef': dice_coef, \
                                               'bce_dice': bce_dice,   \
                                               'dice_coef_loss': dice_coef_loss})
        else:
            model = load_model(checkpointer_savepath, \
                               custom_objects={'mean_iou': mean_iou})
    elif load_from_config == True:
        model = load_prediction_model(args)
        model = load_prediction_model_weights(args)
    elif load_model_weights == True:
        try:
            model = load_prediction_model_weights(args)
        except:
            print ("An exception has occurred, while loading model weights ...")
    else:
        model = model
    return model

try:
    model = load_saved_model(checkpointer_savepath = checkpointer_savepath, 
                         args = args,
                         mean_iou = mean_iou,
                         mean_iou_tf = mean_iou_tf,
                         dice_coef = dice_coef,
                         bce_dice = bce_dice,
                         dice_coef_loss = dice_coef_loss,
                         load_from_checkpoint = load_from_checkpoint,
                         load_from_config = load_from_config,
                         load_model_weights = load_model_weights)
except:
    model = model

sgd = SGD(lr=1e-7, 
          decay=0.5, 
          momentum=1, 
          nesterov=True)
rms = RMSprop(lr=1e-7, 
              rho=0.9, 
              epsilon=1e-08, 
              decay=0.0)
ada = Adagrad(lr=1e-7, 
              epsilon=1e-08, 
              decay=0.0)
adam = Adam(lr=1e-4, 
            beta_1=0.9, 
            beta_2=0.999, 
            epsilon=None, 
            decay=0.0)
    
DEFAULT_OPTIMIZER = adam

use_dice = True
use_dice_loss = True
use_custom_iou = True
if use_dice == True and use_dice_loss == False:
    model.compile(optimizer = DEFAULT_OPTIMIZER, 
              loss = bce_dice, 
              metrics = ['binary_crossentropy', 
                         dice_coef, 
                         mean_iou,
                         mean_iou_tf])
elif use_dice_loss == True and use_dice == True :
    model.compile(optimizer = DEFAULT_OPTIMIZER, 
                   loss = dice_coef_loss, 
                   metrics = [dice_coef,
                              mean_iou,
                              mean_iou_tf,
                              'acc', 
                              'mse'])
elif use_custom_iou == True:
    model.compile(optimizer = DEFAULT_OPTIMIZER, 
                   loss = 'binary_crossentropy', 
                   metrics = [mean_iou,
                              mean_iou_tf,
                              'acc', 
                              'mse'])
else:
    model.compile(optimizer=DEFAULT_OPTIMIZER, 
                  loss='binary_crossentropy', 
                  metrics=[mean_iou_tf, 
                           'acc', 
                           'mse'])
model_summary = False
if model_summary == True:
    model.summary()

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint(checkpointer_savepath, 
                               verbose=1,  
                               save_best_only=True)
results = model.fit(X_train, 
                    Y_train, 
                    validation_split=0.2, 
                    batch_size=8, 
                    epochs=1, 
                    callbacks=[earlystopper, 
                               checkpointer])

def generate_timestamp():
    timestring = time.strftime("%Y_%m_%d-%H_%M_%S")
    print ("Time stamp generated: "+timestring)
    return timestring

timestr = generate_timestamp()

def save_model(args, name, model):
    file_loc = args.output_dir[0]
    file_pointer = os.path.join(file_loc+"//trained_"+ timestr)
    model.save_weights(os.path.join(file_pointer 
                                    + "_weights"
                                    +str(name)
                                    +".model"))    
    model_json = model.to_json()
    with open(os.path.join(file_pointer
                           +"_config"
                           +str(name)
                           +".json"), "w") as json_file:
        json_file.write(model_json)
    print ("Saved the trained model weights to: " + 
           str(os.path.join(file_pointer 
                            + "_weights"+str(name)
                            + ".model")))
    print ("Saved the trained model configuration as a json file to: " + 
    str(os.path.join(file_pointer
                     + "_config"+str(name)
                     + ".json")))

save_model(args, '_UNet', model)