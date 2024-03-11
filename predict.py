# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings("ignore")
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications import EfficientNetB0, EfficientNetB7
from tensorflow.python.keras import backend as K
import sys
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
import threading
import random
import rasterio
import os
import numpy as np
import sys
from sklearn.utils import shuffle as shuffle_lists
from keras.models import *
from keras.layers import *
import numpy as np
from keras import backend as K
from sklearn.model_selection import train_test_split
import joblib
import time

#랜럼시드 고정
RANDOM_STATE = 42 # seed 고정
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

MODEL_PATH = 'model.h5'
TEST_IMG_DIR = 'path/to/test/images'
PREDICTIONS_SAVE_PATH = 'predictions.pkl'
IMAGE_SIZE = 256

"""## inference

- 학습한 모델 불러오기
"""
# 저장 이름

MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값

N_FILTERS = 16 # 필터수 지정
N_CHANNELS = 3 # channel 지정
EPOCHS = 30 # 훈련 epoch 지정
BATCH_SIZE = 64 # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
INITIAL_EPOCH = 0 # 초기 epoch
THESHOLDS = 0.25

# 데이터 위치
IMAGES_PATH = 'datasets/train_img/'
MASKS_PATH = 'datasets/train_mask/'

# 가중치 저장 위치
OUTPUT_DIR = 'datasets/train_output/'
WORKERS = 10

MODEL_NAME = 'unet' # 모델 이름
WEIGHT_NAME = '20240310035525/model_unet_20240310035525_final_weights.h5'


train_meta = pd.read_csv('datasets/train_meta.csv')
test_meta = pd.read_csv('datasets/test_meta.csv')


def get_img_arr(path, bands):
    if len(bands) > 0 :
        img = rasterio.open(path).read(bands).transpose((1, 2, 0)) 
    else:
        img = rasterio.open(path).read().transpose((1, 2, 0))
    img = np.float32(img)/MAX_PIXEL_VALUE
    return img

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def get_unet(nClasses, input_height=256, input_width=256, n_filters = 16, dropout = 0.1, batchnorm = True, n_channels=10):
    input_img = Input(shape=(input_height,input_width, n_channels))

    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(nClasses, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

def get_efficientunet_b0(nClasses, input_height=256, input_width=256, n_filters = 16, dropout = 0.1, batchnorm = True, n_channels=10):
    efficient_net = EfficientNetB0(weights=None, include_top=False, input_shape=(input_height, input_width, n_channels), classes=nClasses)

    # Prepare feature extraction layers
    s1 = efficient_net.get_layer('block2a_expand_activation').output
    s2 = efficient_net.get_layer('block3a_expand_activation').output
    s3 = efficient_net.get_layer('block4a_expand_activation').output
    s4 = efficient_net.get_layer('block6a_expand_activation').output
    bridge = efficient_net.get_layer('top_activation').output

    # Decoding layers or expansive path
    u1 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same')(bridge)
    u1 = concatenate([u1, s4])
    u1 = Dropout(0.1)(u1)
    c6 = conv2d_block(u1, n_filters=n_filters*8, kernel_size=3, batchnorm=True)

    u2 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same')(c6)
    u2 = concatenate([u2, s3])
    u2 = Dropout(0.1)(u2)
    c7 = conv2d_block(u2, n_filters=n_filters*4, kernel_size=3, batchnorm=True)

    u3 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same')(c7)
    u3 = concatenate([u3, s2])
    u3 = Dropout(0.1)(u3)
    c8 = conv2d_block(u3, n_filters=n_filters*2, kernel_size=3, batchnorm=True)

    u4 = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same')(c8)
    u4 = concatenate([u4, s1], axis=3)
    u4 = Dropout(0.1)(u4)
    c9 = conv2d_block(u4, n_filters=n_filters, kernel_size=3, batchnorm=True)
    
    u_final = UpSampling2D((2, 2), interpolation='bilinear')(c9)
    outputs = Conv2D(nClasses, (1, 1), activation='sigmoid')(u_final)

    model = Model(inputs=efficient_net.input, outputs=outputs)

    return model

def get_efficientunet_b7(nClasses, input_height=256, input_width=256, n_filters = 16, dropout = 0.1, batchnorm = True, n_channels=10):
    efficient_net = EfficientNetB7(weights=None, include_top=False, input_shape=(input_height, input_width, n_channels), classes=nClasses)

    # Prepare feature extraction layers
    s1 = efficient_net.get_layer('block2a_expand_activation').output
    s2 = efficient_net.get_layer('block3a_expand_activation').output
    s3 = efficient_net.get_layer('block4a_expand_activation').output
    s4 = efficient_net.get_layer('block6a_expand_activation').output
    bridge = efficient_net.get_layer('top_activation').output

    # Decoding layers or expansive path
    u1 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same')(bridge)
    u1 = concatenate([u1, s4])
    u1 = Dropout(0.1)(u1)
    c6 = conv2d_block(u1, n_filters=n_filters*8, kernel_size=3, batchnorm=True)

    u2 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same')(c6)
    u2 = concatenate([u2, s3])
    u2 = Dropout(0.1)(u2)
    c7 = conv2d_block(u2, n_filters=n_filters*4, kernel_size=3, batchnorm=True)

    u3 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same')(c7)
    u3 = concatenate([u3, s2])
    u3 = Dropout(0.1)(u3)
    c8 = conv2d_block(u3, n_filters=n_filters*2, kernel_size=3, batchnorm=True)

    u4 = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same')(c8)
    u4 = concatenate([u4, s1], axis=3)
    u4 = Dropout(0.1)(u4)
    c9 = conv2d_block(u4, n_filters=n_filters, kernel_size=3, batchnorm=True)
    
    u_final = UpSampling2D((2, 2), interpolation='bilinear')(c9)
    outputs = Conv2D(nClasses, (1, 1), activation='sigmoid')(u_final)

    model = Model(inputs=efficient_net.input, outputs=outputs)

    return model

def get_model(model_name, nClasses=1, input_height=128, input_width=128, n_filters = 16, dropout = 0.1, batchnorm = True, n_channels=10):
    if model_name == 'eb0':
        model = get_efficientunet_b0
    elif model_name == 'eb7':
        model = get_efficientunet_b7
    elif model_name == 'unet':
        model = get_unet
    return model(
            nClasses      = nClasses,
            input_height  = input_height,
            input_width   = input_width,
            n_filters     = n_filters,
            dropout       = dropout,
            batchnorm     = batchnorm,
            n_channels    = n_channels
        )
    
# 두 샘플 간의 유사성 metric
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

# 픽셀 정확도를 계산 metric
def pixel_accuracy (y_true, y_pred):
    sum_n = np.sum(np.logical_and(y_pred, y_true))
    sum_t = np.sum(y_true)

    if (sum_t == 0):
        pixel_accuracy = 0
    else:
        pixel_accuracy = sum_n / sum_t
    return pixel_accuracy


model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy', dice_coef, pixel_accuracy])
model.summary()

model.load_weights(f'datasets/train_output/{WEIGHT_NAME}')
y_pred_dict = {}

for idx, i in enumerate(test_meta['test_img']):
    if idx == 30 :
        break
    img = get_img_arr(f'datasets/test_img/{i}', (7,6,8)) 
    y_pred = model.predict(np.array([img]), batch_size=32)

    y_pred = np.where(y_pred[0, :, :, 0] > THESHOLDS, 1, 0) # 임계값 처리
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred
    # plt.figure(figsize=(10,10))
    # plt.imshow(y_pred)
    # plt.show()

joblib.dump(y_pred_dict, f'predict/{WEIGHT_NAME.split('/')[1]}_y_pred.pkl')
print("저장된 pkl:", f'predict/{WEIGHT_NAME.split('/')[1]}_y_pred.pkl')