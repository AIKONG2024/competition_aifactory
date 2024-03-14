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
from keras.applications import EfficientNetB0,EfficientNetB2, EfficientNetB7
from tensorflow.python.keras import backend as K
import sys
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
from keras.callbacks import Callback, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, precision_recall_curve ,auc
# import tensorflow_hub as hub
import cv2
import segmentation_models as sm

#랜럼시드 고정
RANDOM_STATE = 42 # seed 고정
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값

N_FILTERS = 16 # 필터수 지정
N_CHANNELS = 3 # channel 지정
EPOCHS = 300 # 훈련 epoch 지정
BATCH_SIZE = 32 # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'unet' # 모델 이름
INITIAL_EPOCH = 0 # 초기 epoch
THESHOLDS = 0.25

# 프로젝트 이름
import time
timestr = time.strftime("%Y%m%d%H%M%S")
save_name = timestr

# 데이터 위치
IMAGES_PATH = 'datasets/train_img/'
MASKS_PATH = 'datasets/train_mask/'

# 가중치 저장 위치
OUTPUT_DIR = f'datasets/train_output/{save_name}/'
WORKERS = 24

# 조기종료
EARLY_STOP_PATIENCE = 15

# 중간 가중치 저장 이름
CHECKPOINT_PERIOD = 1
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}.hdf5'.format(MODEL_NAME, save_name)

# 최종 가중치 저장 이름
FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_final_weights.h5'.format(MODEL_NAME, save_name)

# 사용할 GPU 이름
CUDA_DEVICE = 0

#comet_ml 연동
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
comet_api_key = config['comet']['api_key']

from comet_ml import Experiment
experiment = Experiment(
    api_key=comet_api_key,
    project_name='ai-factory-fire',
)

experiment.set_name(f'{MODEL_NAME}_{save_name}')
class CometLogger(Callback):
    def on_epoch_end(self, epoch, logs={}):
        experiment.log_metrics({
            'epoch': epoch,
            'loss': logs['loss'],
            'val_loss': logs['val_loss'],
            'accuracy': logs['accuracy'],
            'val_accuracy': logs['val_accuracy'],
            'miou': logs['miou'],
            'val_miou': logs['val_miou']
        })

class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g

############################################################이미지 전처리#########################################################
def get_img_arr(path, bands):
    if len(bands) > 0 :
        img = rasterio.open(path).read(bands).transpose((1, 2, 0))
    else:
        img = rasterio.open(path).read().transpose((1, 2, 0))
    img = np.float32(img) / MAX_PIXEL_VALUE
    return img

def get_mask_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    seg = np.float32(img)
    return seg

# Data Augmentation 설정
def get_image_data_gen():
    #데이터가 이미지 끝부분에 걸쳐있는 경우가 많아 세밀한 조정 필요
    data_gen_args = dict(
        horizontal_flip=True,
        vertical_flip=True,
    )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    return image_datagen, mask_datagen

#색채 대비
def enhance_image_contrast(image):
    # CLAHE 객체 생성
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    
    # 이미지를 LAB 색공간으로 변환
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # L 채널에 CLAHE 적용
    l_clahe = clahe.apply(l)
    
    # 밝기조절 - 어둡게
    l_clahe = np.clip(l_clahe * 0.3, 0, 255).astype(l.dtype)
    
    # 채널 합치기 및 색공간 변환
    enhanced_lab = cv2.merge((l_clahe, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_image

# @threadsafe_generator
# def generator_from_lists(images_path, masks_path, batch_size=32, shuffle = True, random_state=None):

#     images = []
#     masks = []

#     fopen_image = get_img_arr
#     fopen_mask = get_mask_arr
        
#     i = 0
#     # 데이터 shuffle
#     while True:

#         if shuffle:
#             if random_state is None:
#                 images_path, masks_path = shuffle_lists(images_path, masks_path)
#             else:
#                 images_path, masks_path = shuffle_lists(images_path, masks_path, random_state= random_state + i)
#                 i += 1


#         for img_path, mask_path in zip(images_path, masks_path):

#             img = fopen_image(img_path, bands=(7,6,2))
#             mask = fopen_mask(mask_path)
            
#             # #대비조절
#             # img = np.uint8(img * 255)  # 이미지를 8-bit 정수 타입으로 변환
#             # img = enhance_image_contrast(img)
#             # img = img.astype(np.float32) / 255. #다시 32 float 타입 변환
            
            
#             images.append(img)
#             masks.append(mask)

#             if len(images) >= batch_size:
#                 yield (np.array(images), np.array(masks))
#                 images = []
#                 masks = []
def shuffle_lists(images_path, masks_path, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    combined = list(zip(images_path, masks_path))
    np.random.shuffle(combined)
    shuffled_images_path, shuffled_masks_path = zip(*combined)
    return list(shuffled_images_path), list(shuffled_masks_path)

def rotate_image(image, angle):
    """이미지를 주어진 각도로 회전합니다."""
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def random_crop(image, mask, crop_size):
    """이미지와 마스크를 무작위로 자릅니다."""
    height, width = image.shape[:2]
    x = random.randint(0, max(0, width - crop_size))
    y = random.randint(0, max(0, height - crop_size))
    cropped_image = image[y:y+crop_size, x:x+crop_size]
    cropped_mask = mask[y:y+crop_size, x:x+crop_size]
    return cropped_image, cropped_mask

def adjust_brightness(image, factor=1.2):
    """이미지의 광도를 조절합니다."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # HSV로 변환
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * factor  # V 채널(밝기) 조절
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255  # 최대값을 넘지 않도록 조절
    hsv = np.array(hsv, dtype=np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)  # RGB로 되돌림
    return image

def add_noise(image):
    """이미지에 무작위 노이즈를 추가합니다."""
    mean = 0
    var = 10
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy_image

def augment_image(image, mask):
    """이미지와 마스크에 여러 증강 기법을 적용합니다."""
    # # 50% 확률로 좌우 반전
    # if random.random() > 0.5:
    #     image = np.fliplr(image)
    #     mask = np.fliplr(mask)
    
    # # 50% 확률로 상하 반전
    # if random.random() > 0.5:
    #     image = np.flipud(image)
    #     mask = np.flipud(mask)
    
    # 50% 확률로 이미지 회전
    # if random.random() > 0.5:
    #     angle = random.choice([90, 180, 270])  # 90도 단위로 회전
    #     image = rotate_image(image, angle)
    #     mask = rotate_image(mask, angle)
    
    # # 50% 확률로 광도 조절
    # if random.random() > 0.5:
    #     factor = random.uniform(0.9, 1.1)  # 광도 조절 요소
    #     image = adjust_brightness(image, factor=factor)
    
    # 50% 확률로 노이즈 추가
    # if random.random() > 0.5:
    #     image = add_noise(image)
    
    # 이미지회전
    angle = random.choice([90, 180, 270])  # 90도 단위로 회전
    image = rotate_image(image, angle)
    mask = rotate_image(mask, angle)
    
    # 무작위로 자르기 적용
    crop_size = random.randint(int(image.shape[0] * 0.85), image.shape[0])  # 최소 80% 크기로 자름
    image, mask = random_crop(image, mask, crop_size)
    
    return image, mask

@threadsafe_generator
def generator_from_lists(images_path, masks_path, batch_size=32, shuffle=True, random_state=None):
    print("==데이터 증강 시작==")
    augmented_images_path = []
    augmented_masks_path = []

    # 증강할 이미지의 인덱스 결정 10%
    augment_indices = np.random.choice(len(images_path), size=int(len(images_path) * 0.1), replace=False)
    
    # 증강된 이미지와 원본 이미지를 모두 포함하는 새로운 리스트 생성
    for idx in range(len(images_path)):
        
        img_path = images_path[idx]
        mask_path = masks_path[idx]
        
        img = get_img_arr(img_path, bands=(7,6,2))
        mask = get_mask_arr(mask_path)

        augmented_images_path.append(img)
        augmented_masks_path.append(mask)

        # 증강 인덱스에 해당하는 경우, 증강된 이미지/마스크도 추가
        if idx in augment_indices:
            print(f"증강중...{idx}/ {len(images_path)}")
            img_aug, mask_aug = augment_image(img, mask)
            augmented_images_path.append(img_aug)
            augmented_masks_path.append(mask_aug)

    # 새로운 리스트를 사용하여 배치 생성
    images = []
    masks = []
    total_length = len(augmented_images_path)
    indices = list(range(total_length))
    if shuffle:
        np.random.shuffle(indices)

    for idx in indices:
        images.append(augmented_images_path[idx])
        masks.append(augmented_masks_path[idx])

        if len(images) >= batch_size:
            yield (np.array(images), np.array(masks))
            images = []
            masks = []

    if images and masks:  # 남은 데이터 처리
        yield (np.array(images), np.array(masks))


#############################################모델################################################
def iou_score(y_pred, y_true, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    iou = (intersection + smooth)/(union + smooth)
    return iou

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
def AttnBlock2D(x, g, inter_channel, data_format='channels_first'):

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    f = Activation('relu')(add([theta_x, phi_g]))

    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

    rate = Activation('sigmoid')(psi_f)

    att_x = multiply([x, rate])

    return att_x
# Attention U-Net 
# -*- coding: utf-8 -*-
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, add, multiply
from keras.layers import concatenate, core, Dropout
from keras.models import Model
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers.core import Lambda
import keras.backend as K




def up_and_concate(down_layer, layer, data_format='channels_first'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])

    return concate


def attention_up_and_concate(down_layer, layer, data_format='channels_first'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate


def attention_block_2d(x, g, inter_channel, data_format='channels_first'):
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)

    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    # f(?,g_height,g_width,inter_channel)

    f = Activation('relu')(add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

    rate = Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = multiply([x, rate])

    return att_x


def res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],

              padding='same', data_format='channels_first'):
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    layer = input_layer
    for i in range(2):
        layer = Conv2D(out_n_filters // 4, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)
        if batch_normalization:
            layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(out_n_filters // 4, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer)
        layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer
    out_layer = add([layer, skip_layer])
    return out_layer


# Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],

                  padding='same', data_format='channels_first'):
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer

    layer = skip_layer
    for j in range(2):

        for i in range(2):
            if i == 0:

                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                    layer)
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = Activation('relu')(layer1)
            layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                add([layer1, layer]))
            if batch_normalization:
                layer1 = BatchNormalization()(layer1)
            layer1 = Activation('relu')(layer1)
        layer = layer1

    out_layer = add([layer, skip_layer])
    return out_layer

########################################################################################################
# Define the neural network
def unet(img_w, img_h, n_label, data_format='channels_last'):
    # 수정된 입력 형태
    inputs = Input((img_w, img_h, 3))
    x = inputs
    depth = 4
    features = 32
    skips = []
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)
        features *= 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    for i in reversed(range(depth)):
        features //= 2
        x = UpSampling2D(size=(2, 2), data_format=data_format)(x)
        # 수정된 concatenate 축
        x = concatenate([skips[i], x], axis=3) 
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = core.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)

    return model


########################################################################################################
#Attention U-Net
def att_unet(img_w, img_h, n_label, data_format='channels_first'):
    if data_format == 'channels_last':
        inputs = Input((img_h, img_w, 3))  # channels_last 형식에 맞춤
    else:
        inputs = Input((3, img_w, img_h))  # channels_first 형식
    x = inputs
    depth = 4
    features = 16
    skips = []
    
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)
        features *= 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    for i in reversed(range(depth)):
        features //= 2
        x = UpSampling2D(size=(2, 2), data_format=data_format)(x)
        if data_format == 'channels_first':
            x = concatenate([skips[i], x], axis=1)
        else:
            x = concatenate([skips[i], x], axis=-1)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)

    return model

#######################################################################################################
def simplified_att_unet(img_w, img_h, n_label, data_format='channels_first'):
    if data_format == 'channels_last':
        inputs = Input((img_h, img_w, 3))  # channels_last 형식에 맞춤
    else:
        inputs = Input((3, img_w, img_h))  # channels_first 형식
        
    x = inputs
    depth = 3  # 깊이 감소
    features = 8  # 특성 맵의 수 감소
    skips = []
    
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)
        features *= 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    for i in reversed(range(depth)):
        features //= 2
        x = UpSampling2D(size=(2, 2), data_format=data_format)(x)
        if data_format == 'channels_first':
            x = concatenate([skips[i], x], axis=1)
        else:
            x = concatenate([skips[i], x], axis=-1)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)

    return model
########################################################################################################
#Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
def r2_unet(img_w, img_h, n_label, data_format='channels_first'):
    inputs = Input((3, img_w, img_h))
    x = inputs
    depth = 4
    features = 16
    skips = []
    for i in range(depth):
        x = rec_res_block(x, features, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)

        features = features * 2

    x = rec_res_block(x, features, data_format=data_format)

    for i in reversed(range(depth)):
        features = features // 2
        x = up_and_concate(x, skips[i], data_format=data_format)
        x = rec_res_block(x, features, data_format=data_format)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = core.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    #model.compile(optimizer=Adam(lr=1e-6), loss=[dice_coef_loss], metrics=['accuracy', dice_coef])
    return model


########################################################################################################
#Attention R2U-Net
def att_r2_unet(img_w, img_h, n_label, data_format='channels_last'):
    inputs = Input((img_w, img_h,3))
    x = inputs
    depth = 4
    features = 16
    skips = []
    for i in range(depth):
        x = rec_res_block(x, features, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)

        features = features * 2

    x = rec_res_block(x, features, data_format=data_format)

    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i], data_format=data_format)
        x = rec_res_block(x, features, data_format=data_format)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = core.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    #model.compile(optimizer=Adam(lr=1e-6), loss=[dice_coef_loss], metrics=['accuracy', dice_coef])
    return model


###################################################################################################
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

################################################################################################################
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



def get_unet_small1 (nClasses, input_height=128, input_width=128, n_filters = 16, dropout = 0.1, batchnorm = True, n_channels=3):

    input_img = Input(shape=(input_height,input_width, n_channels))

    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters = n_filters * 2, kernel_size = 3, batchnorm = batchnorm)

    # Expansive Path
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c3)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

    outputs = Conv2D(nClasses, (1, 1), activation='relu')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

def get_unet_small2 (nClasses, input_height=128, input_width=128, n_filters = 16, dropout = 0.1, batchnorm = True, n_channels=3):

    input_img = Input(shape=(input_height,input_width, n_channels))

    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters = n_filters * 4, kernel_size = 3, batchnorm = batchnorm)

    # Expansive Path
    u3 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c2)
    u3 = concatenate([u3, c1])
    u3 = Dropout(dropout)(u3)
    c3 = conv2d_block(u3, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

    outputs = Conv2D(nClasses, (1, 1), activation='relu')(c3)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model



################################### metrics ########################################
# dice score metric
# def dice_coef(y_true, y_pred, smooth=1e-6):
#     intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
#     union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
#     dice = tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)
#     return dice

# 픽셀 정확도 metric
def pixel_accuracy(y_true, y_pred):
    # 임계치 기준으로 이진화
    y_pred = tf.cast(y_pred > THESHOLDS, tf.float32)
    
    # 논리적 AND 연산으로 정확한 예측의 수를 계산
    correct_prediction = tf.logical_and(tf.equal(y_pred, 1), tf.equal(y_true, 1))
    sum_n = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    
    # 실제 True의 총 수
    sum_t = tf.reduce_sum(y_true)
    
    # 조건부로 픽셀 정확도 계산
    pixel_accuracy = tf.cond(sum_t > 0, lambda: sum_n / sum_t, lambda: tf.constant(0.0))
    
    return pixel_accuracy

#miou metric
def miou(y_true, y_pred, smooth=1e-6):
    # 임계치 기준으로 이진화
    y_pred = tf.cast(y_pred > THESHOLDS, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    
    # mIoU 계산
    iou = (intersection + smooth) / (union + smooth)
    miou = tf.reduce_mean(iou)
    return miou

def ohem_loss(y_true, y_pred, n_hard_examples=5):
    """
    Online Hard Example Mining (OHEM) 손실 함수.
    
    y_true: 실제 레이블.
    y_pred: 예측된 확률 또는 레이블.
    n_hard_examples: 고려할 하드 예제의 수.
    """
    # 손실 계산
    losses = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # 손실이 큰 순서로 예제를 선택
    _, indices = tf.nn.top_k(losses, k=n_hard_examples)
    
    # 하드 예제에 대한 손실만 평균하여 반환
    hard_losses = tf.gather(losses, indices)
    return tf.reduce_mean(hard_losses)

def focal_loss(gamma=2., alpha=4.):
    def focal_loss_fixed(y_true, y_pred):
        """
        Focal loss for multi-class or binary classification
        FL(p_t) = -alpha * (1 - p_t) ** gamma * log(p_t)
        """
        # 1e-12를 더해 로그 계산 시 NaN 방지
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Focal loss 계산
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # 배치 내 평균 손실 반환
        return K.mean(K.sum(loss, axis=1))
    return focal_loss_fixed

###################################################################################

#band 이미지와 마스킹 이미지 확인
def show_band_images(image_path, mask_path):
    fig, axs = plt.subplots(3, 4, figsize=(20, 12))
    axs = axs.ravel()
    
    for i in range(10):
        img = rasterio.open(image_path).read(i+1).astype(np.float32) / MAX_PIXEL_VALUE
        axs[i].imshow(img)
        axs[i].set_title(f'Band {i+1}')
        axs[i].axis('off')
    
    img = rasterio.open(mask_path).read(1).astype(np.float32) / MAX_PIXEL_VALUE
    axs[10].imshow(img)
    axs[10].set_title('Mask Image')
    axs[10].axis('off')
    axs[11].axis('off')
    plt.title('Band images compare Mask image')
    plt.tight_layout()
    plt.show() 

#밴드 조합 이미지 확인
def show_bands_image(image_path, band = (0,0,0)):
    img = rasterio.open(image_path).read(band).transpose((1, 2, 0))
    img = np.float32(img)/MAX_PIXEL_VALUE
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')  # 축 표시 없애기
    plt.title(f'Band {band} combine image')
    plt.show()
    return img


###################################################Field################################################

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
try:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
except:
    pass

try:
    np.random.bit_generator = np.random._bit_generator
except:
    pass

train_meta = pd.read_csv('datasets/train_meta.csv')
test_meta = pd.read_csv('datasets/test_meta.csv')

# 저장 폴더 없으면 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# train : val = 8 : 2 나누기
x_tr, x_val = train_test_split(train_meta, test_size=0.20, random_state=RANDOM_STATE)
print(len(x_tr), len(x_val)) #26860 6715

# train : val 지정 및 generator
images_train = [os.path.join(IMAGES_PATH, image) for image in x_tr['train_img'] ]
masks_train = [os.path.join(MASKS_PATH, mask) for mask in x_tr['train_mask'] ]

images_validation = [os.path.join(IMAGES_PATH, image) for image in x_val['train_img'] ]
masks_validation = [os.path.join(MASKS_PATH, mask) for mask in x_val['train_mask'] ]


train_generator = generator_from_lists(images_train, masks_train, batch_size=BATCH_SIZE, random_state=RANDOM_STATE)
validation_generator = generator_from_lists(images_validation, masks_validation, batch_size=BATCH_SIZE, random_state=RANDOM_STATE)


# model 불러오기
# model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
# model = sm.Unet('vgg16', classes=1, input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS), activation='sigmoid', decoder_block_type='upsampling')
model = get_unet_small1(nClasses=1,  input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
# model.compile(optimizer=Adam(lr=1e-5), loss=[dice_coef_loss], metrics=['accuracy', dice_coef, miou])
model.compile(optimizer=Adam(lr=1e-3), loss= sm.losses.binary_focal_loss, metrics=['accuracy', dice_coef, miou])
# model.compile(optimizer = Adam(learning_rate=5e-5), loss = sm.losses.binary_focal_loss, metrics = ['accuracy', miou])
model.summary()


# checkpoint 및 조기종료 설정
es = EarlyStopping(monitor='val_miou', mode='max', verbose=1, patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='val_miou', verbose=1,
save_best_only=True, mode='max', period=CHECKPOINT_PERIOD)
rlr = ReduceLROnPlateau(monitor='val_miou',
                        patience=7, #early stopping 의 절반
                        mode = 'max',
                        verbose= 1,
                        factor=0.5 #learning rate 를 반으로 줄임.
                        )

print('---model 훈련 시작---')
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(images_train) // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(images_validation) // BATCH_SIZE,
    callbacks=[checkpoint, es, CometLogger(),rlr],
    epochs=EPOCHS,
    workers=WORKERS,
    initial_epoch=INITIAL_EPOCH,
)
print('---model 훈련 종료---')


print('가중치 저장')
model_weights_output = os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT)
model.save_weights(model_weights_output)
print("저장된 가중치 명: {}".format(model_weights_output))
