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
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.applications import EfficientNetB0
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
import segmentation_models as sm
import cv2

"""&nbsp;

## 사용할 함수 정의
"""
#랜럼시드 고정
RANDOM_STATE = 4321 # seed 고정
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값

class threadsafe_iter:
    """
    데이터 불러올떼, 호출 직렬화
    """
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

def get_img_762bands(path):
    img = rasterio.open(path).read((7,6,2)).transpose((1, 2, 0))
    img = np.float32(img)/MAX_PIXEL_VALUE

    return img

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

def shuffle_lists(images_path, masks_path, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    combined = list(zip(images_path, masks_path))
    np.random.shuffle(combined)
    shuffled_images_path, shuffled_masks_path = zip(*combined)
    return list(shuffled_images_path), list(shuffled_masks_path)

def rotate_image(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    if len(image.shape) == 2 or image.shape[2] == 1:
        rotated_image = rotated_image[:, :, np.newaxis]
    return rotated_image

def adjust_brightness(image, factor=1.2):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return image

def add_noise(image):
    mean = 0
    var = 10
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy_image

def augment_image(image, mask, IMAGE_SIZE=(256, 256)):
    per = 0.3
    # 확률적으로 이미지 변환 적용
    if random.random() < per:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    
    if random.random() < per:
        image = np.flipud(image)
        mask = np.flipud(mask)
    
    if random.random() < per:
        angle = random.choice([90, 180, 270])
        image = rotate_image(image, angle)
        mask = rotate_image(mask, angle)
    
    if random.random() < per:
        factor = random.uniform(0.9, 1.1)
        image = adjust_brightness(image, factor=factor)
    
    if random.random() < per:
        image = add_noise(image)
    return image, mask

@threadsafe_generator
def generator_from_lists(images_path, masks_path, batch_size=32, shuffle = True, random_state=None, image_mode='10bands'):

    images = []
    masks = []

    fopen_image = get_img_arr
    fopen_mask = get_mask_arr

    if image_mode == '762':
        fopen_image = get_img_762bands

    i = 0
    # 데이터 shuffle
    while True:

        if shuffle:
            if random_state is None:
                images_path, masks_path = shuffle_lists(images_path, masks_path)
            else:
                images_path, masks_path = shuffle_lists(images_path, masks_path, random_state= random_state + i)
                i += 1


        for img_path, mask_path in zip(images_path, masks_path):

            img = fopen_image(img_path)
            mask = fopen_mask(mask_path)
            images.append(img)
            masks.append(mask)

            if len(images) >= batch_size:
                yield (np.array(images), np.array(masks))
                images = []
                masks = []

# Unet 모델 정의
def FCN(nClasses, input_height=128, input_width=128, n_filters = 16, dropout = 0.1, batchnorm = True, n_channels=10 ):


    img_input = Input(shape=(input_height,input_width, n_channels))

    ## Block 1
    x = Conv2D(n_filters, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(n_filters, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    f1 = x

    # Block 2
    x = Conv2D(n_filters, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(n_filters, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    f2 = x

    # Out
    o = (Conv2D(nClasses, (3,3), activation='relu' , padding='same', name="Out"))(x)

    model = Model(img_input, o)

    return model


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

from keras.applications import ResNet50
def get_pretrained_unet(nClasses, input_height=256, input_width=256, n_filters=16, dropout=0.1, batchnorm=True, n_channels=10):
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(input_height, input_width, 3)))

    # 사전 학습된 모델의 중간 계층을 추출 (예시: ResNet50의 경우)
    # 실제 계층 이름은 모델마다 다를 수 있으므로 확인이 필요합니다.
    c1 = base_model.get_layer('conv1_relu').output  # 예시 이름, 실제 사용할 계층에 맞게 수정 필요
    c2 = base_model.get_layer('conv2_block3_out').output
    c3 = base_model.get_layer('conv3_block4_out').output
    c4 = base_model.get_layer('conv4_block6_out').output
    c5 = base_model.output  # 인코더의 마지막 계층

    # 확장 경로 시작
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
    model = Model(inputs=base_model.input, outputs=[outputs])

    return model
    
#Attention Gate
def attention_gate(F_g, F_l, inter_channel):
    """Attention Gate with correct up-sampling for dimension matching."""
    W_g = Conv2D(inter_channel, kernel_size=1, padding='same', kernel_initializer='he_normal')(F_g)
    W_x = Conv2D(inter_channel, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(F_l)
    
    # 여기서 W_x의 차원을 W_g와 맞추기 위해 UpSampling2D를 적절히 사용합니다.
    # W_g의 차원과 W_x의 원래 차원을 기반으로 적절한 업샘플링 비율을 결정합니다.
    # 예시에서는 (32, 32, 64)으로 만들어야 하므로, W_x를 2배 업샘플링합니다.
    W_x_upsampled = UpSampling2D(size=(2, 2))(W_x)  # 이제 W_x_upsampled의 차원이 (32, 32, 64)가 됩니다.

    psi = Activation('relu')(Add()([W_g, W_x_upsampled]))
    psi = Conv2D(1, kernel_size=1, padding='same', kernel_initializer='he_normal')(psi)
    psi = Activation('sigmoid')(psi)

    attended = Multiply()([F_l, psi])
    return attended

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it."""
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding='same', kernel_initializer='he_normal')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding='same', kernel_initializer='he_normal')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def get_pretrained_attention_unet(nClasses, input_height=256, input_width=256, n_filters=16, dropout=0.1, batchnorm=True, n_channels=3):
    inputs = Input(shape=(input_height, input_width, n_channels))
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    # Encoder - getting the output of intermediate layers
    c1 = base_model.get_layer('conv1_relu').output
    c2 = base_model.get_layer('conv2_block3_out').output
    c3 = base_model.get_layer('conv3_block4_out').output
    c4 = base_model.get_layer('conv4_block6_out').output
    c5 = base_model.output

    # Decoder
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = Dropout(dropout)(u6)
    a6 = attention_gate(F_g=u6, F_l=c4, inter_channel=n_filters * 4)  # inter_channel 값을 수정
    u6 = concatenate([u6, a6])
    c6 = conv2d_block(u6, n_filters * 8, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = Dropout(dropout)(u7)
    a7 = attention_gate(F_g=u7, F_l=c3, inter_channel=n_filters * 2)  # 여기에서 F_g를 u7로 수정
    u7 = concatenate([u7, a7])
    c7 = conv2d_block(u7, n_filters * 4, batchnorm=batchnorm)  # n_filters * 4로 수정

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = Dropout(dropout)(u8)
    a8 = attention_gate(F_g=u8, F_l=c2, inter_channel=n_filters)  # 여기에서 F_g를 u8로 수정
    u8 = concatenate([u8, a8])
    c8 = conv2d_block(u8, n_filters * 2, batchnorm=batchnorm)  # n_filters * 2로 수정

    u9 = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = Dropout(dropout)(u9)
    a9 = attention_gate(F_g=u9, F_l=c1, inter_channel=n_filters // 2)  # 여기에서 F_g를 u9로 수정
    u9 = concatenate([u9, a9], axis=3)
    c9 = conv2d_block(u9, n_filters, batchnorm=batchnorm)  # n_filters로 수정

    outputs = Conv2D(nClasses, (1, 1), activation='sigmoid')(c9)  # 이전에 c6가 아니라 c9를 사용해야 합니다.

    model = Model(inputs=[inputs], outputs=[outputs])
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


def get_efficientunet_b0(nClasses, input_height=256, input_width=256, n_filters = 16, dropout = 0.1, batchnorm = True, n_channels=10):
    efficient_net = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(input_height, input_width, n_channels), classes=nClasses)

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
    if model_name == 'fcn':
        model = FCN
    elif model_name == 'unet':
        model = get_unet
    elif model_name == 'unet_small':
        model = get_unet_small1
    elif model_name == 'unet_smaller':
        model = get_unet_small2
    elif model_name == 'eb0':
        model = get_efficientunet_b0
    elif model_name == 'pre_unet':
        model = get_pretrained_unet
    elif model_name == 'pre_attention_unet':
        model = get_pretrained_attention_unet
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

"""&nbsp;

## parameter 설정
"""

#miou metric
def miou(y_true, y_pred, smooth=1e-6):
    # 임계치 기준으로 이진화
    y_pred = tf.cast(y_pred > 0.25, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    
    # mIoU 계산
    iou = (intersection + smooth) / (union + smooth)
    miou = tf.reduce_mean(iou)
    return miou


def ohem_loss(y_true, y_pred, n_hard_examples=20):
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


# 사용할 데이터의 meta정보 가져오기

train_meta = pd.read_csv('datasets/train_meta.csv')
test_meta = pd.read_csv('datasets/test_meta.csv')


# 저장 이름
save_name = 'pre_unet'

N_FILTERS = 16 # 필터수 지정
N_CHANNELS = 3 # channel 지정
EPOCHS = 300 # 훈련 epoch 지정
BATCH_SIZE = 32 # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'pre_attention_unet' # 모델 이름
INITIAL_EPOCH = 0 # 초기 epoch

# 데이터 위치
IMAGES_PATH = 'datasets/train_img/'
MASKS_PATH = 'datasets/train_mask/'

# 가중치 저장 위치
OUTPUT_DIR = 'datasets/train_output/'
WORKERS = 40

# 조기종료
EARLY_STOP_PATIENCE = 15

# 중간 가중치 저장 이름
CHECKPOINT_PERIOD = 5
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}.hdf5'.format(MODEL_NAME, save_name)

# 최종 가중치 저장 이름
FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_final_weights.h5'.format(MODEL_NAME, save_name)

# 사용할 GPU 이름
CUDA_DEVICE = 0


# 저장 폴더 없으면 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


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


# train : val = 8 : 2 나누기
x_tr, x_val = train_test_split(train_meta, test_size=0.2, random_state=RANDOM_STATE)
print(len(x_tr), len(x_val))

# train : val 지정 및 generator
images_train = [os.path.join(IMAGES_PATH, image) for image in x_tr['train_img'] ]
masks_train = [os.path.join(MASKS_PATH, mask) for mask in x_tr['train_mask'] ]

images_validation = [os.path.join(IMAGES_PATH, image) for image in x_val['train_img'] ]
masks_validation = [os.path.join(MASKS_PATH, mask) for mask in x_val['train_mask'] ]

train_generator = generator_from_lists(images_train, masks_train, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")
validation_generator = generator_from_lists(images_validation, masks_validation, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")


# model 불러오기
model = get_model(MODEL_NAME,nClasses=1, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
model.compile(optimizer = Adam(learning_rate=0.001), loss = ohem_loss, metrics = ['accuracy', miou])
model.summary()


# checkpoint 및 조기종료 설정
es = EarlyStopping(monitor='val_miou', mode='max', verbose=1, patience=EARLY_STOP_PATIENCE,  restore_best_weights=True)
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='val_miou', verbose=1,
save_best_only=True, mode='max', period=CHECKPOINT_PERIOD)
rlr = ReduceLROnPlateau(monitor='val_miou',
                        patience=7, #early stopping 의 절반
                        mode = 'max',
                        verbose= 1,
                        factor=0.5 #learning rate 를 반으로 줄임.
                        )
"""&nbsp;

## model 훈련
"""

print('---model 훈련 시작---')
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(images_train) // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(images_validation) // BATCH_SIZE,
    callbacks=[checkpoint, es, rlr],
    epochs=EPOCHS,
    workers=WORKERS,
    initial_epoch=INITIAL_EPOCH
)
print('---model 훈련 종료---')

"""&nbsp;

## model save
"""

print('가중치 저장')
model_weights_output = os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT)
model.save_weights(model_weights_output)
print("저장된 가중치 명: {}".format(model_weights_output))


y_pred_dict = {}

for i in test_meta['test_img']:
    img = get_img_762bands(f'datasets/test_img/{i}')
    y_pred = model.predict(np.array([img]), batch_size=1)

    y_pred = np.where(y_pred[0, :, :, 0] > 0.5, 1, 0) # 임계값 처리
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred

# joblib.dump(y_pred_dict, 'predict/y_pred.pkl')
