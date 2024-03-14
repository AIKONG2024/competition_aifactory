# -*- coding: utf-8 -*-
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, \
    add, multiply
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
def unet(img_w, img_h, n_label, data_format='channels_first'):
    inputs = Input((3, img_w, img_h))
    x = inputs
    depth = 4
    features = 64
    skips = []
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format= data_format)(x)
        features = features * 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    for i in reversed(range(depth)):
        features = features // 2
        # attention_up_and_concate(x,[skips[i])
        x = UpSampling2D(size=(2, 2), data_format=data_format)(x)
        x = concatenate([skips[i], x], axis=1)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = core.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)

    #model.compile(optimizer=Adam(lr=1e-5), loss=[focal_loss()], metrics=['accuracy', dice_coef])
    return model


########################################################################################################
#Attention U-Net
def att_unet(img_w, img_h, n_label, data_format='channels_first'):
    inputs = Input((3, img_w, img_h))
    x = inputs
    depth = 4
    features = 64
    skips = []
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format='channels_first')(x)
        features = features * 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i], data_format=data_format)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = core.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)

    #model.compile(optimizer=Adam(lr=1e-5), loss=[focal_loss()], metrics=['accuracy', dice_coef])
    return model


########################################################################################################
#Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
def r2_unet(img_w, img_h, n_label, data_format='channels_first'):
    inputs = Input((3, img_w, img_h))
    x = inputs
    depth = 4
    features = 64
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
    features = 64
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

from keras.models import *
from keras.layers import *
from keras.activations import *
import tensorflow as tf
from keras import backend as K

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
from sklearn.metrics import precision_score, recall_score, precision_recall_curve ,auc, average_precision_score
# import tensorflow_hub as hub
import cv2
import segmentation_models as sm

#랜럼시드 고정
RANDOM_STATE = 42 # seed 고정
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값

N_FILTERS = 32 # 필터수 지정
N_CHANNELS = 3 # channel 지정
EPOCHS = 50 # 훈련 epoch 지정
BATCH_SIZE = 2 # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'unet' # 모델 이름
INITIAL_EPOCH = 0 # 초기 epoch
THESHOLDS = 0.27

# 프로젝트 이름
import time
timestr = time.strftime("%Y%m%d%H%M%S")
save_name = timestr

# 데이터 위치
IMAGES_PATH = 'datasets/train_img/'
MASKS_PATH = 'datasets/train_mask/'

# 가중치 저장 위치
OUTPUT_DIR = f'datasets/train_output/{save_name}/'
WORKERS = 20

# 조기종료
EARLY_STOP_PATIENCE = 5

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
            # 'dice_coef': logs['dice_coef'],
            # 'val_dice_coef': logs['val_dice_coef'],
            # 'pixel_accuracy': logs['pixel_accuracy'],
            # 'val_pixel_accuracy': logs['val_pixel_accuracy'],
            'miou': logs['miou'],
            'val_miou': logs['val_miou'],
            # 'precision': logs['precision'],
            # 'recall': logs['recall'],
            # 'val_precision': logs['val_precision'],
            # 'val_recall': logs['val_recall'],
            # 'mAP': logs['mAP'],
            # 'val_mAP': logs['val_mAP']
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

import tensorflow as tf

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
# 이미지와 마스크에 동일한 데이터 증강을 적용하기 위한 제너레이터
def image_mask_generator(image_data_gen, mask_data_gen, images_path, masks_path, batch_size):
    # 이미지와 마스크 데이터 제너레이터 생성
    image_generator = image_data_gen.flow_from_directory(
        'data/images',  # 이미지 폴더 경로
        classes=[images_path],
        class_mode=None,
        color_mode='rgb',
        target_size=(256, 256),  # 필요에 따라 조정
        batch_size=batch_size,
        seed=42)
    
    mask_generator = mask_data_gen.flow_from_directory(
        'data/masks',  # 마스크 폴더 경로
        classes=[masks_path],
        class_mode=None,
        color_mode='grayscale',  # 마스크는 보통 그레이스케일
        target_size=(256, 256),  # 필요에 따라 조정
        batch_size=batch_size,
        seed=42)
    
    # 파이썬의 zip을 사용하여 이미지와 마스크 데이터를 동기화
    while True:
        x = image_generator.next()
        y = mask_generator.next()
        yield x, y
        
# 데이터 증강 설정       
data_gen_args = dict(
    horizontal_flip=True,
    vertical_flip=True,
)

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
    l_clahe = np.clip(l_clahe * 1, 0, 255).astype(l.dtype)
    
    # 채널 합치기 및 색공간 변환
    enhanced_lab = cv2.merge((l_clahe, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_image

@threadsafe_generator
def generator_from_lists(images_path, masks_path, batch_size=32, shuffle = True, random_state=None):

    images = []
    masks = []

    fopen_image = get_img_arr
    fopen_mask = get_mask_arr
        
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

            img = fopen_image(img_path, bands=(7,6,8))
            mask = fopen_mask(mask_path)
            
            # #대비조절
            img = np.uint8(img * 255)  # 이미지를 8-bit 정수 타입으로 변환
            img = enhance_image_contrast(img)
            img = img.astype(np.float32) / 255. #다시 32 float 타입 변환
            
            
            images.append(img)
            masks.append(mask)

            if len(images) >= batch_size:
                yield (np.array(images), np.array(masks))
                images = []
                masks = []

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
x_tr, x_val = train_test_split(train_meta, test_size=0.2, random_state=RANDOM_STATE)
print(len(x_tr), len(x_val)) #26860 6715

# train : val 지정 및 generator
images_train = [os.path.join(IMAGES_PATH, image) for image in x_tr['train_img'] ]
masks_train = [os.path.join(MASKS_PATH, mask) for mask in x_tr['train_mask'] ]

images_validation = [os.path.join(IMAGES_PATH, image) for image in x_val['train_img'] ]
masks_validation = [os.path.join(MASKS_PATH, mask) for mask in x_val['train_mask'] ]

train_generator = generator_from_lists(images_train, masks_train, batch_size=BATCH_SIZE, random_state=RANDOM_STATE)
validation_generator = generator_from_lists(images_validation, masks_validation, batch_size=BATCH_SIZE, random_state=RANDOM_STATE)

# model 불러오기
model = att_r2_unet(img_w=256, img_h=256, n_label=1)
model.compile(optimizer = Adam(learning_rate=0.00001), loss = ohem_loss, metrics = ['accuracy', miou])
model.summary()

MODEL_NAME = 'unet' # 모델 이름
WEIGHT_NAME = '20240313115615/model_unet_20240313115615_final_weights.h5'
train_meta = pd.read_csv('datasets/train_meta.csv')
test_meta = pd.read_csv('datasets/test_meta.csv')

model.load_weights(f'datasets/train_output/{WEIGHT_NAME}')
y_pred_dict = {}

# for idx, i in enumerate(test_meta['test_img']):
#     img = get_img_arr(f'datasets/test_img/{i}', (7,6,8)) 
#     img = np.uint8(img * 255) 
#     img = enhance_image_contrast(img)
#     img = img.astype(np.float32) / 255
#     y_pred = model.predict(np.array([img]), batch_size=32)
#     y_pred = np.where(y_pred[0, :, :, 0] > THESHOLDS, 1, 0) # 임계값 처리
#     y_pred = y_pred.astype(np.uint8)
#     y_pred_dict[i] = y_pred
    
 #mAP확인 - train
thresholds = [0.25, 0.3, 0.35, 0.45, 0.5, 0.65, 0.7, 0.75, 0.8]
aps_per_threshold = {threshold: [] for threshold in thresholds}  # 각 임계치별 AP를 저장할 딕셔너리

#임계치마다 500개의 점수 확인
for idx, img_name in enumerate(train_meta['train_img']):
    if idx < 100:
    
        img_path = f'datasets/train_img/{img_name}' 
        mask_path = img_path.replace('train_img', 'train_mask')
        img = get_img_arr(img_path, bands=(7,6,2))
        img = np.uint8(img * 255) 
        img = enhance_image_contrast(img)
        img = img.astype(np.float32) / 255
        img_pred = np.array([img])
        
        # 실제 마스크 로드 및 변환
        true_mask = get_mask_arr(mask_path).flatten()  # 실제 마스크는 이미 0과 1로 이루어져 있다고 가정

        for threshold in thresholds:
            y_pred = model.predict(img_pred, batch_size=1)
            y_pred_thresh = np.where(y_pred[0, :, :, 0] > threshold, 1, 0).flatten()
            y_pred_thresh = y_pred_thresh.astype(np.uint8)
            
            # 각 임계치에서 AP 계산
            iou = miou(true_mask, y_pred_thresh)
            aps_per_threshold[threshold].append(iou)

# 각 임계치별로 AP의 평균을 계산하고 출력
for threshold, aps in aps_per_threshold.items():
    avg_ap = np.mean(aps)
    print(f"Threhold {threshold} AP] {avg_ap}")

# 모든 임계치에 대한 AP의 평균을 계산하여 mAP를 도출
map = np.mean([np.mean(aps) for aps in aps_per_threshold.values()])
print("[mAP]", map)   
#임계치마다 비교 확인 - test
# thresholds = [0.25, 0.5, 0.75]  # 비교할 임계치 값들

# for idx, img_name in enumerate(test_meta['test_img']):
#     if idx == 30:
#         break
#     img_path = f'datasets/test_img/{img_name}'
#     img = get_img_arr(img_path, bands=(7,6,8))
#     img = np.uint8(img * 255)  # 이미지를 8-bit 정수 타입으로 변환
#     img = enhance_image_contrast(img)
#     img = img.astype(np.float32) / 255  # 다시 32 float 타입으로 변환
#     img_pred = np.array([img])
    
#     fig, axs = plt.subplots(1, len(thresholds) + 1, figsize=(20, 5))  # 원본 이미지 + 임계치별 예측 이미지
#     axs[0].imshow(img)
#     axs[0].set_title('Original Image')
#     axs[0].axis('off')
    
#     for i, threshold in enumerate(thresholds):
#         y_pred = model.predict(img_pred, batch_size=1)
#         y_pred_thresh = np.where(y_pred[0, :, :, 0] > threshold, 1, 0)
#         y_pred_thresh = y_pred_thresh.astype(np.uint8)
        
#         axs[i+1].imshow(y_pred_thresh)
#         axs[i+1].set_title(f'Threshold: {threshold}')
#         axs[i+1].axis('off')
# name = WEIGHT_NAME.split('/')[1]
# joblib.dump(y_pred_dict, f'predict/{name}_y_pred.pkl')
# print("저장된 pkl:", f'predict/{name}_y_pred.pkl')

