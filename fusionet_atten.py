# -*- coding: utf-8 -*-
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
from sklearn.metrics import precision_score, recall_score, precision_recall_curve ,auc
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
EPOCHS = 100 # 훈련 epoch 지정
BATCH_SIZE = 2 # batch size 지정
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
EARLY_STOP_PATIENCE = 10

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
            'dice_coef': logs['dice_coef'],
            'val_dice_coef': logs['val_dice_coef'],
            'pixel_accuracy': logs['pixel_accuracy'],
            'val_pixel_accuracy': logs['val_pixel_accuracy'],
            'miou': logs['miou'],
            'val_miou': logs['val_miou'],
            'precision': logs['precision'],
            'recall': logs['recall'],
            'val_precision': logs['val_precision'],
            'val_recall': logs['val_recall'],
            'mAP': logs['mAP'],
            'val_mAP': logs['val_mAP']
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


def add(tensor_a,tensor_b):
    return Add()([tensor_a,tensor_b])

def multiply(tensor_a,tensor_b):
    return Multiply()([tensor_a,tensor_b])

def concat(tensor_a,tensor_b):
    return Concatenate(axis=3)([tensor_a,tensor_b])

def ConvBlock(inputs,n_filters,kernel_size=3,relu_=True):
    net = Conv2D(n_filters, kernel_size=kernel_size, activation='relu', strides=(1,1), padding='same')(inputs)
    return net

def dilate(inputs,n_filters,dilation):
    net = Conv2D(n_filters,kernel_size=3,strides=1,activation='elu',dilation_rate=(dilation,dilation),padding='same')(inputs)
    return net

def Dblock(net,n_filters=512):
    dil_2 = dilate(net,n_filters,2)
    dil_3 = dilate(dil_2, n_filters, 4)
    dil_4 = dilate(dil_3, n_filters, 8)
    dil_5 = dilate(dil_4, n_filters, 16)
    dil_6 = dilate(dil_5, n_filters, 32)
    
    net = add(dil_2,dil_3)
    net = add(net,dil_4)
    net = add(net,dil_5)
    net = add(net,dil_6)
    #net = dil_2 + dil_3 + dil_4 + dil_5 + dil_6
    return net

def Upsample(inputs,rate=2):
    return UpSampling2D(size=rate)(inputs)

def UpBlock(inputs,n_filters):
    net = UpSampling2D(2)(inputs)
    net = Conv2D(n_filters,kernel_size=2,activation='relu',strides=1,padding='same')(net)
    return net

def RRBlock(inputs,n_filters):
    net_1x1 = Conv2D(n_filters,kernel_size=1,activation='relu',strides=1,padding='same')(inputs)
    net = RerBlock(net_1x1,n_filters)
    net = RerBlock(net,n_filters)
    return add(net,net_1x1)

def shortcut(inputs, res, n_filters, equal=True):
    if not equal:
        net = Conv2D(n_filters, kernel_size=1,activation = 'relu',strides=1,padding ='same')(inputs)
        net = add(res, net)
    else:
        net = add(inputs, res)
    return net

def RerBlock(inputs, n_filters, n=2):

    net = ConvBlock(inputs, n_filters)
    net = ConvBlock(net, n_filters)
    net_1 = shortcut(inputs, net, n_filters, equal=False)

    net = ConvBlock(net_1, n_filters)
    net = ConvBlock(net, n_filters)
    net_2 = shortcut(net_1, net, n_filters)

    net = ConvBlock(net_2, n_filters)
    net = ConvBlock(net, n_filters)
    net_3 = shortcut(net_2, net, n_filters)

    return net_3

def attention(tensor,att_tensor,n_filters=512,kernel_size=1):
    g1 = Conv2D(n_filters, kernel_size=kernel_size, activation='relu', strides=1, padding='same')(tensor)
    x1 = Conv2D(n_filters, kernel_size=kernel_size, activation='relu', strides=1, padding='same')(att_tensor)
    net = add(g1,x1)
    #net = relu(net)
    net = Conv2D(1, kernel_size=kernel_size, activation='sigmoid', strides=1, padding='same')(net)
    net = multiply(net,att_tensor)
    return net

def build_model(input_size,  keep_prob=1.0, one_hot_label=False):
    input_layer = Input(input_size)
    n_filters = 64
  
    net = RRBlock(input_layer,n_filters)
    skip1 = net
    net = MaxPooling2D(pool_size=2,strides=2,padding='valid')(net)
    
    net = RRBlock(net,n_filters*2)
    skip2 = net
    net = MaxPooling2D(pool_size=(2, 2),strides=2)(net)
    
    net = RRBlock(net,n_filters*4)
    skip3 = net
    net = MaxPooling2D(pool_size=(2, 2),strides=2)(net)
    
    net = RRBlock(net,n_filters*8)
    skip4 = net
    net = MaxPooling2D(pool_size=(2, 2),strides=2)(net)
    
    net = Dblock(net,n_filters*16)
    D = net
    up_4 = UpBlock(net, 512)
   
    net = attention(up_4, skip4, 8 * n_filters)
    net = concat(net, up_4)
    net = RRBlock(net, n_filters * 8)

    up_3 = UpBlock(net, n_filters * 4)
    net = attention(up_3, skip3, 4 * n_filters)
    net = concat(net, up_3)
    net = RRBlock(net, n_filters * 4)

    up_2 = UpBlock(net, n_filters * 2)
    net = attention(up_2, skip2, 2 * n_filters)
    net = concat(net, up_2)
    net = RRBlock(net, n_filters * 2)

    up_1 = UpBlock(net, n_filters)
    net = attention(up_1, skip1, n_filters)
    net = concat(net, up_1)
    net = RRBlock(net, n_filters)

    #net = Upsample(nei, rate=2)
    net = Conv2D(1, kernel_size=1, activation='sigmoid',strides=1, padding='same')(net)
    model = Model(inputs=input_layer,outputs=net)
    #print(model)
    return model


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
    l_clahe = np.clip(l_clahe * 0.9, 0, 255).astype(l.dtype)
    
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

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

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
model = build_model(input_size=(IMAGE_SIZE[0],IMAGE_SIZE[1] ,N_CHANNELS))
model.compile(optimizer = Adam(learning_rate=0.001), loss = 'binary_crossentropy', metrics = ['accuracy', miou])
model.summary()


# checkpoint 및 조기종료 설정
es = EarlyStopping(monitor='val_miou', mode='max', verbose=1, patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='val_miou', verbose=1,
save_best_only=True, mode='max', period=CHECKPOINT_PERIOD)
rlr = ReduceLROnPlateau(monitor='val_loss',
                        patience=8, #early stopping 의 절반
                        mode = 'auto',
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
    initial_epoch=INITIAL_EPOCH
)
print('---model 훈련 종료---')


print('가중치 저장')
model_weights_output = os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT)
model.save_weights(model_weights_output)
print("저장된 가중치 명: {}".format(model_weights_output))
