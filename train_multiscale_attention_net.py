# -*- coding: utf-8 -*-
#참조논문 : https://www.mdpi.com/2072-4292/14/4/992

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
from keras import backend as K
import joblib
import time
from keras.callbacks import Callback, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, precision_recall_curve ,auc
from skimage.transform import resize
# import tensorflow_hub as hub
import cv2

#랜럼시드 고정
RANDOM_STATE = 42 # seed 고정
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값

N_FILTERS = 16 # 필터수 지정
N_CHANNELS = 3 # channel 지정
EPOCHS = 300 # 훈련 epoch 지정
BATCH_SIZE = 15 # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'AFD' # 모델 이름
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
WORKERS = 32

# 조기종료
EARLY_STOP_PATIENCE = 30

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
            'iou_score': logs['iou_score'],
            'val_iou_score': logs['val_iou_score'],
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
def get_img_arr(path):
    with rasterio.open(path) as src:
        swir2 = src.read(7).astype(float)
        swir1 = src.read(6).astype(float)
        nir = src.read(5).astype(float)
        blue = src.read(2).astype(float)

    #AFI
    afi = (swir2 / blue).astype(float)
    # 정규화
    swir2 = np.float32(swir2) / MAX_PIXEL_VALUE
    swir1 = np.float32(swir1)/ MAX_PIXEL_VALUE
    blue = np.float32(blue)/ MAX_PIXEL_VALUE
    nir = np.float32(nir) / MAX_PIXEL_VALUE
    
    
    # 배열 생성
    img = np.stack([swir2, swir1, nir], axis=-1)
    
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

def augment_image(image, mask, per=0.4):
    # 확률적으로 이미지 변환 적용
    if random.random() < per:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
        
    if random.random() < per:
        image = np.flipud(image)
        mask = np.flipud(mask)

    return image, mask

@threadsafe_generator
def generator_from_lists(images_path, masks_path, batch_size=32, shuffle = True, random_state=None, is_train = False):

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

            img = fopen_image(img_path)
            mask = fopen_mask(mask_path)
            
            if is_train:
                img, mask = augment_image(img, mask)
                
            images.append(img)
            masks.append(mask)

            if len(images) >= batch_size:
                yield (np.array(images), np.array(masks))
                images = []
                masks = []


#############################################모델################################################

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, dilation_rate=1):
    # 기본 컨볼루션
    c1 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding="same")(input_tensor)
    if batchnorm:
        c1 = BatchNormalization()(c1)
    c1 = Activation("relu")(c1)

    # dilation rate 적용 컨볼루션
    c2 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), dilation_rate=dilation_rate, padding="same")(input_tensor)
    if batchnorm:
        c2 = BatchNormalization()(c2)
    c2 = Activation("relu")(c2)

    return Concatenate()([c1, c2])

def upsample_block(input_tensor, skip_tensor, n_filters, kernel_size=3, batchnorm=True):
    x = Conv2DTranspose(n_filters, kernel_size, strides=(2, 2), padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Concatenate()([x, skip_tensor])
    return x

def conv2d_sets(input_tensor, n_filters, batchnorm, dilation_rate):
    c1 = conv2d_block(input_tensor, n_filters=n_filters, kernel_size=3, batchnorm=batchnorm, dilation_rate=dilation_rate)
    c2 = conv2d_block(input_tensor, n_filters=n_filters, kernel_size=5, batchnorm=batchnorm, dilation_rate=dilation_rate)
    c3 = conv2d_block(input_tensor, n_filters=n_filters, kernel_size=7, batchnorm=batchnorm, dilation_rate=dilation_rate)
    return Concatenate()([c1, c2, c3])

def attention_gate(F_g, F_l, inter_channel):
    W_g = Conv2D(inter_channel, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(F_g)
    W_g = BatchNormalization()(W_g)

    W_x = Conv2D(inter_channel, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(F_l)
    W_x = BatchNormalization()(W_x)

    psi = Activation('relu')(add([W_g, W_x]))
    psi = Conv2D(1, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)

    return multiply([F_l, psi])

def get_attention_AFD(input_height=256, input_width=256, nClasses=1, n_filters=16, dropout=0.5, batchnorm=True, n_channels=1, dilation_rate = 1):
    input_img = Input(shape=(input_height, input_width, n_channels))
    
    # Encoder
    e1 = conv2d_sets(input_img, n_filters*1, batchnorm=batchnorm, dilation_rate=dilation_rate)
    p1 = MaxPooling2D((2, 2))(e1)
    
    e2 = conv2d_sets(p1, n_filters*2, batchnorm=batchnorm, dilation_rate=dilation_rate)
    p2 = MaxPooling2D((2, 2))(e2)
    
    e3 = conv2d_sets(p2, n_filters*4, batchnorm=batchnorm, dilation_rate=dilation_rate)
    p3 = MaxPooling2D((2, 2))(e3)
    
    e4 = conv2d_sets(p3, n_filters*8, batchnorm=batchnorm, dilation_rate=dilation_rate)
    p4 = MaxPooling2D((2, 2))(e4)
    
    e5 = conv2d_sets(p4, n_filters*16, batchnorm=batchnorm, dilation_rate=dilation_rate)
    
    # Decoder
    d1 = upsample_block(e5, attention_gate(e5, e4, n_filters*8), n_filters*8, kernel_size=3, batchnorm=batchnorm)
    d2 = conv2d_sets(d1, n_filters*8, batchnorm=batchnorm, dilation_rate=dilation_rate)
    
    d2 = upsample_block(d2, attention_gate(d2, e3, n_filters*4), n_filters*4, kernel_size=3, batchnorm=batchnorm)
    d3 = conv2d_sets(d2, n_filters*4, batchnorm=batchnorm, dilation_rate=dilation_rate)
    
    d3 = upsample_block(d3, attention_gate(d3, e2, n_filters*2), n_filters*2, kernel_size=3, batchnorm=batchnorm)
    d4 = conv2d_sets(d3, n_filters*2, batchnorm=batchnorm, dilation_rate=dilation_rate)
    
    d4 = upsample_block(d4, attention_gate(d4, e1, n_filters), n_filters, kernel_size=3, batchnorm=batchnorm)
    d5 = conv2d_sets(d4, n_filters, batchnorm=batchnorm, dilation_rate=dilation_rate)

    outputs = Conv2D(nClasses, (1, 1), activation='sigmoid')(d5)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

################################### metrics ########################################
# dice score metric
def dice_coef(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice
        
def ohem_loss(y_true, y_pred, n_hard_examples=20):
    # 손실 계산
    losses = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # 손실이 큰 순서로 예제를 선택
    _, indices = tf.nn.top_k(losses, k=n_hard_examples)
    
    # 하드 예제에 대한 손실만 평균하여 반환
    hard_losses = tf.gather(losses, indices)
    return tf.reduce_mean(hard_losses)        

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
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

train_meta = pd.read_csv('datasets/train_meta.csv')
test_meta = pd.read_csv('datasets/test_meta.csv')

# 저장 폴더 없으면 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

x_tr, x_val = train_test_split(train_meta, test_size=0.3, random_state=RANDOM_STATE)

images_train = [os.path.join(IMAGES_PATH, image) for image in x_tr['train_img'] ]
masks_train = [os.path.join(MASKS_PATH, mask) for mask in x_tr['train_mask'] ]

images_validation = [os.path.join(IMAGES_PATH, image) for image in x_val['train_img'] ]
masks_validation = [os.path.join(MASKS_PATH, mask) for mask in x_val['train_mask'] ]

train_generator = generator_from_lists(images_train, masks_train, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, is_train=True)
validation_generator = generator_from_lists(images_validation, masks_validation, batch_size=BATCH_SIZE, random_state=RANDOM_STATE)

import segmentation_models as sm
# model 불러오기
model = get_attention_AFD(input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS, nClasses=1, dilation_rate=2)
model.summary()
model.compile(optimizer = Adam(learning_rate=1e-2), loss=sm.losses.bce_jaccard_loss, metrics = ['accuracy', sm.metrics.iou_score])


# checkpoint 및 조기종료 설정
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='val_iou_score', verbose=1,
save_best_only=True, mode='max', period=CHECKPOINT_PERIOD)
rlr = ReduceLROnPlateau(monitor='val_loss',
                        patience=10, #early stopping 의 절반
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
    initial_epoch=INITIAL_EPOCH
)
print('---model 훈련 종료---')


print('가중치 저장')
model_weights_output = os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT)
model.save_weights(model_weights_output)
print("저장된 가중치 명: {}".format(model_weights_output))
y_pred_dict = {}

for idx, i in enumerate(test_meta['test_img']):
    img = get_img_arr(f'datasets/test_img/{i}') 
    y_pred = model.predict(np.array([img]), batch_size=32)
    y_pred = np.where(y_pred[0, :, :, 0] > THESHOLDS, 1, 0) # 임계값 처리
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred
    
joblib.dump(y_pred_dict, f'predict/{MODEL_NAME}_{save_name}_y_pred.pkl')
print("저장된 pkl:", f'predict/{MODEL_NAME}_{save_name}_y_pred.pkl')
