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
from keras.applications import EfficientNetB0,EfficientNetB2,EfficientNetB3, EfficientNetB7
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
EPOCHS = 500 # 훈련 epoch 지정
BATCH_SIZE = 4 # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'pretrained_attention_unet_resnet18' # 모델 이름
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
WORKERS = 60

# 조기종료
EARLY_STOP_PATIENCE = 20

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
            'u3_output_final_activation_loss': logs['u3_output_final_activation_loss'],
            'val_u3_output_final_activation_loss': logs['val_u3_output_final_activation_loss'],
            'u3_output_final_activation_iou_score': logs['u3_output_final_activation_iou_score'],
            'val_u3_output_final_activation_iou_score': logs['val_u3_output_final_activation_iou_score']
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
    var = 1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy_image

def augment_image(image, mask, IMAGE_SIZE=(256, 256)):
    per = 0.4
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
    
    # if random.random() < per:
    #     factor = random.uniform(0.9, 1.1)
    #     image = adjust_brightness(image, factor=factor)
    
    # if random.random() < per:
    #     image = add_noise(image)
    return image, mask


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

            img = fopen_image(img_path, bands=(7,6,2))
            mask = fopen_mask(mask_path)
            
            if is_train:
                # #대비조절
                # img = np.uint8(img * 255)  # 이미지를 8-bit 정수 타입으로 변환
                # img = enhance_image_contrast(img)
                # img = img.astype(np.float32) / 255. #다시 32 float 타입 변환
                img, mask = augment_image(img, mask)
            images.append(img)
            masks.append(mask)

            if len(images) >= batch_size:
                yield (np.array(images), np.array(masks))
                images = []
                masks = []
                
                
            
################
#collect
from keras_unet_collection import models
from keras_unet_collection import activations
from keras_unet_collection import losses
from keras_unet_collection import models, base
def get_c_unet3plus():
    # model 불러오기
    filter_num_down = [32, 64, 128, 256, 512]
    filter_num_skip = [32, 32, 32, 32]
    filter_num_aggregate = 160

    stack_num_down = 2
    stack_num_up = 1
    n_labels = 1
    name = 'u3'
    input_tensor = keras.layers.Input((256, 256, 3))
    X_decoder = base.unet_3plus_2d_base(
        input_tensor, filter_num_down, filter_num_skip, filter_num_aggregate, 
        stack_num_down=stack_num_down, stack_num_up=stack_num_up, activation='ReLU', 
        batch_norm=True, pool=True, unpool=True, backbone=None, name=name)
    # allocating deep supervision tensors
    OUT_stack = []
    # reverse indexing `X_decoder`, so smaller tensors have larger list indices 
    X_decoder = X_decoder[::-1]

    # deep supervision outputs
    for i in range(1, len(X_decoder)):
        # 3-by-3 conv2d --> upsampling --> sigmoid output activation
        pool_size = 2**(i)
        X = Conv2D(n_labels, 3, padding='same', name='{}_output_conv1_{}'.format(name, i-1))(X_decoder[i])
        
        X = UpSampling2D((pool_size, pool_size), interpolation='bilinear', 
                        name='{}_output_sup{}'.format(name, i-1))(X)
        
        X = Activation('sigmoid', name='{}_output_sup{}_activation'.format(name, i-1))(X)
        # collecting deep supervision tensors
        OUT_stack.append(X)

    # the final output (without extra upsampling)
    # 3-by-3 conv2d --> sigmoid output activation
    X = Conv2D(n_labels, 3, padding='same', name='{}_output_final'.format(name))(X_decoder[0])
    X = Activation('sigmoid', name='{}_output_final_activation'.format(name))(X)
    # collecting final output tensors
    OUT_stack.append(X)
    X_CGM = X_decoder[-1]
    X_CGM = Dropout(rate=0.1)(X_CGM)
    X_CGM = Conv2D(filter_num_skip[-1], 1, padding='same')(X_CGM)
    X_CGM = GlobalMaxPooling2D()(X_CGM)
    X_CGM = Activation('sigmoid')(X_CGM)

    # CGM_mask = max(X_CGM, axis=-1) # <----- This value could be trained with "none-organ image"

    # for i in range(len(OUT_stack)):
    #     if i < len(OUT_stack)-1:
    #         OUT_stack[i] = multiply([OUT_stack[i], X_CGM], name='{}_output_sup{}_CGM'.format(name, i))
    #     else:
    #         OUT_stack[i] = multiply([OUT_stack[i], X_CGM], name='{}_output_final_CGM'.format(name))
    
    unet3plus = Model(inputs=[input_tensor,], outputs=OUT_stack)
    return unet3plus
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
x_tr, x_val = train_test_split(train_meta, test_size=0.2, random_state=RANDOM_STATE)
print(len(x_tr), len(x_val)) #26860 6715

# train : val 지정 및 generator
images_train = [os.path.join(IMAGES_PATH, image) for image in x_tr['train_img'] ]
masks_train = [os.path.join(MASKS_PATH, mask) for mask in x_tr['train_mask'] ]

images_validation = [os.path.join(IMAGES_PATH, image) for image in x_val['train_img'] ]
masks_validation = [os.path.join(MASKS_PATH, mask) for mask in x_val['train_mask'] ]


train_generator = generator_from_lists(images_train, masks_train, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, is_train=True)
validation_generator = generator_from_lists(images_validation, masks_validation, batch_size=BATCH_SIZE, random_state=RANDOM_STATE)

from keras_unet_collection import losses

def hybrid_loss(y_true, y_pred):

    loss_focal = losses.focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3)
    loss_iou = losses.iou_seg(y_true, y_pred)
    
    # (x) 
    loss_ssim = losses.ms_ssim(y_true, y_pred, max_val=1.0, filter_size=4)
    
    return loss_focal+loss_iou + loss_ssim

import segmentation_models as sm
model = get_c_unet3plus()
model.compile(loss=[sm.losses.bce_dice_loss, sm.losses.bce_dice_loss, sm.losses.bce_dice_loss, sm.losses.bce_dice_loss, sm.losses.bce_dice_loss],
                  loss_weights=[0.25, 0.25, 0.25, 0.25, 1.0],
                  optimizer=keras.optimizers.Adam(learning_rate=1e-2), metrics=[sm.metrics.iou_score])
model.summary()


# checkpoint 및 조기종료 설정
es = EarlyStopping(monitor='val_u3_output_final_activation_loss', mode='min', verbose=1, patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='val_u3_output_final_activation_loss', verbose=1,
save_best_only=True, mode='min', period=CHECKPOINT_PERIOD)
rlr = ReduceLROnPlateau(monitor='val_u3_output_final_activation_loss',
                        patience=10, #early stopping 의 절반
                        mode = 'min',
                        verbose= 1,
                        factor=0.3 #learning rate 를 반으로 줄임.
                        )

print('---model 훈련 시작---')
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(images_train) // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(images_validation) // BATCH_SIZE,
    callbacks=[checkpoint, es,rlr,CometLogger()],
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
    img = get_img_arr(f'datasets/test_img/{i}', (7,6,2)) 
    y_pred = model.predict(np.array([img]), batch_size=32)
    y_pred = np.where(y_pred[0, :, :, 0] > THESHOLDS, 1, 0) # 임계값 처리
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred
    
joblib.dump(y_pred_dict, f'predict/{MODEL_NAME}_{save_name}_y_pred.pkl')
print("저장된 pkl:", f'predict/{MODEL_NAME}_{save_name}_y_pred.pkl')