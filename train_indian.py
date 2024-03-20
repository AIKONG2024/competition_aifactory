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
from skimage.transform import resize
import segmentation_models as sm
# import tensorflow_hub as hub
import cv2

#랜럼시드 고정
RANDOM_STATE = 42 # seed 고정
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값

N_FILTERS = 8 # 필터수 지정
N_CHANNELS = 3 # channel 지정
EPOCHS = 300 # 훈련 epoch 지정
BATCH_SIZE = 8 # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'pret_att_unet' # 모델 이름
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

def augment_image(image, mask, IMAGE_SIZE=(256, 256), per=0.4):
    # 확률적으로 이미지 변환 적용
    if random.random() < per:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    
    if random.random() < per:
        angle = random.choice([90, 180, 270])
        image = rotate_image(image, angle)
        mask = rotate_image(mask, angle)
    
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

            img = fopen_image(img_path, bands=(7,6,2))
            mask = fopen_mask(mask_path)
            
            if is_train:
                img, mask = augment_image(img, mask)
            images.append(img)
            masks.append(mask)

            if len(images) >= batch_size:
                yield (np.array(images), np.array(masks))
                images = []
                masks = []



###########################################################################

from keras.layers import Layer
import keras.backend as K
class SwishActivation(Layer):
    def __init__(self, **kwargs):
        super(SwishActivation, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * K.sigmoid(inputs)

    def get_config(self):
        config = super().get_config()
        return config
    
# conv_block 함수 내에서 SwishActivation을 사용하는 방법
from keras.layers import Add, Conv2D, BatchNormalization, ReLU


def conv_block(X, filters, block):
    # Residual block with dilated convolutions
    # Add skip connection at last after doing convolution operation to input X

    b = 'block_' + str(block) + '_'
    f1, f2, f3 = filters
    X_skip = X
    # block_a
    X = Conv2D(filters=f1, kernel_size=(1, 1), dilation_rate=(1, 1),
               padding='same', kernel_initializer='he_normal', name=b + 'a')(X)
    X = BatchNormalization(name=b + 'batch_norm_a')(X)
    X = SwishActivation(name=b + 'swish_activation_a')(X)
    # block_b
    X = Conv2D(filters=f2, kernel_size=(3, 3), dilation_rate=(2, 2),
               padding='same', kernel_initializer='he_normal', name=b + 'b')(X)
    X = BatchNormalization(name=b + 'batch_norm_b')(X)
    X = SwishActivation(name=b + 'swish_activation_b')(X)
    # block_c
    X = Conv2D(filters=f3, kernel_size=(1, 1), dilation_rate=(1, 1),
               padding='same', kernel_initializer='he_normal', name=b + 'c')(X)
    X = BatchNormalization(name=b + 'batch_norm_c')(X)
    # skip_conv
    X_skip = Conv2D(filters=f3, kernel_size=(3, 3), padding='same', name=b + 'skip_conv')(X_skip)
    X_skip = BatchNormalization(name=b + 'batch_norm_skip_conv')(X_skip)
    # block_c + skip_conv
    X = Add(name=b + 'add')([X, X_skip])
    X = ReLU(name=b + 'relu')(X)
    return X
    
def base_feature_maps(input_layer):
    # base covolution module to get input image feature maps 
    
    # block_1
    base = conv_block(input_layer,[32,32,64],'1')
    # block_2
    base = conv_block(base,[64,64,128],'2')
    # block_3
    base = conv_block(base,[128,128,256],'3')
    return base

def pyramid_feature_maps(input_layer):
    # pyramid pooling module
    
    base = base_feature_maps(input_layer)
    # red
    red = GlobalAveragePooling2D(name='red_pool')(base)
    red = tf.keras.layers.Reshape((1,1,256))(red)
    red = Convolution2D(filters=32,kernel_size=(1,1),name='red_1_by_1')(red)
    red = UpSampling2D(size=256,interpolation='bilinear',name='red_upsampling')(red)
    # yellow
    yellow = AveragePooling2D(pool_size=(2,2),name='yellow_pool')(base)
    yellow = Convolution2D(filters=32,kernel_size=(1,1),name='yellow_1_by_1')(yellow)
    yellow = UpSampling2D(size=2,interpolation='bilinear',name='yellow_upsampling')(yellow)
    # blue
    blue = AveragePooling2D(pool_size=(4,4),name='blue_pool')(base)
    blue = Convolution2D(filters=32,kernel_size=(1,1),name='blue_1_by_1')(blue)
    blue = UpSampling2D(size=4,interpolation='bilinear',name='blue_upsampling')(blue)
    
    # base + red + yellow + blue + green
    return tf.keras.layers.concatenate([base,red,yellow,blue])

def last_conv_module(input_layer):
    X = pyramid_feature_maps(input_layer)
    X = Convolution2D(filters=1,kernel_size=3,padding='same',name='last_conv_3_by_3')(X)
    X = BatchNormalization(name='last_conv_3_by_3_batch_norm')(X)
    X = Activation('sigmoid',name='last_conv_sigmoid')(X)
    # X = tf.keras.layers.Flatten(name='last_conv_flatten')(X)
    return X

def build_model(input_shape):
    # Input layer
    input_layer = Input(shape=input_shape)

    # Constructing the model
    last_layer = last_conv_module(input_layer)

    # Creating the model
    model = Model(inputs=input_layer, outputs=last_layer)

    return model

input_shape = (256, 256, 3)  # Replace height, width, and channels with actual values
        
##########################################################################
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


train_generator = generator_from_lists(images_train, masks_train, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, is_train= True)
validation_generator = generator_from_lists(images_validation, masks_validation, batch_size=BATCH_SIZE, random_state=RANDOM_STATE)


model = build_model((IMAGE_SIZE[0],IMAGE_SIZE[1],N_CHANNELS))
model.compile(optimizer = Adam(learning_rate=0.001), loss =sm.losses.bce_jaccard_loss, metrics = ['accuracy', sm.metrics.iou_score])
model.summary()


# checkpoint 및 조기종료 설정
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='val_loss', verbose=1,
save_best_only=True, mode='min', period=CHECKPOINT_PERIOD)
rlr = ReduceLROnPlateau(monitor='val_loss',
                        patience=10, #early stopping 의 절반
                        mode = 'min',
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
    img = get_img_arr(f'datasets/test_img/{i}', (7,6,2)) 
    y_pred = model.predict(np.array([img]), batch_size=32)
    y_pred = np.where(y_pred[0, :, :, 0] > THESHOLDS, 1, 0) # 임계값 처리
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred
    
joblib.dump(y_pred_dict, f'predict/{MODEL_NAME}_{save_name}_y_pred.pkl')
print("저장된 pkl:", f'predict/{MODEL_NAME}_{save_name}_y_pred.pkl')