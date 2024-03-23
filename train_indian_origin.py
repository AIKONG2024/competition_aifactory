
from keras.layers import Input, Convolution2D, BatchNormalization, Activation, LeakyReLU, Add, ReLU, GlobalAveragePooling2D, Reshape, AveragePooling2D, UpSampling2D, Flatten
from keras.models import Model
import tensorflow as tf
# from keras_self_attention import SeqSelfAttention
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
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate, DepthwiseConv2D
from keras.applications import MobileNetV2
from keras.models import *
from keras.layers import *
import numpy as np
from keras import backend as K
from sklearn.model_selection import train_test_split
import segmentation_models as sm
import joblib
import keras_unet_collection

tf.random.set_seed(3)
np.random.seed(3)
MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값
THESHOLDS = 0.25

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

def get_img_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img

def get_img_762bands(path):
    img = rasterio.open(path).read((7,6,2)).transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img



def get_mask_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    seg = np.float32(img)
    return seg

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
    # green
    green = AveragePooling2D(pool_size=(8,8),name='green_pool')(base)
    green = Convolution2D(filters=32,kernel_size=(1,1),name='green_1_by_1')(green)
    green = UpSampling2D(size=8,interpolation='bilinear',name='green_upsampling')(green)
    # base + red + yellow + blue + green
    return tf.keras.layers.concatenate([base,red,yellow,blue,green])

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

# Build the model
model = build_model(input_shape)
model.summary()

# # 두 샘플 간의 유사성 metric
# def dice_coef(y_true, y_pred, smooth=1):
#     intersection = K.sum(y_true * y_pred, axis=[1,2,3])
#     union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
#     dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
#     return dice

# # 픽셀 정확도를 계산 metric
# def pixel_accuracy (y_true, y_pred):
#     sum_n = np.sum(np.logical_and(y_pred, y_true))
#     sum_t = np.sum(y_true)
 
#     if (sum_t == 0):
#         pixel_accuracy = 0
#     else:
#         pixel_accuracy = sum_n / sum_t
#     return pixel_accuracy  

# # 사용할 데이터의 meta정보 가져오기

train_meta = pd.read_csv('datasets/train_meta.csv')
test_meta = pd.read_csv('datasets/test_meta.csv')

#  저장 이름
save_name = 'indian0318'

N_FILTERS = 22 # 필터수 지정
N_CHANNELS = 3 # channel 지정
EPOCHS = 1000 # 훈련 epoch 지정
BATCH_SIZE = 12  # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'concat' # 모델 이름
RANDOM_STATE = 3812 # seed 고정
INITIAL_EPOCH = 0 # 초기 epoch

# 데이터 위치
IMAGES_PATH = 'C:\\_data\\dataset\\train_img\\'
MASKS_PATH = 'C:\\_data\\dataset\\train_mask\\'

# 가중치 저장 위치
OUTPUT_DIR = 'C:\\_data\\dataset\\output\\'
WORKERS = 20

# 조기종료
EARLY_STOP_PATIENCE = 21


# 중간 가중치 저장 이름
CHECKPOINT_PERIOD = 10
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}indian0320Swi.hdf5'.format(MODEL_NAME, save_name)
 
# 최종 가중치 저장 이름
FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_indian0320Swi.h5'.format(MODEL_NAME, save_name)

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
x_tr, x_val = train_test_split(train_meta, test_size=0.15, random_state=RANDOM_STATE)
print(len(x_tr), len(x_val))

# train : val 지정 및 generator
images_train = [os.path.join(IMAGES_PATH, image) for image in x_tr['train_img'] ]
masks_train = [os.path.join(MASKS_PATH, mask) for mask in x_tr['train_mask'] ]

images_validation = [os.path.join(IMAGES_PATH, image) for image in x_val['train_img'] ]
masks_validation = [os.path.join(MASKS_PATH, mask) for mask in x_val['train_mask'] ]

train_generator = generator_from_lists(images_train, masks_train, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")
validation_generator = generator_from_lists(images_validation, masks_validation, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")


# model 불러오기
# model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
# model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])
# model.summary()

# model = get_attention_unet()
# model = get_model(MODEL_NAME, nClasses=1, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
learning_rate = 0.005
model.compile(optimizer=Adam(learning_rate=learning_rate), loss=sm.losses.bce_jaccard_loss, metrics=['accuracy', miou])
model.summary()

# checkpoint 및 조기종료 설정
es = EarlyStopping(monitor='val_miou', mode='max', verbose=1, patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
# es = EarlyStopping(monitor='val_miou', mode='max', verbose=1, patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='val_miou', verbose=1,
save_best_only=True, mode='max')

rlr = ReduceLROnPlateau(monitor='val_miou',             # 통상 early_stopping patience보다 작다
                        patience=10,
                        mode='max',
                        verbose=1,
                        factor=0.5,
                        # 통상 디폴트보다 높게 잡는다?
                        )

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

print('가중치 저장')
model_weights_output = os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT)
model.save_weights(model_weights_output)
print("저장된 가중치 명: {}".format(model_weights_output))

# model.load_weights('C:\\_data\\dataset\\output\\model_concat_indian0318_indian0319Swi.h5')

# y_pred_dict = {}

# for i in test_meta['test_img']:
#     img = get_img_762bands(f'C:\\_data\\dataset\\test_img\\{i}')
#     y_pred = model.predict(np.array([img]), batch_size=1)
    
#     y_pred = np.where(y_pred[0, :, :, 0] > 0.23, 1, 0) # 임계값 처리
#     y_pred = y_pred.astype(np.uint8)
#     y_pred_dict[i] = y_pred

# joblib.dump(y_pred_dict, 'C:\\_data\\dataset\\output\\03200759.pkl')

# print(y_pred_dict)