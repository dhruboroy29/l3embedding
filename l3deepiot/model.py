'''
Created by: Dhrubojyoti Roy
Purpose:    Distills model construct_cnn_L3_melspec2 (audio, vision and merge components)
            from ../l3embedding/model.py, ../l3embedding/audio_model.py, ../l3embedding/vision_model.py
TODO:       Eventually get rid of this; integrate DeepIoT seamlessly with ../l3embedding/
'''

'''
Imports
'''
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, \
    Flatten, Activation, concatenate, Dense, Lambda
from keras import backend as K
from kapre.time_frequency import Melspectrogram
import tensorflow as tf
import keras.regularizers as regularizers
from .DeepIoT_dropOut import dropout as DeepIoT_dropout, prev_binary_tensor as prev_binary_tensor
from .DeepIoT_utils import *


'''
Model
'''
@gpu_wrapper
def construct_cnn_L3_melspec2():
    """
    Constructs a model that replicates that used in Look, Listen and Learn

    Relja Arandjelovic and (2017). Look, Listen and Learn. CoRR, abs/1705.08168, .

    Returns
    -------
    model:  L3 CNN model
            (Type: keras.models.Model)
    inputs: Model inputs
            (Type: list[keras.layers.Input])
    outputs: Model outputs
            (Type: keras.layers.Layer)
    """
    vision_model, x_i, y_i = construct_cnn_L3_orig_vision_model()
    audio_model, x_a, y_a = construct_cnn_L3_melspec2_audio_model()

    m = L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, 'cnn_L3_melspec2') # 'cnn_L3_kapredbinputbn'
    return m


# Audio model with added DeepIoT_dropout layers
def construct_cnn_L3_melspec2_audio_model(batch_size=64, train=True):
    """
    Constructs a model that replicates the audio subnetwork  used in Look,
    Listen and Learn

    Relja Arandjelovic and (2017). Look, Listen and Learn. CoRR, abs/1705.08168, .

    Returns
    -------
    model:  L3 CNN model
            (Type: keras.models.Model)
    inputs: Model inputs
            (Type: list[keras.layers.Input])
    outputs: Model outputs
            (Type: keras.layers.Layer)
    """
    ####
    # Audio subnetwork with added DeepIoT_dropout layers and variable scopes
    ####
    weight_decay = 1e-5
    n_dft = 2048
    #n_win = 480
    #n_hop = n_win//2
    n_mels = 256
    n_hop = 242
    asr = 48000
    audio_window_dur = 1

    # Added by Dhrubo: DeepIoT mask
    out_binary_mask = {}

    # INPUT
    x_a = Input(batch_shape=(batch_size, 1, asr * audio_window_dur), dtype='float32')


    # MELSPECTROGRAM PREPROCESSING
    # 128 x 199 x 1
    y_a = Melspectrogram(n_dft=n_dft, n_hop=n_hop, n_mels=n_mels,
                      power_melgram=1.0, htk=True, # n_win=n_win,
                      return_decibel_melgram=True, padding='same')(x_a)


    # CONV BLOCK 1
    pool_size_a_1 = (2, 2)
    with tf.variable_scope("conv1_1"):
        y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    with tf.variable_scope("conv1_BN1"):
        y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    with tf.variable_scope("conv1_2"):
        y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    with tf.variable_scope("conv1_BN2"):
        y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_1, strides=2)(y_a)
    conv1_shape = y_a.get_shape().as_list()
    y_a = Lambda(lambda input: DeepIoT_dropout(input, dropOut_prun(prob_list_conv1, prun_thres, sol_train),
                                                               is_training=train,
                                                               noise_shape=[conv1_shape[0], 1, 1, conv1_shape[3]],
                                                               name='conv1_dropout1'))(y_a)


    #out_binary_mask[u'conv1'] = conv1_dropB1


    # CONV BLOCK 2
    pool_size_a_2 = (2, 2)
    with tf.variable_scope("conv2_1"):
        y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    with tf.variable_scope("conv2_BN1"):
        y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    with tf.variable_scope("conv2_2"):
        y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    with tf.variable_scope("conv2_BN2"):
        y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_2, strides=2)(y_a)
    conv2_shape = y_a.get_shape().as_list()
    y_a = Lambda(lambda input: DeepIoT_dropout(input, dropOut_prun(prob_list_conv2, prun_thres, sol_train),
                                               is_training=train,
                                               noise_shape=[conv2_shape[0], 1, 1, conv2_shape[3]],
                                               name='conv2_dropout1'))(y_a)
    #out_binary_mask[u'conv2'] = conv2_dropB1


    # CONV BLOCK 3
    pool_size_a_3 = (2, 2)
    with tf.variable_scope("conv3_1"):
        y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    with tf.variable_scope("conv3_BN1"):
        y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    with tf.variable_scope("conv3_2"):
        y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    with tf.variable_scope("conv3_BN2"):
        y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_3, strides=2)(y_a)
    conv3_shape = y_a.get_shape().as_list()
    y_a = Lambda(lambda input: DeepIoT_dropout(input, dropOut_prun(prob_list_conv3, prun_thres, sol_train),
                                        is_training=train,
                                        noise_shape=[conv3_shape[0], 1, 1, conv3_shape[3]],
                                        name='conv3_dropout1'))(y_a)
    '''y_a, conv3_dropB1 = DeepIoT_dropout(y_a, dropOut_prun(prob_list_conv3, prun_thres, sol_train),
                                        is_training=train,
                                        noise_shape=[conv3_shape[0], 1, 1, conv3_shape[3]],
                                        name='conv3_dropout1')'''
    #out_binary_mask[u'conv3'] = conv3_dropB1


    # CONV BLOCK 4
    pool_size_a_4 = (32, 24)
    with tf.variable_scope("conv4_1"):
        y_a = Conv2D(n_filter_a_4, filt_size_a_4, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    with tf.variable_scope("conv4_BN1"):
        y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    with tf.variable_scope("conv4_2"):
        y_a = Conv2D(n_filter_a_4, filt_size_a_4,
                 kernel_initializer='he_normal',
                 name='audio_embedding_layer', padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    with tf.variable_scope("conv4_BN2"):
        y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_4)(y_a)
    conv4_shape = y_a.get_shape().as_list()
    y_a = Lambda(lambda input: DeepIoT_dropout(input, dropOut_prun(prob_list_conv4, prun_thres, sol_train),
                                        is_training=train,
                                        noise_shape=[conv4_shape[0], 1, 1, conv4_shape[3]],
                                        name='conv4_dropout1'))(y_a)
    '''y_a, conv4_dropB1 = DeepIoT_dropout(y_a, dropOut_prun(prob_list_conv4, prun_thres, sol_train),
                                        is_training=train,
                                        noise_shape=[conv4_shape[0], 1, 1, conv4_shape[3]],
                                        name='conv4_dropout1')'''
    #out_binary_mask[u'conv4'] = conv4_dropB1


    # FLATTEN
    y_a = Flatten()(y_a)

    m = Model(inputs=x_a, outputs=y_a)
    m.name = 'audio_model'

    return m, x_a, y_a


# Vision model
def construct_cnn_L3_orig_vision_model(batch_size=64):
    """
    Constructs a model that replicates the vision subnetwork  used in Look,
    Listen and Learn

    Relja Arandjelovic and (2017). Look, Listen and Learn. CoRR, abs/1705.08168, .

    Returns
    -------
    model:  L3 CNN model
            (Type: keras.models.Model)
    inputs: Model inputs
            (Type: list[keras.layers.Input])
    outputs: Model outputs
            (Type: keras.layers.Layer)
    """
    weight_decay = 1e-5
    ####
    # Image subnetwork
    ####
    # INPUT
    x_i = Input(batch_shape=(batch_size, 224, 224, 3), dtype='float32')

    # CONV BLOCK 1
    n_filter_i_1 = 64
    filt_size_i_1 = (3, 3)
    pool_size_i_1 = (2, 2)
    y_i = Conv2D(n_filter_i_1, filt_size_i_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(x_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = Conv2D(n_filter_i_1, filt_size_i_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = Activation('relu')(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = MaxPooling2D(pool_size=pool_size_i_1, strides=2, padding='same')(y_i)

    # CONV BLOCK 2
    n_filter_i_2 = 128
    filt_size_i_2 = (3, 3)
    pool_size_i_2 = (2, 2)
    y_i = Conv2D(n_filter_i_2, filt_size_i_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = Conv2D(n_filter_i_2, filt_size_i_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = MaxPooling2D(pool_size=pool_size_i_2, strides=2, padding='same')(y_i)

    # CONV BLOCK 3
    n_filter_i_3 = 256
    filt_size_i_3 = (3, 3)
    pool_size_i_3 = (2, 2)
    y_i = Conv2D(n_filter_i_3, filt_size_i_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = Conv2D(n_filter_i_3, filt_size_i_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = MaxPooling2D(pool_size=pool_size_i_3, strides=2, padding='same')(y_i)

    # CONV BLOCK 4
    n_filter_i_4 = 512
    filt_size_i_4 = (3, 3)
    pool_size_i_4 = (28, 28)
    y_i = Conv2D(n_filter_i_4, filt_size_i_4, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = Conv2D(n_filter_i_4, filt_size_i_4,
                 name='vision_embedding_layer', padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = MaxPooling2D(pool_size=pool_size_i_4, padding='same')(y_i)
    y_i = Flatten()(y_i)

    m = Model(inputs=x_i, outputs=y_i)
    m.name = 'vision_model'

    return m, x_i, y_i


# Merge models
def L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, model_name, layer_size=128):
    """
    Merges the audio and vision subnetworks and adds additional fully connected
    layers in the fashion of the model used in Look, Listen and Learn

    Relja Arandjelovic and (2017). Look, Listen and Learn. CoRR, abs/1705.08168, .

    Returns
    -------
    model:  L3 CNN model
            (Type: keras.models.Model)
    inputs: Model inputs
            (Type: list[keras.layers.Input])
    outputs: Model outputs
            (Type: keras.layers.Layer)
    """
    # Merge the subnetworks
    weight_decay = 1e-5
    y = concatenate([vision_model(x_i), audio_model(x_a)])
    y = Dense(layer_size, activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l2(weight_decay))(y)
    y = Dense(2, activation='softmax',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l2(weight_decay))(y)
    m = Model(inputs=[x_i, x_a], outputs=y)
    m.name = model_name

    return m, [x_i, x_a], y


'''
Models we're interested in
'''
MODELS = {
    'cnn_L3_melspec2': construct_cnn_L3_melspec2
}
