'''
Created by: Dhrubojyoti Roy
Purpose:    Distills model construct_cnn_L3_melspec2 (audio, vision and merge components)
            from ../l3embedding/model.py, ../l3embedding/audio_model.py, ../l3embedding/vision_model.py
TODO:       Eventually get rid of this; integrate DeepIoT seamlessly with ../l3embedding/
'''

'''
Imports
'''
from l3embedding.training_utils import multi_gpu_model
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, \
    Flatten, Activation, concatenate, Dense
from keras import backend as K
from kapre.time_frequency import Melspectrogram
import tensorflow as tf
import keras.regularizers as regularizers
from .DeepIoT_dropOut import dropout as DeepIoT_dropout

'''
DeepIoT constants: added by Dhrubo
'''

CONV_KEEP_PROB = 0.8

# Number of filters
n_filter_a_1 = 64
n_filter_a_2 = 128
n_filter_a_3 = 256
n_filter_a_4 = 512

# Size of filters
filt_size_a_1 = (3, 3)
filt_size_a_2 = (3, 3)
filt_size_a_3 = (3, 3)
filt_size_a_4 = (3, 3)

# Creating prob_list_ variables
prob_list_conv1 = tf.get_variable("prob_list_conv1", [1, 1, 1, n_filter_a_1], tf.float32,
                             tf.constant_initializer(CONV_KEEP_PROB), trainable=False)
prob_list_conv2 = tf.get_variable("prob_list_conv2", [1, 1, 1, n_filter_a_2], tf.float32,
                                  tf.constant_initializer(CONV_KEEP_PROB), trainable=False)
prob_list_conv3 = tf.get_variable("prob_list_conv3", [1, 1, 1, n_filter_a_3], tf.float32,
                                  tf.constant_initializer(CONV_KEEP_PROB), trainable=False)
prob_list_conv4 = tf.get_variable("prob_list_conv4", [1, 1, 1, n_filter_a_4], tf.float32,
                                  tf.constant_initializer(CONV_KEEP_PROB), trainable=False)

# sol_train and prune_thresh
sol_train = tf.Variable(0, dtype=tf.float32, trainable=False)
prun_thres = tf.get_variable("prun_thres", [], tf.float32, tf.constant_initializer(0.0), trainable=False)

# Initializing dictionaries
prob_list_dict = {u'conv1': prob_list_conv1, u'conv2': prob_list_conv2, u'conv3': prob_list_conv3,
                  u'conv4': prob_list_conv4}

org_dim_dict = {u'conv1': n_filter_a_1, u'conv2': n_filter_a_2, u'conv3': n_filter_a_3,
                u'conv4': n_filter_a_4}

layer_size_dict = {u'conv1': filt_size_a_1[0] * filt_size_a_1[1], u'conv2': filt_size_a_2[0] * filt_size_a_2[1],
                   u'conv3': filt_size_a_3[0] * filt_size_a_3[1], u'conv4': filt_size_a_4[0] * filt_size_a_4[1]}


'''
Utils
'''
def gpu_wrapper(model_f):
    """
    Decorator for creating multi-gpu models
    """

    def wrapped(num_gpus=0, *args, **kwargs):
        m, inp, out = model_f(*args, **kwargs)
        if num_gpus > 1:
            m = multi_gpu_model(m, gpus=num_gpus)

        return m, inp, out

    return wrapped

###### DeepIoT Util Start
def dropOut_prun(drop_prob, prun_thres, sol_train):
    base_prob = 0.5
    pruned_drop_prob = tf.cond(sol_train > 0.5,
                               lambda: tf.where(tf.less(drop_prob, prun_thres), tf.zeros_like(drop_prob), drop_prob),
                               lambda: tf.where(tf.less(drop_prob, prun_thres), drop_prob * base_prob, drop_prob))
    return pruned_drop_prob


def count_prun(prob_list_dict, prun_thres):
    left_num_dict = {}
    for layer_name in prob_list_dict.keys():
        prob_list = prob_list_dict[layer_name]
        pruned_idt = tf.where(tf.less(prob_list, prun_thres), tf.zeros_like(prob_list), tf.ones_like(prob_list))
        left_num = tf.reduce_sum(pruned_idt)
        left_num_dict[layer_name] = left_num
    return left_num_dict


def gen_cur_prun(sess, left_num_dict):
    cur_left_num = {}
    for layer_name in left_num_dict.keys():
        cur_left_num[layer_name] = sess.run(left_num_dict[layer_name])
    return cur_left_num


def compress_ratio(cur_left_num, org_dim_dict):
    ord_list = [u'conv1', u'conv2', u'conv3', u'conv4']

    org_size = 0
    comps_size = 0
    for idx, layer_name in enumerate(ord_list):
        if idx == 0:
            org_size += layer_size_dict[layer_name] * 1 * org_dim_dict[layer_name] + org_dim_dict[layer_name]
            comps_size += layer_size_dict[layer_name] * 1 * cur_left_num[layer_name] + cur_left_num[layer_name]
        else:
            last_layer_name = ord_list[idx - 1]
            org_size += layer_size_dict[layer_name] * org_dim_dict[last_layer_name] * org_dim_dict[layer_name] + \
                        org_dim_dict[layer_name]
            comps_size += layer_size_dict[layer_name] * cur_left_num[last_layer_name] * cur_left_num[layer_name] + \
                          cur_left_num[layer_name]
    comps_ratio = float(comps_size) * 100. / org_size
    print('Original Size:', org_size, 'Compressed Size:', comps_size, 'Left Ratio:', comps_ratio)
    return comps_ratio

###### DeepIoT Util End


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

    m = L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, 'cnn_L3_kapredbinputbn')
    return m


# Audio model with added DeepIoT_dropout layers
def construct_cnn_L3_melspec2_audio_model(train=True):
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
    # Audio subnetwork with added DeepIoT_dropout layers
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
    x_a = Input(shape=(1, asr * audio_window_dur), dtype='float32')


    # MELSPECTROGRAM PREPROCESSING
    # 128 x 199 x 1
    y_a = Melspectrogram(n_dft=n_dft, n_hop=n_hop, n_mels=n_mels,
                      power_melgram=1.0, htk=True, # n_win=n_win,
                      return_decibel_melgram=True, padding='same')(x_a)


    ## Model: variable_scopes added by Dhrubo for use with compressor ##

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
    y_a, conv1_dropB1 = DeepIoT_dropout(y_a, dropOut_prun(prob_list_conv1, prun_thres, sol_train),
                                                    is_training=train,
                                                    noise_shape=[conv1_shape[0], 1, 1, conv1_shape[3]],
                                                    name='conv1_dropout1')
    out_binary_mask[u'conv1'] = conv1_dropB1


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
    y_a, conv2_dropB1 = DeepIoT_dropout(y_a, dropOut_prun(prob_list_conv2, prun_thres, sol_train),
                                        is_training=train,
                                        noise_shape=[conv2_shape[0], 1, 1, conv2_shape[3]],
                                        name='conv2_dropout1')
    out_binary_mask[u'conv2'] = conv2_dropB1


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
    y_a, conv3_dropB1 = DeepIoT_dropout(y_a, dropOut_prun(prob_list_conv3, prun_thres, sol_train),
                                        is_training=train,
                                        noise_shape=[conv3_shape[0], 1, 1, conv3_shape[3]],
                                        name='conv3_dropout1')
    out_binary_mask[u'conv3'] = conv3_dropB1


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
    y_a, conv4_dropB1 = DeepIoT_dropout(y_a, dropOut_prun(prob_list_conv4, prun_thres, sol_train),
                                        is_training=train,
                                        noise_shape=[conv4_shape[0], 1, 1, conv4_shape[3]],
                                        name='conv4_dropout1')
    out_binary_mask[u'conv4'] = conv4_dropB1


    # FLATTEN
    y_a = Flatten()(y_a)

    m = Model(inputs=x_a, outputs=y_a)
    m.name = 'audio_model'

    return m, x_a, y_a


# Vision model
def construct_cnn_L3_orig_vision_model():
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
    x_i = Input(shape=(224, 224, 3), dtype='float32')

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
