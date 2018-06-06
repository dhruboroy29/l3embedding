import tensorflow as tf

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