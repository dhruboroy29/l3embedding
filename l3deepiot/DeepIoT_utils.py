from .training_utils import multi_gpu_model
import tensorflow as tf
from .consts_globals import *

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