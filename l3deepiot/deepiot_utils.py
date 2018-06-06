import tensorflow as tf

###### Util Start
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
    layer_size_dict = {u'acc_conv1': 2 * 3 * CONV_LEN, u'acc_conv2': CONV_LEN_INTE, u'gyro_conv1': 2 * 3 * CONV_LEN,
                       u'gyro_conv2': CONV_LEN_INTE,
                       u'acc_conv3': CONV_LEN_LAST, u'gyro_conv3': CONV_LEN_LAST, u'sensor_conv1': 2 * CONV_MERGE_LEN,
                       u'sensor_conv2': 2 * CONV_MERGE_LEN,
                       u'sensor_conv3': 2 * CONV_MERGE_LEN, u'cell_0': 8, u'cell_1': 1}
    ord_list1 = [u'acc_conv1', u'acc_conv2', u'acc_conv3']
    ord_list2 = [u'gyro_conv1', u'gyro_conv2', u'gyro_conv3']
    ord_list3 = [u'sensor_conv1', u'sensor_conv2', u'sensor_conv3', u'cell_0', u'cell_1']

    org_size = 0
    comps_size = 0
    for idx, layer_name in enumerate(ord_list1):
        if idx == 0:
            org_size += layer_size_dict[layer_name] * 1 * org_dim_dict[layer_name] + org_dim_dict[layer_name]
            comps_size += layer_size_dict[layer_name] * 1 * cur_left_num[layer_name] + cur_left_num[layer_name]
        else:
            last_layer_name = ord_list1[idx - 1]
            org_size += layer_size_dict[layer_name] * org_dim_dict[last_layer_name] * org_dim_dict[layer_name] + \
                        org_dim_dict[layer_name]
            comps_size += layer_size_dict[layer_name] * cur_left_num[last_layer_name] * cur_left_num[layer_name] + \
                          cur_left_num[layer_name]
    for idx, layer_name in enumerate(ord_list2):
        if idx == 0:
            org_size += layer_size_dict[layer_name] * 1 * org_dim_dict[layer_name] + org_dim_dict[layer_name]
            comps_size += layer_size_dict[layer_name] * 1 * cur_left_num[layer_name] + cur_left_num[layer_name]
        else:
            last_layer_name = ord_list1[idx - 1]
            org_size += layer_size_dict[layer_name] * org_dim_dict[last_layer_name] * org_dim_dict[layer_name] + \
                        org_dim_dict[layer_name]
            comps_size += layer_size_dict[layer_name] * cur_left_num[last_layer_name] * cur_left_num[layer_name] + \
                          cur_left_num[layer_name]
    for idx, layer_name in enumerate(ord_list3):
        if idx == 0:
            last_layer_name = u'acc_conv3'
        else:
            last_layer_name = ord_list3[idx - 1]
        if not 'cell' in layer_name:
            org_size += layer_size_dict[layer_name] * org_dim_dict[last_layer_name] * org_dim_dict[layer_name] + \
                        org_dim_dict[layer_name]
            comps_size += layer_size_dict[layer_name] * cur_left_num[last_layer_name] * cur_left_num[layer_name] + \
                          cur_left_num[layer_name]
        else:
            org_size += (layer_size_dict[layer_name] * org_dim_dict[last_layer_name] + org_dim_dict[layer_name]) * 3 * \
                        org_dim_dict[layer_name] + 3 * org_dim_dict[layer_name]
            comps_size += (layer_size_dict[layer_name] * cur_left_num[last_layer_name] + cur_left_num[layer_name]) * 3 * \
                          cur_left_num[layer_name] + 3 * cur_left_num[layer_name]
    org_size += org_dim_dict[u'cell_1'] * 6 + 6
    comps_size += cur_left_num[u'cell_1'] * 6 + 6
    comps_ratio = float(comps_size) * 100. / org_size
    print('Original Size:', org_size, 'Compressed Size:', comps_size, 'Left Ratio:', comps_ratio)
    return comps_ratio


###### Util End