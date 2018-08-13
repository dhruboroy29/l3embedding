import getpass
import git
import json
import datetime
import os
import pickle
import random
import csv

import numpy as np
import keras # Added by Dhrubo
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
import pescador
from skimage import img_as_float

from gsheets import get_credentials, append_row, update_experiment, get_row
from l3deepiot.model import MODELS, out_binary_mask #, load_model TODO: Remove this comment after modifying load_model() for DeepIoT
from l3deepiot.audio import pcm2float
from l3deepiot.compressor import * # Added by Dhrubo
from log import *
import h5py
import copy

from googleapiclient import discovery

LOGGER = logging.getLogger('l3deepiot')
LOGGER.setLevel(logging.DEBUG)


class LossHistory(keras.callbacks.Callback):
    """
    Keras callback to record loss history
    """

    def __init__(self, outfile):
        super().__init__()
        self.outfile = outfile

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.loss = []
        self.val_loss = []

    # def on_batch_end(self, batch, logs={}):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        loss_dict = {'loss': self.loss, 'val_loss': self.val_loss}
        with open(self.outfile, 'wb') as fp:
            pickle.dump(loss_dict, fp)

class GSheetLogger(keras.callbacks.Callback):
    """
    Keras callback to update Google Sheets Spreadsheet
    """

    def __init__(self, google_dev_app_name, spreadsheet_id, param_dict):
        super(GSheetLogger).__init__()
        self.google_dev_app_name = google_dev_app_name
        self.spreadsheet_id = spreadsheet_id
        self.credentials = get_credentials(google_dev_app_name)
        self.service = discovery.build('sheets', 'v4', credentials=self.credentials)
        self.param_dict = copy.deepcopy(param_dict)

        row_num = get_row(self.service, self.spreadsheet_id, self.param_dict, 'embedding')
        if row_num is None:
            append_row(self.service, self.spreadsheet_id, self.param_dict, 'embedding')

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.best_train_loss = float('inf')
        self.best_valid_loss = float('inf')
        self.best_train_acc = float('-inf')
        self.best_valid_acc = float('-inf')

    # def on_batch_end(self, batch, logs={}):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        latest_epoch = epoch
        latest_train_loss = logs.get('loss')
        latest_valid_loss = logs.get('val_loss')
        latest_train_acc = logs.get('acc')
        latest_valid_acc = logs.get('val_acc')

        if latest_train_loss < self.best_train_loss:
            self.best_train_loss = latest_train_loss
        if latest_valid_loss < self.best_valid_loss:
            self.best_valid_loss = latest_valid_loss
        if latest_train_acc > self.best_train_acc:
            self.best_train_acc = latest_train_acc
        if latest_valid_acc > self.best_valid_acc:
            self.best_valid_acc = latest_valid_acc

        values = [
            latest_epoch, latest_train_loss, latest_valid_loss,
            latest_train_acc, latest_valid_acc, self.best_train_loss,
            self.best_valid_loss, self.best_train_acc, self.best_valid_acc]

        update_experiment(self.service, self.spreadsheet_id, self.param_dict,
                          'R', 'AA', values, 'embedding')


class TimeHistory(keras.callbacks.Callback):
    """
    Keras callback to log epoch and batch running time
    """
    # Copied from https://stackoverflow.com/a/43186440/1260544
    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self.batch_times = []

    def on_epoch_begin(self, batch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs=None):
        t = time.time() - self.epoch_time_start
        LOGGER.info('Epoch took {} seconds'.format(t))
        self.epoch_times.append(t)

    def on_batch_begin(self, batch, logs=None):
        self.batch_time_start = time.time()

    def on_batch_end(self, batch, logs=None):
        t = time.time() - self.batch_time_start
        LOGGER.info('Batch took {} seconds'.format(t))
        self.batch_times.append(t)


def cycle_shuffle(iterable, shuffle=True):
    lst = list(iterable)
    while True:
        yield from lst
        if shuffle:
            random.shuffle(lst)


def data_generator(data_dir, batch_size=512, random_state=20180123,
                   start_batch_idx=None, keys=None):
    random.seed(random_state)

    batch = None
    curr_batch_size = 0
    batch_idx = 0

    # Limit keys to avoid producing batches with all of the metadata fields
    if not keys:
        keys = ['audio', 'video', 'label']

    for fname in cycle_shuffle(os.listdir(data_dir)):
        batch_path = os.path.join(data_dir, fname)
        blob_start_idx = 0

        blob = h5py.File(batch_path, 'r')
        blob_size = len(blob['label'])

        while blob_start_idx < blob_size:
            blob_end_idx = min(blob_start_idx + batch_size - curr_batch_size, blob_size)

            # If we are starting from a particular batch, skip computing all of
            # the prior batches
            if start_batch_idx is None or batch_idx >= start_batch_idx:
                if batch is None:
                    batch = {k:blob[k][blob_start_idx:blob_end_idx]
                             for k in keys}
                else:
                    for k in keys:
                        batch[k] = np.concatenate([batch[k],
                                                   blob[k][blob_start_idx:blob_end_idx]])

            curr_batch_size += blob_end_idx - blob_start_idx
            blob_start_idx = blob_end_idx

            if blob_end_idx == blob_size:
                blob.close()

            if curr_batch_size == batch_size:
                # If we are starting from a particular batch, skip yielding all
                # of the prior batches
                if start_batch_idx is None or batch_idx >= start_batch_idx:
                    # Preprocess video so samples are in [-1,1]
                    batch['video'] = 2 * img_as_float(batch['video']).astype('float32') - 1

                    # Convert audio to float
                    batch['audio'] = pcm2float(batch['audio'], dtype='float32')

                    yield batch

                batch_idx += 1
                curr_batch_size = 0
                batch = None


def single_epoch_data_generator(data_dir, epoch_size, **kwargs):
    while True:
        data_gen = data_generator(data_dir, **kwargs)
        for idx, item in enumerate(data_gen):
            yield item
            # Once we generate all batches for an epoch, restart the generator
            if (idx + 1) == epoch_size:
                break


def get_restart_info(history_path):
    last = None
    with open(history_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row

    return int(last['epoch']), float(last['val_acc']), float(last['val_loss'])


def train(train_data_dir, validation_data_dir, output_dir,
          num_epochs=150, train_epoch_size=512, validation_epoch_size=1024,
          train_batch_size=64, validation_batch_size=64,
          model_type='cnn_L3_orig', random_state=20180123,
          learning_rate=1e-4, verbose=False, checkpoint_interval=10,
          log_path=None, disable_logging=False, gpus=1, continue_model_dir=None,
          gsheet_id=None, google_dev_app_name=None):

    # Link Tensorflow session to Keras session
    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        tf.global_variables_initializer().run()

    init_console_logger(LOGGER, verbose=verbose)
    if not disable_logging:
        init_file_logger(LOGGER, log_path=log_path)
    LOGGER.debug('Initialized logging.')

    # Form model ID
    data_subset_name = os.path.basename(train_data_dir)
    data_subset_name = data_subset_name[:data_subset_name.rindex('_')]
    model_id = os.path.join(data_subset_name, model_type)

    param_dict = {
          'username': getpass.getuser(),
          'train_data_dir': train_data_dir,
          'validation_data_dir': validation_data_dir,
          'model_id': model_id,
          'output_dir': output_dir,
          'num_epochs': num_epochs,
          'train_epoch_size': train_epoch_size,
          'validation_epoch_size': validation_epoch_size,
          'train_batch_size': train_batch_size,
          'validation_batch_size': validation_batch_size,
          'model_type': model_type,
          'random_state': random_state,
          'learning_rate': learning_rate,
          'verbose': verbose,
          'checkpoint_interval': checkpoint_interval,
          'log_path': log_path,
          'disable_logging': disable_logging,
          'gpus': gpus,
          'continue_model_dir': continue_model_dir,
          'git_commit': git.Repo(os.path.dirname(os.path.abspath(__file__)),
                                 search_parent_directories=True).head.object.hexsha,
          'gsheet_id': gsheet_id,
          'google_dev_app_name': google_dev_app_name
    }
    LOGGER.info('Training with the following arguments: {}'.format(param_dict))

    # TODO: Uncomment after modifying load_model() for DeepIoT
    '''if continue_model_dir:
        latest_model_path = os.path.join(continue_model_dir, 'model_latest.h5')
        m, inputs, outputs = load_model(latest_model_path, model_type, return_io=True, src_num_gpus=gpus)
    else:'''
    # Initialize L3 model (critic)
    m, inputs, outputs = MODELS[model_type](num_gpus=gpus) # Why is num_gpus inportant here?

    # Make sure the directories we need exist
    # TODO: Uncomment after modifying load_model() for DeepIoT
    '''if continue_model_dir:
        model_dir = continue_model_dir
    else:'''
    model_dir = os.path.join(output_dir, model_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    '''def lossloss(ytrue, ypred):
        batch_logits = binary_crossentropy(ytrue, ypred)
        return batch_logits
        #return K.mean(batch_logits, axis=-1)'''

    LOGGER.info('Compiling model...')

    m.compile(Adam(lr=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

    LOGGER.info('Model files can be found in "{}"'.format(model_dir))

    param_dict['model_dir'] = model_dir
    train_config_path = os.path.join(model_dir, 'config.json')
    with open(train_config_path, 'w') as fd:
        json.dump(param_dict, fd, indent=2)


    param_dict.update({
          'latest_epoch': '-',
          'latest_train_loss': '-',
          'latest_validation_loss': '-',
          'latest_train_acc': '-',
          'latest_validation_acc': '-',
          'best_train_loss': '-',
          'best_validation_loss': '-',
          'best_train_acc': '-',
          'best_validation_acc': '-',
    })

    # Save the model
    model_spec_path = os.path.join(model_dir, 'model_spec.pkl')
    model_spec = keras.utils.serialize_keras_object(m)
    with open(model_spec_path, 'wb') as fd:
        pickle.dump(model_spec, fd)
    model_json_path = os.path.join(model_dir, 'model.json')
    model_json = m.to_json()
    with open(model_json_path, 'w') as fd:
        json.dump(model_json, fd, indent=2)

    latest_weight_path = os.path.join(model_dir, 'model_latest.h5')
    best_valid_acc_weight_path = os.path.join(model_dir, 'model_best_valid_accuracy.h5')
    best_valid_loss_weight_path = os.path.join(model_dir, 'model_best_valid_loss.h5')
    checkpoint_weight_path = os.path.join(model_dir, 'model_checkpoint.{epoch:02d}.h5')

    # Load information about last epoch for initializing callbacks and data generators
    # TODO: Uncomment after modifying load_model() for DeepIoT
    '''if continue_model_dir is not None:
        prev_train_hist_path = os.path.join(continue_model_dir, 'history_csvlog.csv')
        last_epoch_idx, last_val_acc, last_val_loss = get_restart_info(prev_train_hist_path)'''

    # Set up callbacks
    cb = []
    cb.append(keras.callbacks.ModelCheckpoint(latest_weight_path,
                                              save_weights_only=True,
                                              verbose=1))

    best_val_acc_cb = keras.callbacks.ModelCheckpoint(best_valid_acc_weight_path,
                                              save_weights_only=True,
                                              save_best_only=True,
                                              verbose=1,
                                              monitor='val_acc')
    # TODO: Uncomment after modifying load_model() for DeepIoT
    '''if continue_model_dir is not None:
        best_val_acc_cb.best = last_val_acc'''
    cb.append(best_val_acc_cb)

    best_val_loss_cb = keras.callbacks.ModelCheckpoint(best_valid_loss_weight_path,
                                              save_weights_only=True,
                                              save_best_only=True,
                                              verbose=1,
                                              monitor='val_loss')
    # TODO: Uncomment after modifying load_model() for DeepIoT
    '''if continue_model_dir is not None:
        best_val_loss_cb.best = last_val_loss'''
    cb.append(best_val_loss_cb)

    checkpoint_cb = keras.callbacks.ModelCheckpoint(checkpoint_weight_path,
                                              save_weights_only=True,
                                              period=checkpoint_interval)
    # TODO: Uncomment after modifying load_model() for DeepIoT
    '''if continue_model_dir is not None:
        checkpoint_cb.epochs_since_last_save = (last_epoch_idx + 1) % checkpoint_interval'''
    cb.append(checkpoint_cb)

    timer_cb = TimeHistory()
    cb.append(timer_cb)

    history_checkpoint = os.path.join(model_dir, 'history_checkpoint.pkl')
    cb.append(LossHistory(history_checkpoint))

    history_csvlog = os.path.join(model_dir, 'history_csvlog.csv')
    cb.append(keras.callbacks.CSVLogger(history_csvlog, append=True,
                                        separator=','))

    if gsheet_id:
        cb.append(GSheetLogger(google_dev_app_name, gsheet_id, param_dict))

    LOGGER.info('Begin critic initialization phase')
    LOGGER.info('Setting up train data generator...')
    # TODO: Uncomment after modifying load_model() for DeepIoT
    '''if continue_model_dir is not None:
        train_start_batch_idx = train_epoch_size * (last_epoch_idx + 1)
    else:'''
    train_start_batch_idx = None

    train_gen = data_generator(
        train_data_dir,
        batch_size=train_batch_size,
        random_state=random_state,
        start_batch_idx=train_start_batch_idx)


    train_gen = pescador.maps.keras_tuples(train_gen,
                                          ['video', 'audio'],
                                          'label')

    LOGGER.info('Setting up validation data generator...')
    val_gen = single_epoch_data_generator(
        validation_data_dir,
        validation_epoch_size,
        batch_size=validation_batch_size,
        random_state=random_state)

    val_gen = pescador.maps.keras_tuples(val_gen,
                                         ['video', 'audio'],
                                         'label')

    # Fit the model
    LOGGER.info('Fitting model...')
    if verbose:
        verbosity = 1
    else:
        verbosity = 2

    # TODO: Uncomment after modifying load_model() for DeepIoT
    '''if continue_model_dir is not None:
        initial_epoch = last_epoch_idx + 1
    else:'''

    #TODO: UNCOMMENT THIS RIGHT AFTER TESTING
    '''initial_epoch = 0
    history = m.fit_generator(train_gen, train_epoch_size, num_epochs,
                              validation_data=val_gen,
                              validation_steps=validation_epoch_size,
                              #use_multiprocessing=True,
                              callbacks=cb,
                              verbose=verbosity,
                              initial_epoch=initial_epoch)

    # Save history
    with open(os.path.join(model_dir, 'history_initcritic.pkl'), 'wb') as fd:
        pickle.dump(history.history, fd)'''

    # Initialize compressor
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'audio_model/' in var.name]

    drop_prob_dict = compressor(d_vars)

    ###### Start Compressing
    print('Begin compressor-critic joint training phase')
    LOGGER.info('Setting up train data generator...')
    # TODO: Uncomment after modifying load_model() for DeepIoT
    '''if continue_model_dir is not None:
        train_start_batch_idx = train_epoch_size * (last_epoch_idx + 1)
    else:'''
    train_start_batch_idx = None

    train_gen = data_generator(
        train_data_dir,
        batch_size=train_batch_size,
        random_state=random_state,
        start_batch_idx=train_start_batch_idx)

    '''train_gen = pescador.maps.keras_tuples(train_gen,
                                           ['video', 'audio'],
                                           'label')'''

    LOGGER.info('Setting up validation data generator...')
    val_gen = single_epoch_data_generator(
        validation_data_dir,
        validation_epoch_size,
        batch_size=validation_batch_size,
        random_state=random_state)

    val_gen = pescador.maps.keras_tuples(val_gen,
                                         ['video', 'audio'],
                                         'label')

    # Fit the model
    LOGGER.info('Fitting models...')
    if verbose:
        verbosity = 1
    else:
        verbosity = 2

    with sess.as_default():
        thres_update_count = 0
        sess.run(tf.assign(sol_train, 0.0))
        for iteration in range(TOTAL_ITER_NUM):
            # Generate next batch
            cur_tr = next(train_gen)
            # Train Critic (batchwise)
            hist = m.fit(x={"audio": cur_tr['audio'], "video": cur_tr['video']}, y=cur_tr['label'],
                             batch_size=train_batch_size)

            # Roundabout way to get logits
            intermediate_layer_model = Model(inputs=m.input,
                                             outputs=m.layers[6].output)
            logits = tf.convert_to_tensor(intermediate_layer_model.predict(
                x={"audio": cur_tr['audio'], "video": cur_tr['video']}, batch_size=train_batch_size))

            labels = tf.cast(tf.convert_to_tensor(cur_tr['label']), tf.float32) # int64 is not permitted
            #predict = tf.argmax(logits,axis=-1)

            # Compute crossentropy
            batchLoss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits)
            batchLossMean, batchLossVar = tf.nn.moments(batchLoss, axes=[0])
            lossMean = tf.reduce_mean(batchLossMean)
            lossStd = tf.reduce_mean(tf.sqrt(batchLossVar))

            movingAvg_decay = 0.99
            ema = tf.train.ExponentialMovingAverage(0.9)
            maintain_averages_op = ema.apply([lossMean, lossStd])

            compsLoss = gen_compressor_loss(drop_prob_dict, out_binary_mask, batchLoss, ema, lossMean, lossStd)
            update_drop_op_dict = update_drop_op(drop_prob_dict, prob_list_dict)

            t_vars = tf.trainable_variables()
            # no_c_vars = [var for var in t_vars if not 'compressor/' in var.name]
            c_vars = [var for var in t_vars if 'compressor/' in var.name]

            compsOptimizer = tf.train.RMSPropOptimizer(0.001).minimize(compsLoss,
                                                                       var_list=c_vars, global_step=comps_global_step)

            left_num_dict = count_prun(prob_list_dict, prun_thres)
            #_, lossV, _trainY, _predict = sess.run([discOptimizer, loss, batch_label, predict])

            # Train Compressor
            _, compsLossV, _, lossMeanV, lossStdV = \
                sess.run([compsOptimizer, compsLoss, maintain_averages_op, ema.average(lossMean), ema.average(lossStd)])

            for layer_name in update_drop_op_dict.keys():
                sess.run(update_drop_op_dict[layer_name])

            if iteration % UPDATE_STEP == 0 and thres_update_count <= THRES_STEP:
                cur_thres = START_THRES + thres_update_count * (FINAL_THRES - START_THRES) / THRES_STEP
                print('Cur Threshold:', cur_thres)
                sess.run(tf.assign(prun_thres, cur_thres))
                thres_update_count += 1

            if iteration % 200 == 199:
                history = m.evaluate_generator(val_gen, steps=validation_epoch_size)
                dev_accuracy = np.mean(history.history['acc'])
                # TODO: Add validation here, depending on training time
                #dev_cross_entropy = []
                #for eval_idx in range(EVAL_ITER_NUM):
                #   eval_loss_v, _trainY, _predict = sess.run([loss_eval, batch_eval_label, predict_eval])
                #    _label = np.argmax(_trainY, axis=1)
                #    _accuracy = np.mean(_label == _predict)
                #    dev_accuracy.append(_accuracy)
                #    dev_cross_entropy.append(eval_loss_v)
                #plot.plot('dev accuracy', np.mean(dev_accuracy))
                #plot.plot('dev cross entropy', np.mean(dev_cross_entropy))
                cur_left_num = gen_cur_prun(sess, left_num_dict)
                print('Left Element in L3 embedding:', cur_left_num)
                cur_comps_ratio = compress_ratio(cur_left_num, org_dim_dict)
                if cur_comps_ratio < 7.0 and np.mean(dev_accuracy) >= 0.93:
                    break

    ###### Start Fine-Tuning critic
    print('\nBegin critic fine-tunning phase')
    sess.run(tf.assign(sol_train, 1.0))
    cur_left_num = gen_cur_prun(sess, left_num_dict)
    print('Compressed l3 embedding:', cur_left_num)
    compress_ratio(cur_left_num, org_dim_dict)
    LOGGER.info('Setting up train data generator...')
    # TODO: Uncomment after modifying load_model() for DeepIoT
    '''if continue_model_dir is not None:
        train_start_batch_idx = train_epoch_size * (last_epoch_idx + 1)
    else:'''
    train_start_batch_idx = None

    train_gen = data_generator(
        train_data_dir,
        batch_size=train_batch_size,
        random_state=random_state,
        start_batch_idx=train_start_batch_idx)

    train_gen = pescador.maps.keras_tuples(train_gen,
                                           ['video', 'audio'],
                                           'label')

    LOGGER.info('Setting up validation data generator...')
    val_gen = single_epoch_data_generator(
        validation_data_dir,
        validation_epoch_size,
        batch_size=validation_batch_size,
        random_state=random_state)

    val_gen = pescador.maps.keras_tuples(val_gen,
                                         ['video', 'audio'],
                                         'label')

    # Fit the model
    LOGGER.info('Fitting models...')
    if verbose:
        verbosity = 1
    else:
        verbosity = 2

    initial_epoch = 0
    history = m.fit_generator(train_gen, train_epoch_size, num_epochs,
                              validation_data=val_gen,
                              validation_steps=validation_epoch_size,
                              # use_multiprocessing=True,
                              callbacks=cb,
                              verbose=verbosity,
                              initial_epoch=initial_epoch)

    # Save history
    LOGGER.info('Done training. Saving results to disk...')
    with open(os.path.join(model_dir, 'history_finetunecritic.pkl'), 'wb') as fd:
        pickle.dump(history.history, fd)
    print('Final validation accuracy: ', history.history['val_acc'])

    '''for iteration in xrange(TOTAL_ITER_NUM):
        _, lossV, _trainY, _predict = sess.run([discOptimizer, loss, batch_label, predict])
        _label = np.argmax(_trainY, axis=1)
        _accuracy = np.mean(_label == _predict)
        plot.plot('train cross entropy', lossV)
        plot.plot('train accuracy', _accuracy)

        if iteration % 50 == 49:
            dev_accuracy = []
            dev_cross_entropy = []
            for eval_idx in xrange(EVAL_ITER_NUM):
                eval_loss_v, _trainY, _predict = sess.run([loss_eval, batch_eval_label, predict_eval])
                _label = np.argmax(_trainY, axis=1)
                _accuracy = np.mean(_label == _predict)
                dev_accuracy.append(_accuracy)
                dev_cross_entropy.append(eval_loss_v)
            plot.plot('dev accuracy', np.mean(dev_accuracy))
            plot.plot('dev cross entropy', np.mean(dev_cross_entropy))'''

    LOGGER.info('Done!')


# Module test
if __name__=='__main__':
    print('In main')

    train_data_dir = '../music_sample/train_samp'
    validation_data_dir = '../music_sample/valid_samp'
    output_dir = '../music_sample/out'
    num_epochs = 1
    # Two mini-batches in an epoch
    train_epoch_size = 1
    validation_epoch_size = 1
    train_batch_size = 64
    validation_batch_size = 64
    model_type = 'cnn_L3_melspec2'
    random_state = 20180123
    learning_rate = 1e-4
    verbose = False
    checkpoint_interval = 10
    log_path = None
    disable_logging = False
    gpus = 0
    continue_model_dir = None
    gsheet_id = None

    train(train_data_dir=train_data_dir,validation_data_dir=validation_data_dir,output_dir=output_dir,
          num_epochs=num_epochs, train_epoch_size=train_epoch_size, validation_epoch_size=validation_epoch_size,
          train_batch_size=train_batch_size, validation_batch_size=validation_batch_size,
          model_type=model_type, random_state=random_state, learning_rate=learning_rate, verbose=verbose)