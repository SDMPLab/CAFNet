from keras.callbacks import Callback, CallbackList, BaseLogger, History
import time
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.interpolate import griddata
from keras.models import load_model
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import json
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
from custom_layer import *
from custom_functions import *
import numpy as np
import pandas as pd
import copy
import warnings

np.random.seed(19)
if tf.__version__.startswith('1.'):
    tf.set_random_seed(19)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
else:
    tf.random.set_seed(19)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

K.set_image_data_format('channels_first')

custom_objects = {'cmse': cmse,
                  'cmc': cmc,
                  'cacc': cacc,
                  'custom_mean_squared_error': custom_mean_squared_error,
                  'custom_mse_cosine': custom_mse_cosine,
                  'custom_binary_accuracy': custom_binary_accuracy,
                  'wasserstein_loss': wasserstein_loss,
                  'wasserstein_accuracy': wasserstein_accuracy,
                  'leaky_relu': leaky_relu,
                  'l2_error_norm': l2_error_norm,
                  'multiply_constant': multiply_constant,
                  'multiply_constant_reciprocal': multiply_constant_reciprocal,
                  'AccumulatorCell': AccumulatorCell}


def register_custom_objects(name, value):
    global custom_objects
    custom_objects[name] = value


def angle_pos(s, offset=0, include_zero=False):
    ap = (np.arange(0, s) + offset) * (360 / s) % 360
    if not include_zero:
        ap = ap[1:]
    return ap


def get_s_d(ap):
    y = -32 * np.sin(ap * (np.pi / 180)) + 32
    x = 32 * np.cos(ap * (np.pi / 180)) + 32
    z = np.ones_like(y)
    s_d = np.array([y, x, z])
    srt = np.lexsort(s_d[-1::-1])
    return s_d, srt


def create_dict(keys, datasets, var_name, func=lambda a: a):
    ret_dict = {}
    for k in keys:
        ret_dict[k] = func(datasets[k][var_name])
    return ret_dict


def dicts_op(keys, dict_list, func=lambda a: a):
    ret_dict = {}
    for k in keys:
        if isinstance(dict_list, list):
            temp = []
            for d in dict_list:
                temp.append(d[k])
            ret_dict[k] = func(temp)
        else:
            ret_dict[k] = func(dict_list[k])
    return ret_dict


MAX_INC = 3


class Dataset:
    def __init__(self, params, name):
        self.name = name
        self.d = params['d'].astype(np.float).reshape((-1, 1))
        self.data_size = len(self.d)
        self.data_label = params.get('data_label', np.ones_like(d)).astype(np.int)
        self.val_size = len(np.where(self.data_label)[0])
        self.train_size = self.data_size - self.val_size
        self.r = params['r'].astype(np.float)
        self.roc = params['roc'].astype(np.float)
        self.theta = params['theta'].astype(np.float)
        self.ca = params['ca'].astype(np.float)
        self.csp = params['csp'].astype(np.float)
        self.count_inc_label = params['count_inc_label'].astype(np.int)
        self.group_inc_label = params['group_inc_label'].astype(np.int)
        self.count_label = self.count_inc_label[self.group_inc_label == 1]
        self.roc_data = np.zeros((self.data_size, MAX_INC))
        self.roc_data[self.count_label != 0, 0] = self.roc[(self.count_inc_label != 0) & (self.group_inc_label == 1)]
        self.roc_data[self.count_label == 2, 1] = self.roc[(self.count_inc_label == 2) & (self.group_inc_label == 2)]
        self.theta_data = np.zeros((self.data_size, MAX_INC))
        self.theta_data[self.count_label != 0, 0] = self.theta[(self.count_inc_label != 0) &
                                                               (self.group_inc_label == 1)]
        self.theta_data[self.count_label == 2, 1] = self.theta[
            (self.count_inc_label == 2) & (self.group_inc_label == 2)]
        self.r_data = np.zeros((self.data_size, MAX_INC))
        self.r_data[self.count_label != 0, 0] = self.r[(self.count_inc_label != 0) & (self.group_inc_label == 1)]
        self.r_data[self.count_label == 2, 1] = self.r[(self.count_inc_label == 2) & (self.group_inc_label == 2)]
        self.ca_data = np.zeros((self.data_size, MAX_INC))
        self.ca_data[self.count_label != 0, 0] = self.ca[(self.count_inc_label != 0) & (self.group_inc_label == 1)]
        self.ca_data[self.count_label == 2, 1] = self.ca[(self.count_inc_label == 2) & (self.group_inc_label == 2)]
        self.csp_data = np.zeros((self.data_size, MAX_INC))
        self.csp_data[self.count_label != 0, 0] = self.csp[(self.count_inc_label != 0) & (self.group_inc_label == 1)]
        self.csp_data[self.count_label == 2, 1] = self.csp[(self.count_inc_label == 2) & (self.group_inc_label == 2)]
        temp = np.flip(np.argsort(self.r_data, axis=1), axis=1)
        temp_func = lambda a: a[a[MAX_INC:].astype(np.int)]
        self.roc_data = np.apply_along_axis(temp_func, 1, np.concatenate([self.roc_data, temp], axis=1))
        self.theta_data = np.apply_along_axis(temp_func, 1, np.concatenate([self.theta_data, temp], axis=1))
        self.r_data = np.apply_along_axis(temp_func, 1, np.concatenate([self.r_data, temp], axis=1))
        self.ca_data = np.apply_along_axis(temp_func, 1, np.concatenate([self.ca_data, temp], axis=1))
        self.csp_data = np.apply_along_axis(temp_func, 1, np.concatenate([self.csp_data, temp], axis=1))
        self.N_count = params.get('N_count').astype(np.int)
        self.freq = params['freq'].astype(np.float).reshape((-1, 1))
        self.num = params['inc_sample_num'].astype(np.int) - 1
        _, self.indices = np.unique(self.num, return_index=True)
        self.inhomo_ct = (self.count_inc_label[self.indices] != 0).astype(np.float)
        self.MUa = params['MUa'].astype('float16').reshape((-1, 1))
        self.MUsp = params['MUsp'].astype('float16').reshape((-1, 1))
        self.MU0 = np.concatenate([self.MUa, self.MUsp], axis=1)
        self.inc_areas = None
        self.pos_roc = None
        self.pos_x = None
        self.pos_y = None
        self.pos_data = None
        self.ns = None
        self.MU = None
        self.MU_norm = None
        self.MU_iter_uncorrected = None
        self.MU_iter_corrected = None
        self.MU_deep = None

    def acknowledge_samples(self, samples):
        self.inc_areas = samples['inc_areas'].astype(np.float)
        self.pos_roc = 64 * self.roc_data / self.d
        self.pos_x = self.pos_roc * np.cos(self.theta_data * (np.pi / 180)) + 32
        self.pos_y = -self.pos_roc * np.sin(self.theta_data * (np.pi / 180)) + 32
        temp = np.sqrt(np.diff(self.pos_x) ** 2 + np.diff(self.pos_y) ** 2)
        temp = self.pos_roc[:, 1] > temp[:, 1]
        self.pos_roc[temp, 1] = 0
        self.pos_x[temp, 1] = 0
        self.pos_y[temp, 1] = 0
        self.inc_areas[temp, 0] += self.inc_areas[temp, 1]
        self.inc_areas[temp, 1] = 0
        self.pos_data = np.concatenate([self.pos_x[:, [0]] / 64, self.pos_y[:, [0]] / 64, self.pos_x[:, [1]] / 64,
                                        self.pos_y[:, [1]] / 64], axis=1)
        self.ns = samples['ns_used'].astype(np.int)
        self.MU = np.array([np.concatenate([[x], [y]])
                            for (x, y) in zip(samples['Grid_MUA'], samples['Grid_MUSP'])]).astype('float16')
        self.MU_norm = self.MU / np.reshape(self.MU0, self.MU0.shape + (1, 1)).astype('float16')

    def acknowledge_iter_results(self, iter_results):
        self.MU_iter_uncorrected = np.array([np.concatenate([[x], [y]])
                                             for (x, y) in zip(iter_results['Grid_MUA_res'],
                                                               iter_results['Grid_MUSP_res'])]).astype('float16')
        self.MU_iter_corrected = np.copy(self.MU_iter_uncorrected)
        self.MU_iter_corrected[self.MU_iter_corrected < 0] = 0
        self.MU_iter_corrected[self.MU_iter_corrected > 100] = 100

    def set_MU_deep(self, MU_deep):
        self.MU_deep = MU_deep


temp = np.linspace(-31.5, 31.5, 64) / 32
X, Y = np.meshgrid(temp, temp)
temp = np.linspace(-1, 1, 65)
X2, Y2 = np.meshgrid(temp, temp)
Xf = np.floor(np.abs(X * 32))
Yf = np.floor(np.abs(Y * 32))
boundary_mask = (Xf ** 2 + Yf ** 2 <= 32 ** 2) & ((Xf + 1) ** 2 + (Yf + 1) ** 2 >= 32 ** 2)
Xb = X[boundary_mask]
Yb = Y[boundary_mask]
Theta = np.arctan2(Y, X)
Theta = (Theta * 180 / np.pi) + 180
Thetab = Theta[boundary_mask]
idx = np.argsort(Thetab)
Xb = Xb[idx]
Yb = Yb[idx]
Thetab = Thetab[idx]
Ib = np.arange(len(Xb))
sum_bound = np.sum(boundary_mask)
stats_inc_keys = ('mean', 'max', 'min', 'std', 'median', 'q_p90', 'q_p10', 'tr_range', 'tr_mean')
channel_keys = ('mu_a', 'mu_sp')


def get_d_index(ap, indexing=None):
    isnan = np.isnan(ap)
    ap[isnan] = 0
    y = np.sin(ap * (np.pi / 180))
    x = np.cos(ap * (np.pi / 180))
    idx = griddata((Xb, Yb), Ib, (x, y), method='nearest')
    if indexing == 'pythonic':
        idx[isnan] = len(Xb)
    else:
        idx[isnan] = -1
    return idx


def get_s_d_locations(ns, spos=None, dpos=None):
    old_settings = np.seterr(invalid='ignore')
    max_ns_used = max(ns)
    if spos is None:
        spos, dpos = [], []
        for x in ns:
            t1 = np.arange(x)
            t3 = np.kron(t1, np.ones(x - 1))
            t4 = t3 + np.kron(np.ones(x), t1[1:])
            t3 = t3 * (360 / x) % 360
            t4 = t4 * (360 / x) % 360
            spos.append(t3)
            dpos.append(t4)
    max_len = max_ns_used * (max_ns_used - 1)
    temp3, temp4 = [], []
    for i, (t3, t4) in enumerate(zip(spos, dpos)):
        t3 = t3.astype(np.float)
        t4 = t4.astype(np.float)
        if len(t3) > max_len:
            t3 = t3[:0]
            t4 = t4[:0]
        t2 = np.zeros(max_len - len(t3)) * np.nan
        temp3.append(np.concatenate([t3, t2]))
        temp4.append(np.concatenate([t4, t2]))
    spos = get_d_index(np.array(temp3)) + 1
    dpos = get_d_index(np.array(temp4)) + 1
    spos = np.expand_dims(spos, -1)
    dpos = np.expand_dims(dpos, -1)
    s_d = np.concatenate([spos, dpos], axis=-1).astype(np.int)
    np.seterr(**old_settings)
    return s_d


def get_s_d_locations2(ns):
    old_settings = np.seterr(invalid='ignore')
    max_ns_used = max(ns)
    temp1 = np.array([np.linspace(0, 360, x, endpoint=False) for x in ns])
    temp2 = np.array([np.zeros(max_ns_used * (max_ns_used - 1) - x * (x - 1)) for x in ns]) * np.nan
    temp3 = np.array([np.concatenate([np.kron(t1, np.ones(x - 1)), t2]) for x, t1, t2 in zip(ns, temp1, temp2)])
    temp4 = np.array([np.concatenate([np.kron(np.ones(x), t1[1:]), t2]) for x, t1, t2 in zip(ns, temp1, temp2)])
    temp4 = (temp3 + temp4) % 360
    temp3 = temp3.reshape(temp3.shape + (1,))
    temp4 = temp4.reshape(temp4.shape + (1,))
    s_d = np.concatenate([temp3, temp4], axis=-1)
    y = -32 * np.sin(s_d * (np.pi / 180)) + 32
    x = 32 * np.cos(s_d * (np.pi / 180)) + 32
    temp5 = np.isnan(x[:, :, [0]]) is False
    s_d = np.concatenate([y[:, :, [0]], x[:, :, [0]], y[:, :, [1]], x[:, :, [1]], temp5], axis=-1)
    srt = np.array([np.lexsort(np.transpose(s[:, -1::-1])) for s in s_d])
    s_d[np.isnan(s_d)] = 0
    np.seterr(**old_settings)
    return s_d, srt


print('Loading datasets..')

test_names = {'uncalibrated', 'calibrated', 'simulation', 'uncalibrated_16', 'calibrated_16', 'simulation_16'}
exp_data_16 = loadmat('exp_data_rev_16.mat', squeeze_me=True)
exp_data = loadmat('exp_data.mat', squeeze_me=True)

def unwrap_sample(phs, ns):
    for i in range(ns):
        phs[i * (ns - 1):(i + 1) * (ns - 1)] = np.unwrap(phs[i * (ns - 1):(i + 1) * (ns - 1)], axis=0)
    return phs


def add_measurement_dataset(name):
    max_ns = max(ns[name])
    max_data_len = max_ns * (max_ns - 1)
    data = []
    for i, y in enumerate(PHI_meas_lookup[name]):
        a = y.astype(np.complex)
        if len(a) > max_data_len:
            a = a[:0]
        b = np.ones(max_data_len - len(a))
        data.append(np.concatenate([a, b]).reshape((-1, 1)))
    data = np.array(data)

    if name in source_pos:
        s_d = get_s_d_locations(ns[name], source_pos[name], detector_pos[name])
    else:
        s_d = get_s_d_locations(ns[name])
    # s_d = get_s_d_locations(ns[name])
    PHI_meas_raw_seq_pair[name] = data
    log_data = np.log(data)
    phase_data = np.imag(log_data)
    for i, s in enumerate(ns[name]):
        phase_data[i] = unwrap_sample(phase_data[i], s)
    PHI_meas_seq_pair[name] = np.concatenate([np.real(log_data), phase_data, np.array(s_d)], axis=-1).astype('float16')
    if name != 'training':
        data = np.squeeze(data)
        PHI_meas_raw_fixed[name] = data
        log_data = np.squeeze(log_data)
        phase_data = np.squeeze(phase_data)
        PHI_meas_fixed[name] = np.concatenate([np.real(log_data), phase_data], axis=-1).astype('float16')


def add_training_dataset(name, samples_in, param_samples_in, iter_results_in):
    if isinstance(samples_in, str):
        samples_in = loadmat(samples_in, squeeze_me=True)
    if isinstance(param_samples_in, str):
        param_samples_in = loadmat(param_samples_in, squeeze_me=True)
    if isinstance(iter_results_in, str):
        iter_results_in = loadmat(iter_results_in, squeeze_me=True)
    samples[name] = samples_in
    param_samples[name] = param_samples_in
    iter_results[name] = iter_results_in
    datasets[name] = Dataset(param_samples[name], name)
    datasets[name].acknowledge_samples(samples[name])
    datasets[name].acknowledge_iter_results(iter_results[name])
    d[name] = datasets[name].d
    freq[name] = datasets[name].freq
    ns[name] = datasets[name].ns
    roc_data[name] = datasets[name].roc_data
    theta_data[name] = datasets[name].theta_data
    r_data[name] = datasets[name].r_data
    ca_data[name] = datasets[name].ca_data
    csp_data[name] = datasets[name].csp_data
    MUa[name] = datasets[name].MUa
    MUsp[name] = datasets[name].MUsp
    MU0[name] = datasets[name].MU0
    MU[name] = datasets[name].MU
    MU_norm[name] = datasets[name].MU_norm
    MU_norm[name] = datasets[name].MU_norm
    MU_iter_corrected[name] = datasets[name].MU_iter_corrected
    MU_iter_uncorrected[name] = datasets[name].MU_iter_uncorrected
    PHI_meas_lookup[name] = samples[name]['PHI_meas']
    add_measurement_dataset(name)

train_names, samples, param_samples, iter_results = {}, {}, {}, {}
datasets, d, freq, ns, roc_data, theta_data, r_data, ca_data, csp_data = {}, {}, {}, {}, {}, {}, {}, {}, {}
MUa, MUsp, MU0, MU, MU_norm, MU_iter_corrected, MU_iter_uncorrected = {}, {}, {}, {}, {}, {}, {}
PHI_meas_raw_seq_pair, PHI_meas_seq_pair, PHI_meas_raw_fixed, PHI_meas_fixed = {}, {}, {}, {}
print('Create dictionaries..')
MU_iter_uncorrected = {'uncalibrated': np.array([np.concatenate([[x], [y]])
                                                 for (x, y) in zip(exp_data['Grid_MUA_res'],
                                                                   exp_data['Grid_MUSP_res'])]).astype('float16'),
                       'calibrated': np.array([np.concatenate([[x], [y]])
                                               for (x, y) in zip(exp_data['Grid_MUA_res_clb'],
                                                                 exp_data['Grid_MUSP_res_clb'])]).astype('float16'),
                       'simulation': np.array([np.concatenate([[x], [y]])
                                               for (x, y) in zip(exp_data['Grid_MUA_res_sim'],
                                                                 exp_data['Grid_MUSP_res_sim'])]).astype('float16'),
                       'uncalibrated_16': np.array([np.concatenate([[x], [y]])
                                                    for (x, y) in zip(exp_data_16['Grid_MUA_res'],
                                                                      exp_data_16['Grid_MUSP_res'])]).astype('float16'),
                       'calibrated_16': np.array([np.concatenate([[x], [y]])
                                                  for (x, y) in zip(exp_data_16['Grid_MUA_res_clb'],
                                                                    exp_data_16['Grid_MUSP_res_clb'])]).astype('float16'),
                       'simulation_16': np.array([np.concatenate([[x], [y]])
                                                  for (x, y) in zip(exp_data_16['Grid_MUA_res_sim'],
                                                                    exp_data_16['Grid_MUSP_res_sim'])]).astype('float16')}
MU_iter_uncorrected['uncalibrated2_16'] = MU_iter_uncorrected['uncalibrated_16']
MU_iter_uncorrected['calibrated2_16'] = MU_iter_uncorrected['calibrated_16']
MU_iter_uncorrected['simulation2_16'] = MU_iter_uncorrected['simulation_16']
MU_iter_uncorrected['uncalibrated3_16'] = MU_iter_uncorrected['uncalibrated_16']
MU_iter_uncorrected['calibrated3_16'] = MU_iter_uncorrected['calibrated_16']
MU_iter_uncorrected['simulation3_16'] = MU_iter_uncorrected['simulation_16']

PHI_meas_lookup = {'uncalibrated': exp_data['PHI_meas'], 'calibrated': exp_data['PHI_CLB_meas'],
                   'simulation': exp_data['PHI_meas_sim'], 'uncalibrated_16': exp_data_16['PHI_meas'],
                   'calibrated_16': exp_data_16['PHI_CLB_meas'], 'simulation_16': exp_data_16['PHI_meas_sim']}



source_pos = {'uncalibrated_16': exp_data_16['source_pos'], 'calibrated_16': exp_data_16['source_pos'],
              'simulation_16': exp_data_16['source_pos_sim']}
detector_pos = {'uncalibrated_16': exp_data_16['detector_pos'], 'calibrated_16': exp_data_16['detector_pos'],
                'simulation_16': exp_data_16['detector_pos_sim']}

add_training_dataset('trainingtata', 'samples_500.mat', 'param_samples_5001.mat', 'iterative_results_500.mat')

temp_lookup = {'exp': exp_data, 'exp_16': exp_data_16}
temp_d, temp_freq, temp_ns = {}, {}, {}
temp_roc_data, temp_theta_data, temp_r_data, temp_ca_data, temp_csp_data = {}, {}, {}, {}, {}
temp_MUa, temp_MUsp, temp_MU = {}, {}, {}
for k, v in temp_lookup.items():
    temp_len = len(v['MUa'])
    temp_d[k] = v['d'].astype('float16').reshape((-1, 1))
    if k == 'exp':
        temp_freq[k] = np.ones((temp_len, 1)) * 20e6
        temp_ns[k] = np.ones(temp_len, dtype=np.int) * 16
    else:
        temp_freq[k] = v['freq'].astype(np.float).reshape((-1, 1))
        temp_ns[k] = np.ones(temp_len, dtype=np.int) * int(k[-2:])
    temp_roc_data[k] = np.zeros((temp_len, MAX_INC))
    temp_theta_data[k] = np.zeros((temp_len, MAX_INC))
    temp_r_data[k] = np.zeros((temp_len, MAX_INC))
    temp_ca_data[k] = np.zeros((temp_len, MAX_INC))
    temp_csp_data[k] = np.zeros((temp_len, MAX_INC))
    for i, (t_roc, t_theta, t_r, t_ca, t_csp) in enumerate(zip(v['roc'], v['theta'], v['r'], v['ca'], v['csp'])):
        try:
            l = len(t_roc)
        except TypeError:
            l = 1
        t_roc = np.array(t_roc, dtype=np.float).flatten()
        t_theta = np.array(t_theta, dtype=np.float).flatten()
        t_r = np.array(t_r, dtype=np.float).flatten()
        t_ca = np.array(t_ca, dtype=np.float).flatten()
        t_csp = np.array(t_csp, dtype=np.float).flatten()
        temp_roc_data[k][i, :l] = t_roc
        temp_theta_data[k][i, :l] = t_theta
        temp_r_data[k][i, :l] = t_r
        temp_ca_data[k][i, :l] = t_ca
        temp_csp_data[k][i, :l] = t_csp
        temp = np.flip(np.argsort(temp_r_data[k], axis=1), axis=1)
        temp_func = lambda a: a[a[MAX_INC:].astype(np.int)]
        temp_roc_data[k] = np.apply_along_axis(temp_func, 1, np.concatenate([temp_roc_data[k], temp], axis=1))
        temp_theta_data[k] = np.apply_along_axis(temp_func, 1, np.concatenate([temp_theta_data[k], temp], axis=1))
        temp_r_data[k] = np.apply_along_axis(temp_func, 1, np.concatenate([temp_r_data[k], temp], axis=1))
        temp_ca_data[k] = np.apply_along_axis(temp_func, 1, np.concatenate([temp_ca_data[k], temp], axis=1))
        temp_csp_data[k] = np.apply_along_axis(temp_func, 1, np.concatenate([temp_csp_data[k], temp], axis=1))
    temp_MUa[k] = v['MUa'].astype('float16').reshape((-1, 1))
    temp_MUsp[k] = v['MUsp'].astype('float16').reshape((-1, 1))
    temp_MU[k] = np.array([np.concatenate([[x], [y]]) for (x, y) in zip(v['Grid_MUA'], v['Grid_MUSP'])])
for k in test_names:
    if k.endswith('_16'):
        k2 = 'exp_16'
    else:
        k2 = 'exp'
    d[k] = temp_d[k2]
    freq[k] = temp_freq[k2]
    ns[k] = temp_ns[k2]
    roc_data[k] = temp_roc_data[k2]
    theta_data[k] = temp_theta_data[k2]
    r_data[k] = temp_r_data[k2]
    ca_data[k] = temp_ca_data[k2]
    csp_data[k] = temp_csp_data[k2]
    MUa[k] = temp_MUa[k2].astype('float16')
    MUsp[k] = temp_MUsp[k2].astype('float16')
    MU0[k] = np.concatenate([MUa[k], MUsp[k]], axis=1).astype('float16')
    MU[k] = temp_MU[k2].astype('float16')
    MU_norm[k] = MU[k] / np.reshape(MU0[k], MU0[k].shape + (1, 1)).astype('float16')
    MU_iter_corrected[k] = MU_iter_uncorrected[k].copy()
    MU_iter_corrected[k][MU_iter_corrected[k] < 0] = 0
    MU_iter_corrected[k][MU_iter_corrected[k] > 100] = 100
    add_measurement_dataset(k)
MU_iter = MU_iter_corrected
MU0_deep = MU0.copy()

mask = loadmat('mask.mat')['mask'].astype(np.bool)
empty_image = mask / mask * ~mask

c1_kmeans = 1
c2_kmeans = 100
X_kmeans = (X[mask].reshape(-1, 1) + 1) * c1_kmeans
Y_kmeans = (Y[mask].reshape(-1, 1) + 1) * c1_kmeans
boundary_kmeans = (mask & boundary_mask)[mask]
PHI_0_flag, n_flag, PHI_meas_seq_pair_masked = {}, {}, {}
for k, v in PHI_meas_seq_pair.items():
    s = v.shape
    PHI_0_flag[k] = np.zeros(s[:2], dtype=np.bool)
    s2 = ns[k] * (ns[k] - 1)
    n_flag[k] = (s2 * np.random.random(size=ns[k].shape) * 0.2).astype(np.int)
    for i in range(s[0]):
        PHI_0_flag[k][i, np.random.randint(s2[i], size=n_flag[k][i])] = 1
    PHI_meas_seq_pair_masked[k] = np.copy(v)
    PHI_meas_seq_pair_masked[k][PHI_0_flag[k], :] = 0
print('Dictionaries created.')
print('Prepare measurement data..')

model = None
input_for_model = None
hist_data = None
PREPROCESSING_MODE = None
PHI_meas_raw = None
PHI_meas = None
PHI_meas_16x15 = None
seq_len = None
p_theta = np.linspace(0, 2 * np.pi, 181)
MU_deep = {}
key_1 = 'trainingtata'

noise_max = np.max(param_samples[key_1]['t_noise']) / 100
noise_add_max = 10 ** (np.max(-param_samples[key_1]['t_noise_db']) / 20)
const_freq = np.mean(param_samples[key_1]['t_freq'])
const_d = np.mean(param_samples[key_1]['t_d'])



print('All datasets loaded.')


def set_preprocessing_mode(mode):
    global PREPROCESSING_MODE, PHI_meas_raw, PHI_meas, seq_len
    if mode == 'FIXED':
        PHI_meas_raw = PHI_meas_raw_fixed
        PHI_meas = PHI_meas_fixed
        seq_len = {}
        for k, v in ns.items():
            seq_len[k] = v * (v - 1) * 2
    elif mode == 'SEQUENTIAL':
        PHI_meas_raw = PHI_meas_raw_seq_pair
        PHI_meas = PHI_meas_seq_pair
        seq_len = {}
        for k, v in ns.items():
            seq_len[k] = v * (v - 1)
    else:
        raise Exception('Preprocessing mode not allowed.')
    PREPROCESSING_MODE = mode



set_preprocessing_mode('FIXED')

PHI_meas_16x15 = dict()
for x, j in PHI_meas_fixed.items():
    PHI_meas_16x15[x]=j.reshape(-1,2,16,15)



def get_raw_measurements(meas):
    if PREPROCESSING_MODE == 'FIXED':
        real = meas[:, :(meas.shape[1] // 2)]
        imag = meas[:, (meas.shape[1] // 2):]
    # elif PREPROCESSING_MODE == 'SEQUENTIAL_SOURCE':
    #     real = meas[:, :, :(ns_target - 1)]
    #     imag = meas[:, :, (ns_target - 1):((ns_target - 1) * 2)]
    elif PREPROCESSING_MODE == 'SEQUENTIAL':
        real = meas[:, :, 0:1]
        imag = meas[:, :, 1:2]
    else:
        raise Exception('Preprocessing mode not allowed.')
    return np.exp(real + imag * 1j)


def set_raw_measurements(meas, meas_raw):
    log_meas_raw = np.log(meas_raw)
    if PREPROCESSING_MODE == 'FIXED':
        meas[:, :(meas.shape[1] // 2)] = np.real(log_meas_raw)
        meas[:, (meas.shape[1] // 2):] = np.imag(log_meas_raw)
    elif PREPROCESSING_MODE == 'SEQUENTIAL':
        mask = np.all(meas[:, :, 2:], axis=-1, keepdims=True)
        meas[:, :, 0:1] = meas[:, :, 0:1] * (1 - mask) + np.real(log_meas_raw) * mask
        meas[:, :, 1:2] = meas[:, :, 1:2] * (1 - mask) + np.imag(log_meas_raw) * mask
    else:
        raise Exception('Preprocessing mode not allowed.')
    return meas


def correct_MU_iter(b):
    global MU_iter
    if b:
        MU_iter = MU_iter_corrected
    else:
        MU_iter = MU_iter_uncorrected


def mask_image(image, mask_value):
    image = image.copy()
    image[~mask] = mask_value
    return image



def get_current_results():
    return MU_deep


def get_model():
    return model


def set_model(param_model):
    global model
    if param_model is not model:
        if isinstance(param_model, str):
            model = load_model(param_model, custom_objects=custom_objects)
        elif param_model is not None:
            model = param_model


def set_weights(weights):
    global model
    if isinstance(weights, str):
        model.load_weights(weights)
    else:
        model.set_weights(weights)


def set_hist_data(hist_data_input):
    global hist_data
    if hist_data_input is not hist_data:
        if isinstance(hist_data_input, str):
            with open(hist_data_input, 'r') as f:
                hist_data = json.load(f)
        else:
            hist_data = hist_data

