import tensorflow as tf
from tensorflow.keras import layers, models
from helper1516 import *
np.random.seed(19)
from sklearn.model_selection import train_test_split
import numpy as np
from keras import backend
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns


backend.set_image_data_format('channels_first')
print(backend.image_data_format())
smooth = 1.
dropout_rate = 0.5
act = "relu"
###############################################################################

def fn_mask(x):
    from scipy.io import loadmat
    mask = loadmat('mask.mat')['mask'].astype(np.float)
    return multiply_constant(x, mask)
##########################################################################################################
from keras import backend
backend.set_image_data_format('channels_first')
print(backend.image_data_format())

lr = 0.001
beta_1 = 0.5

x0_20_2d_exp = PHI_meas_16x15[key_1]
y0_20_2d_exp = MU[key_1]

total_num_20_2d_exp = x0_20_2d_exp.shape[0]
indices_20_2d_exp = np.arange(total_num_20_2d_exp)

# === Define percentage splits ===
test_split = 0.05 
val_split = 0.1    

X_train_20_2d_exp, x_test_20_2d_exp, Y_train_20_2d_exp, y_test_20_2d_exp, id_train_20_2d_exp, id_test_20_2d_exp = train_test_split(
    x0_20_2d_exp, y0_20_2d_exp, indices_20_2d_exp, test_size=test_split, random_state=1
)

x_train_20_2d_exp, x_val_20_2d_exp, y_train_20_2d_exp, y_val_20_2d_exp, id_train_20_2d_exp, id_val_20_2d_exp = train_test_split(
    X_train_20_2d_exp, Y_train_20_2d_exp, id_train_20_2d_exp, test_size=val_split / (1 - test_split), random_state=2
)

x_train_20_2d_exp = [x_train_20_2d_exp]
x_val_20_2d_exp = [x_val_20_2d_exp]
x_test_20_2d_exp = [x_test_20_2d_exp]

y_train_20_2d_exp = [y_train_20_2d_exp[:, [0]], y_train_20_2d_exp[:, [1]]]
y_val_20_2d_exp = [y_val_20_2d_exp[:, [0]], y_val_20_2d_exp[:, [1]]]
y_test_20_2d_exp = [y_test_20_2d_exp[:, [0]], y_test_20_2d_exp[:, [1]]]

PHI_meas_in_20_2d_exp = PHI_meas_16x15[key_1]
input_shape_20_2d_exp = PHI_meas_in_20_2d_exp.shape[1:]
input_20_2d_exp = Input(input_shape_20_2d_exp)

print("Total samples:", total_num_20_2d_exp)
print(f"Training: {len(x_train_20_2d_exp[0])}, Validation: {len(x_val_20_2d_exp[0])}, Test: {len(x_test_20_2d_exp[0])}")


def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):
    x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Activation("relu")(x)
    x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Activation("relu")(x)
    return x
def fusion_module(x, num_filters, chanDim):

    conv_1x1 = conv_module(x, num_filters, 1, 1, (1, 1), chanDim)

    conv_3x3 = conv_module(x, num_filters, 3, 3, (1, 1), chanDim)

    conv_5x5 = conv_module(x, num_filters, 5, 5, (1, 1), chanDim)

    pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(num_filters, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(pool_proj)

    x = layers.Concatenate(axis=chanDim)([conv_1x1, conv_3x3, conv_5x5, pool_proj])
    return x
def channel_attention_block(x, ratio=16):
     
     channel = x.shape[1]


     avg_pool = GlobalAveragePooling2D()(x)
     avg_pool = Reshape((1, 1, channel))(avg_pool)
     

     max_pool = GlobalMaxPooling2D()(x)
     max_pool = Reshape((1, 1, channel))(max_pool)
     

     shared_layer_one = Conv2D(channel // ratio, (1, 1), activation='relu', kernel_initializer='he_normal', use_bias=True)
     shared_layer_two = Conv2D(channel, (1, 1), activation='sigmoid', kernel_initializer='he_normal', use_bias=True)

     avg_out = shared_layer_two(shared_layer_one(avg_pool))
     max_out = shared_layer_two(shared_layer_one(max_pool))

     cbam_out = Add()([avg_out, max_out])
     x_out = Multiply()([x, cbam_out])
     if return_weights:
        return x_out, cbam_out
     else:
        return x_out
def feature_extraction_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x
def Build_model(input_, num_class=1, return_attention=True):
    x = Conv2D(8, 3, padding='same', activation="softplus", data_format='channels_first')(input_)
    x = BatchNormalization(axis=1)(x)
    x = Flatten()(x)

    x = Dense(4096, activation="softplus")(x)
    x = BatchNormalization(axis=1)(x)
    x1 = Reshape((1, 64, 64))(x)  # Ensure total elements match after reshape
    
    base_filters = 64
    chanDim = 1  # Channels-first data format
    x1 = feature_extraction_block(x1, base_filters)

    for i in range(6):
      x1 = fusion_module(x1, base_filters // 4, chanDim)

    x1 = channel_attention_block(x1)

    x1 = Conv2D(1, (1, 1), activation='sigmoid', name='output', 

                 kernel_initializer='he_normal', padding='same')(x1) 

    x_1 = Lambda(fn_mask)(x1) 

    x_1 = BatchNormalization(axis=1)(x_1) 

    x_2 = Lambda(fn_mask)(x1) 

    x_2 = BatchNormalization(axis=1)(x_2) 

    model = models.Model(input_, outputs=[x_1, x_2]) 

    return model 

model = Build_model(input_20_2d_exp)
model.summary()

metrics = ['mae', 'mse', 'mape', 'msle', 'logcosh', 'cosine_similarity']
model.compile(optimizer="adam",
loss="mse",
metrics=metrics)


history=model.fit(x_train_20_2d_exp, y_train_20_2d_exp, validation_data=(x_val_20_2d_exp, y_val_20_2d_exp), batch_size=64, epochs=100, verbose=1)
model_name= 'CAFNet'
model.save(model_name + '.h5')


PHI_meas_16x15_={}
PHI_meas_16x15_[key_1]=PHI_meas_16x15[key_1]

MU_deep_De = {}

import keras
loaded_2 = keras.models.load_model("CAFNet.h5")
model_20_de = loaded_2


def obtain_results3(param_model=model_20_de, param_input=input_for_model, weights=None, sel=slice(2), sel_bg=slice(4, 6),
                   print_summary=False, exclude_keys=[]):
    global MU_deep_De, deep_output, input_for_model
    set_model(param_model)
    if weights is not None:
        set_weights(weights)
    if param_input is not input_for_model:
        if isinstance(param_input, list):
            input_for_model = {}
            for k in param_input[0].keys():
                if k in exclude_keys:
                    continue
                input_for_model[k] = [inp[k] for inp in param_input]
        else:
            input_for_model = param_input.copy()
            for k in exclude_keys:
                input_for_model.pop(k)
    deep_output = {}
    for k, v in input_for_model.items():
        deep_output[k] = model_20_de.predict(v)
    if print_summary:
        model_20_de.summary()
    if isinstance(deep_output[list(deep_output.keys())[0]], list):
        for k, v in deep_output.items():
            if isinstance(sel, int):
                MU_deep_De[k] = v[sel]
            elif isinstance(sel, slice):
                MU_deep_De[k] = np.concatenate(v[sel], axis=1)
            else:
                MU_deep_De[k] = np.concatenate([v[i] for i in sel], axis=1)
            if isinstance(sel_bg, int):
                MU0_deep[k] = v[sel_bg]
            elif isinstance(sel_bg, slice):
                MU0_deep[k] = np.concatenate(v[sel_bg], axis=1)
            else:
                MU0_deep[k] = np.concatenate([v[i] for i in sel_bg], axis=1)
    else:
        MU_deep_De = deep_output
    for k in deep_output.keys():
        if k in train_names:
            datasets[k].set_MU_deep(MU_deep_De[k])
    return MU_deep_De

MU_deep_De = obtain_results3(model_20_de,param_input=[PHI_meas_16x15_],sel_bg=1)


def plot_result(s_num, dataset_name='trainingtata', color_range='normal', vmin=None, vmax=None,
                roc=None, roc_index=None, correction=True, contrast=False, title=None):
    if correction:
        MU_iter_ = MU_iter_corrected
    else:
        MU_iter_ = MU_iter_uncorrected
    if not isinstance(vmin, list) or len(vmin) != 2:
        vmin = [None, None]
    if not isinstance(vmax, list) or len(vmax) != 2:
        vmax = [None, None]
    if title is None:
        title = ''
    title = str(title)
    scale = d[dataset_name][s_num] / 2
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 10), gridspec_kw={'hspace': 0.1, 'wspace': 0.1})
    plt.rcParams.update({'font.size': 18})
    fig.suptitle(title,fontsize=10)
    plt.rcParams.update({'font.size': 10})
    axes = axes.flatten()
    for ax in axes:
        ax.set_aspect('equal')
    image_dense = MU_deep_De[dataset_name][s_num]
    truth_image = MU[dataset_name][s_num]
    standard_image = MU_iter_[dataset_name][s_num]
    try:
        image_1d = MU_deep_De[dataset_name][s_num]
    except:
        temp = mask * np.nan
        deep_image = np.concatenate([[temp], [temp]])
    if contrast:
        MU0_ = np.reshape(MU0[dataset_name][s_num], MU0[dataset_name][s_num].shape + (1, 1))
        MU0_deep_ = np.reshape(MU0_deep[dataset_name][s_num], MU0_deep[dataset_name][s_num].shape + (1, 1))
        truth_image /= MU0_
        standard_image /= MU0_
        image_1d /= MU0_deep_

    images = [axes[0].pcolormesh(X2 * scale, Y2 * scale, mask_image(truth_image[0], np.nan), cmap='inferno'),
              axes[1].pcolormesh(X2 * scale, Y2 * scale, mask_image(standard_image[0], np.nan), cmap='inferno'),
              axes[2].pcolormesh(X2 * scale, Y2 * scale, mask_image(image_dense[0], np.nan), cmap='inferno')]

    axes[0].set_title('Ground Truth',fontsize=15)
    axes[1].set_title('TR',fontsize=15)
    axes[2].set_title('CAFNet',fontsize=15)

    axes[0].set_ylabel(r'$\mu_a$', fontsize=20)
    if color_range == 'normal':
        vmin = [0, 0]
        vmax = [np.max(mask_image(truth_image[i], -np.inf)) * 2 for i in range(2)]
    elif color_range == 'auto':
        vmin = [min(np.min(mask_image(truth_image[i], np.inf)), np.min(mask_image(standard_image[i], np.inf)),
                     np.min(mask_image(image_dense[i], np.inf))) for i in range(2)]
        vmax = [max(np.max(mask_image(truth_image[i], -np.inf)), np.max(mask_image(standard_image[i], -np.inf)),
                     np.max(mask_image(image_dense[i], -np.inf))) for i in range(2)]
    if color_range is None:
        fig.colorbar(images[0], ax=axes[0], orientation='horizontal', fraction=.1)
        fig.colorbar(images[1], ax=axes[1], orientation='horizontal', fraction=.1)
        fig.colorbar(images[2], ax=axes[2], orientation='horizontal', fraction=.1)

    else:
        for im in images:
            im.set_clim(vmin=vmin[0], vmax=vmax[0])
        fig.colorbar(images[0], ax=axes[0:3], orientation='horizontal', fraction=.1)
    images = [axes[3].pcolormesh(X2 * scale, Y2 * scale, mask_image(truth_image[1], np.nan), cmap='jet'),
              axes[4].pcolormesh(X2 * scale, Y2 * scale, mask_image(standard_image[1], np.nan), cmap='jet'),
              axes[5].pcolormesh(X2 * scale, Y2 * scale, mask_image(image_dense[1], np.nan),
                                 cmap='jet')]
    axes[3].set_ylabel(r'$\mu^\prime_s$',labelpad=0.9, fontsize=20)
    if color_range is None:
        fig.colorbar(images[0], ax=axes[3], orientation='horizontal', fraction=.1)
        fig.colorbar(images[1], ax=axes[4], orientation='horizontal', fraction=.1)
        fig.colorbar(images[2], ax=axes[5], orientation='horizontal', fraction=.1)
    else:
        for im in images:
            im.set_clim(vmin=vmin[1], vmax=vmax[1])
        fig.colorbar(images[0], ax=axes[3:6], orientation='horizontal', fraction=.1)
    if roc_index is not None:
        roc = roc_data[dataset_name][s_num, roc_index]
    if roc is not None:
        for ax in axes[:3]:
            ax.plot(roc * np.cos(p_theta), roc * np.sin(p_theta), c='#00ff00')
        for ax in axes[3:]:
            ax.plot(roc * np.cos(p_theta), roc * np.sin(p_theta), c='#000000')
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        

for i in [key]:

    plot_result(i,'trainingtata',None)
