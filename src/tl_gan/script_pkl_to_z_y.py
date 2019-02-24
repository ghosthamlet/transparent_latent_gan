import os
import glob
import pickle
import numpy as np
import PIL.Image
import h5py
import sys
import pandas as pd

import tensorflow.keras
import tensorflow.keras.applications
import tensorflow.keras.layers as layers
from tensorflow.keras.applications.mobilenet import preprocess_input

""" make sure this notebook is running from root directory """
while os.path.basename(os.getcwd()) in ('notebooks', 'src'):
    os.chdir('..')
assert ('README.md' in os.listdir('./')), 'Can not find project root, please cd to project root before running the following code'

path_model_save = 'asset_model/cnn_face_attr_celeba'

def create_cnn_model(size_output=40, tf_print=False):
    """
    create keras model with convolution layers of MobileNet and added fully connected layers on to top
    :param size_output: number of nodes in the output layer
    :param tf_print:    True/False to print
    :return: keras model object
    """

    if size_output is None:
        # get number of attrubutes, needed for defining the final layer size of network
        df_attr = pd.read_csv(path_celeba_att, sep='\s+', header=1, index_col=0)
        size_output = df_attr.shape[1]

    # Load the convolutional layers of pretrained model: mobilenet
    base_model = tensorflow.keras.applications.mobilenet.MobileNet(include_top=False, input_shape=(128,128,3),
                                                          alpha=1, depth_multiplier=1,
                                                          dropout=0.001, weights="imagenet",
                                                          input_tensor=None, pooling=None)

    # add fully connected layers
    fc0 = base_model.output
    fc0_pool = layers.GlobalAveragePooling2D(data_format='channels_last', name='fc0_pool')(fc0)
    fc1 = layers.Dense(256, activation='relu', name='fc1_dense')(fc0_pool)
    fc2 = layers.Dense(size_output, activation='tanh', name='fc2_dense')(fc1)

    model = tensorflow.keras.models.Model(inputs=base_model.input, outputs=fc2)

    # freeze the early layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='sgd', loss='mean_squared_error')

    if tf_print:
        print('use convolution layers of MobileNet, add fully connected layers')
        print(model.summary())

    return model

def get_list_model_save(path_model_save=path_model_save):
    return glob.glob(os.path.join(path_model_save, 'model*.h5'))

##
""" load model for prediction """
model = create_cnn_model(size_output=40)
model.load_weights(get_list_model_save()[-1])

# path to model generated results
path_gan_sample_pkl = './asset_results/stylegan_ffhq_sample_pkl/'
path_gan_sample_img = './asset_results/stylegan_ffhq_sample_jpg/'

if not os.path.exists(path_gan_sample_pkl):
    os.mkdir(path_gan_sample_pkl)

if not os.path.exists(path_gan_sample_img):
    os.mkdir(path_gan_sample_img)

# name of new data files
def get_filename_from_idx(idx):
    return 'sample_{:0>6}'.format(idx)

filename_sample_z = 'stylegan_sample_z.h5'
filename_sample_y = 'stylegan_sample_y.h5'

pathfile_y = os.path.join(path_gan_sample_img, filename_sample_y)
pathfile_z = os.path.join(path_gan_sample_img, filename_sample_z)

list_y = []
list_z = []

# get the pkl file list
list_pathfile_pkl = glob.glob(os.path.join(path_gan_sample_pkl, '*.pkl'))
list_pathfile_pkl.sort()

# loop to transform data and save image

i_counter = 0
for pathfile_pkl in list_pathfile_pkl:
    if i_counter % 100 == 0:
        print(pathfile_pkl)
    with open(pathfile_pkl, 'rb') as f:
        pkl_content = pickle.load(f)
    x = pkl_content['x']
    z = pkl_content['z']
    i_counter += x.shape[0]
    
    img_batch = np.stack(x, axis=0)
    x_processed = preprocess_input(img_batch)
    y = model.predict(x_processed)
    list_y.append(y)
        
    list_z.append(z)
    
    # Processed, delete pickled file to clear space
    # os.remove(pathfile_pkl)


# Grow previously processed datasets
if os.path.exists(pathfile_y) and os.path.exists(pathfile_z):
    with h5py.File(pathfile_y, 'r') as f:
        old_y = f['y'][:]
        # save y (features)
        y_concat = np.concatenate((old_y, np.concatenate(list_y)), axis=0)
        pathfile_sample_y = os.path.join(path_gan_sample_img, filename_sample_y)
    with h5py.File(pathfile_sample_y, 'w') as f:
        print(y_concat.shape)
        f.create_dataset('y', data=y_concat)
    with h5py.File(pathfile_z, 'r') as f:
        old_z = f['z'][:]
        # save z (latent variables)
        z_concat = np.concatenate((old_z, np.concatenate(list_z)), axis=0)
        pathfile_sample_z = os.path.join(path_gan_sample_img, filename_sample_z)
    with h5py.File(pathfile_sample_z, 'w') as f:
        print(z_concat.shape)
        f.create_dataset('z', data=z_concat)

else:
    # save y (features)
    y_concat = np.concatenate(list_y, axis=0)
    pathfile_sample_y = os.path.join(path_gan_sample_img, filename_sample_y)
    with h5py.File(pathfile_sample_y, 'w') as f:
        f.create_dataset('y', data=y_concat)    

    # save z (latent variables)
    z_concat = np.concatenate(list_z, axis=0)
    pathfile_sample_z = os.path.join(path_gan_sample_img, filename_sample_z)
    with h5py.File(pathfile_sample_z, 'w') as f:
        f.create_dataset('z', data=z_concat)
    
