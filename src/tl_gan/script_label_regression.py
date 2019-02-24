""" functions to regress y (labels) based on z (latent space) """

import os
import glob
import numpy as np
import pickle
import h5py
import pandas as pd

import feature_axis

import time

def gen_time_str():
    """ tool function to generate time str like 20180927_205959 """
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())

""" make sure this notebook is running from root directory """
while os.path.basename(os.getcwd()) in ('notebooks', 'src', 'tl_gan'):
     os.chdir('..')
print(os.getcwd())
assert ('README.md' in os.listdir('./')), 'Can not find project root, please cd to project root before running the following code'

# import src.misc as misc
# import src.tl_gan.feature_axis as feature_axis

""" get y and z from pre-generated files """
# path_gan_sample_img = './asset_results/pggan_celeba_sample_jpg/'
# path_celeba_att = './data/raw/celebA_annotation/list_attr_celeba.txt'
# path_feature_direction = './asset_results/pg_gan_celeba_feature_direction_40'

path_gan_sample_img = 'asset_results/stylegan_ffhq_sample_jpg/'
# path_celeba_att = './data/raw/celebA_annotation/list_attr_celeba.txt'
path_feature_direction = 'asset_results/stylegan_feature_direction_40'

filename_sample_y = 'stylegan_sample_y.h5'
filename_sample_z = 'stylegan_sample_z.h5'

pathfile_y = os.path.join(path_gan_sample_img, filename_sample_y)
pathfile_z = os.path.join(path_gan_sample_img, filename_sample_z)

with h5py.File(pathfile_y, 'r') as f:
    y = f['y'][:]
with h5py.File(pathfile_z, 'r') as f:
    z = f['z'][:]

# read feature name
# df_attr = pd.read_csv(path_celeba_att, sep='\s+', header=1, index_col=0)
# y_name = df_attr.columns.values.tolist()
y_name = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

##
""" regression: use latent space z to predict features y """
feature_slope = feature_axis.find_feature_axis(z, y, method='tanh')

##
""" normalize the feature vectors """
yn_normalize_feature_direction = True
if yn_normalize_feature_direction:
    feature_direction = feature_axis.normalize_feature_axis(feature_slope)
else:
    feature_direction = feature_slope

""" save_regression result to hard disk """
if not os.path.exists(path_feature_direction):
    os.mkdir(path_feature_direction)

pathfile_feature_direction = os.path.join(path_feature_direction, 'feature_direction_{}.pkl'.format(gen_time_str()))
dict_to_save = {'direction': feature_direction, 'name': y_name}
with open(pathfile_feature_direction, 'wb') as f:
    pickle.dump(dict_to_save, f)


##
""" disentangle correlated feature axis """
pathfile_feature_direction = glob.glob(os.path.join(path_feature_direction, 'feature_direction_*.pkl'))[-1]

with open(pathfile_feature_direction, 'rb') as f:
    feature_direction_name = pickle.load(f)

feature_direction = feature_direction_name['direction']
feature_name = np.array(feature_direction_name['name'])

len_z, len_y = feature_direction.shape


feature_direction_disentangled = feature_axis.disentangle_feature_axis_by_idx(
    feature_direction, idx_base=range(len_y//4), idx_target=None)

feature_axis.plot_feature_cos_sim(feature_direction_disentangled, feature_name=feature_name)

##

