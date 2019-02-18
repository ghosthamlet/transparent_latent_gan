import os
import glob
import sys
import numpy as np
import pickle
import tensorflow as tf
import PIL
import ipywidgets
import io
import pandas as pd

""" make sure this notebook is running from root directory """
while os.path.basename(os.getcwd()) in ('notebooks', 'src'):
    os.chdir('..')
assert ('README.md' in os.listdir('./')), 'Can not find project root, please cd to project root before running the following code'

import src.tl_gan.generate_image as generate_image
import src.tl_gan.feature_axis as feature_axis
from src.tl_gan.feature_celeba_organize import feature_name_celeba_org as feature_name_list

from src.notebooks.conversions import feature_to_image as feature_to_image, b64_image_to_feature, b64_image_to_prediction_image, dict_to_image

path_model = './asset_model/karras2019stylegan-ffhq-1024x1024.pkl'

sess = tf.InteractiveSession()
global Gs

try:
    with open(path_model, 'rb') as file:
        G, D, Gs = pickle.load(file)
        print('original', Gs)
except FileNotFoundError:
    print('before running the code, download pre-trained model to project_root/asset_model/')
    raise



def generate_images(num_image, lower_bound, upper_bound, feature_name, feature_labels, others_constant_level):
    step_size = (upper_bound - lower_bound) / (num_image - 1)
    
    cur_level = lower_bound
    
    for i in range(num_image):
        feature_vec = [others_constant_level] * 40
        
        feature_vec[feature_labels.index(feature_name)] = cur_level
        
        dirName = 'src/notebooks/out/{}/'.format(feature_name)
 
        try:
            # Create target Directory
            os.mkdir(dirName)
            print("Directory " , dirName ,  " Created ") 
        except FileExistsError:
            print("Directory " , dirName ,  " already exists")

        
        print("range feature vec", feature_vec)
        feature_to_image(feature_vec, save_name='src/notebooks/out/{}/other{}_this{}.jpg'.format(feature_name, others_constant_level, cur_level), our_Gs=Gs)
        cur_level += step_size

for feature_name in feature_name_list:
    generate_images(50, -0.3, 0.3, feature_name, feature_name_list, 0)
    break
    

'''
for i in range(1):
    for j in range(11):
        feature_to_image([0] * i + [(5 - j) * 0.05] + [0] * (40 - i - 1), save_name='src/notebooks/out/test_{}th_val_{}.jpg'.format(i, (5 - j) * 0.05))
'''