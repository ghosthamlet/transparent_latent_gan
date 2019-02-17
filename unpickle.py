import pickle
import keras
from keras.models import load_model


# file_model1 = open('asset_model/karras2018iclr-celebahq-1024x1024.pkl', 'rb')
# file_model2 = open('asset_model/cnn_face_attr_celeba/model_20180927_032934.h5', 'rb')

# model1 = pickle.load(file_model1)
model2 = load_model('asset_model/cnn_face_attr_celeba/model_20180927_032934.h5')

# print(type(model1))
print(model2)
# print(type(model2))

