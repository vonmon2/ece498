import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import requests
import gzip

def load_dataset(path):
    num_img = 1000
    with gzip.open(path, 'rb') as infile:
        data = np.frombuffer(infile.read(), dtype=np.uint8).reshape(num_img, 784)
    return data

def get_testset():
    url = 'https://courses.engr.illinois.edu/ece498icc/sp2020/lab1_request_dataset.php'
    values = {'request': 'testdata', 'netid':'bcallas2'}
    r = requests.post(url, data=values, allow_redirects=True)
    filename = r.url.split("/")[-1]
    testset_id = filename.split(".")[0].split("_")[-1]
    with open(filename, 'wb') as f: 
        f.write(r.content)
    return load_dataset(filename), testset_id

keras_model = tf.keras.models.load_model('keras_model.h5')

data, testset_id = get_testset()
data = data.reshape(1000, 28, 28, 1)

pred = keras_model.predict(data)

predacc = requests.post('https://courses.engr.illinois.edu/ece498icc/sp2020/lab1_request_dataset.php', data = {'request': 'verify', 'netid':'bcallas2', 'testset_id': testset_id, 'prediction': pred})
print('Prediction Accuracy: ' + str(predacc))