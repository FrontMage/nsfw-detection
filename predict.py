import os
import numpy as np
import cv2
from nsfw_2 import define_model

DIR = '/home/xinbg/Pictures/wallpapers/Art'


def read_img(url: str):
    b, g, r = cv2.split(cv2.resize(cv2.imread(url), dsize=(
        192, 192), interpolation=cv2.INTER_CUBIC))
    return cv2.merge([r, g, b])


m = define_model()
m.load_weights('./weights.hdf5')

for i in os.listdir(DIR):
    try:
        r = m.predict(np.array([read_img(os.path.join(DIR, i))]))
        print(r)
    except Exception as e:
        pass

DIR2 = './data/test_set/nsfw'
for i in os.listdir(DIR2):
    try:
        r = m.predict(np.array([read_img(os.path.join(DIR2, i))]))
        print(r)
    except Exception as e:
        pass
