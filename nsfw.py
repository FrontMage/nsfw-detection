#%%
import os
import numpy as np
import cv2
import keras
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import pickle
from tqdm import tqdm


def read_img(url: str):
    b, g, r = cv2.split(cv2.resize(cv2.imread(url), dsize=(
        128, 128), interpolation=cv2.INTER_CUBIC))
    return cv2.merge([r, g, b])


def show_img(img):
    plt.imshow(img)
    plt.show()


#%%
# Show some sample images
# sample_cat = read_img(os.path.join(TRAIN_CAT_DIR,
#                                'cat.1.jpg'))

# show_img(sample_cat)

# sample_dog = read_img(os.path.join(TRAIN_DOG_DIR,'dog.1.jpg'))
# show_img(sample_dog)
#%%
def load_cat_dog_data():
    # This should be classified as 0
    TRAIN_CAT_DIR = '/home/xinbg/Downloads/dogs-vs-cats/train'
    TEST_CAT_DIR = '/home/xinbg/Downloads/dogs-vs-cats/test1'
    # This should be classified as 1
    TRAIN_DOG_DIR = '/home/xinbg/Downloads/dogs-vs-cats/train'
    TEST_DOG_DIR = '/home/xinbg/Downloads/dogs-vs-cats/test1'

    CLASSES = ['cat', 'dog']

    TRAIN_IMAGES = os.listdir(TRAIN_CAT_DIR)
    TEST_IMAGES = os.listdir(TEST_CAT_DIR)
    exists = os.path.isfile('./x.dog.train')
    if not exists:
        xs = []
        ys = []

        for idx in tqdm(range(len(TRAIN_IMAGES))):
            i = TRAIN_IMAGES[idx]
            p = os.path.join(TRAIN_CAT_DIR, i)
            xs.append(read_img(p))
            if 'cat' in i:
                ys.append(0)
            if 'dog' in i:
                ys.append(1)
        pickle.dump(xs, open('./x.dog.train', 'w+b'))
        pickle.dump(ys, open('./y.dog.train', 'w+b'))
    else:
        xs = pickle.load(open('./x.dog.train', 'rb'))
        ys = pickle.load(open('./y.dog.train', 'rb'))

    X = np.array(xs).astype('float32')
    X = X/255
    Y = np.array(ys).astype('float32')
    Y = to_categorical(Y, 2)
    return X, Y


def load_nsfw_data():
    PORN_DIR = '/home/xinbg/Downloads/dataset/dataset/train_set/nsfw'
    NEUTRAL_DIR = '/home/xinbg/Downloads/dataset/dataset/train_set/sfw'
    if not os.path.isfile('./x.nsfw.train'):
        xs = []
        ys = []
        porns = os.listdir(PORN_DIR)
        for idx in tqdm(range(len(porns))):
            try:
                xs.append(read_img(os.path.join(PORN_DIR, porns[idx])))
                ys.append(1)
            except Exception as e:
                print('Image parse failed with: ',
                      os.path.join(PORN_DIR, porns[idx]))
        neutrals = os.listdir(NEUTRAL_DIR)
        for idx in tqdm(range(len(neutrals))):
            try:
                xs.append(read_img(os.path.join(NEUTRAL_DIR, neutrals[idx])))
                ys.append(0)
            except Exception as e:
                print('Image parse failed with: ',
                      os.path.join(NEUTRAL_DIR, neutrals[idx]))
        pickle.dump(xs, open('./x.nsfw.train', 'w+b'))
        pickle.dump(ys, open('./y.nsfw.train', 'w+b'))
    else:
        xs = pickle.load(open('./x.nsfw.train', 'rb'))
        ys = pickle.load(open('./y.nsfw.train', 'rb'))
    X = np.array(xs).astype('float32')
    X = X/255
    Y = np.array(ys).astype('float32')
    Y = to_categorical(Y, 2)
    return X, Y


X, Y = load_nsfw_data()
print('X train shape: ', X.shape)
print('Y train shape: ', Y.shape)

#%%


def define_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3),
                                  activation='relu',
                                  input_shape=X.shape[1:]))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3),
                                  activation='relu',
                                  ))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(128, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(2, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def train():
    model = define_model()
    model.fit(X, Y,
              shuffle=True,
              epochs=3,
              validation_split=0.3
              )
    return model


def load(path: str):
    model = define_model()
    model.load_weights(path)
    return model


m = train()
m.save('./nsfw.model', overwrite=True)
# m = load('./nsfw.model')
# print(m.predict(np.array([read_img(
#     '/home/xinbg/Downloads/dataset/dataset/test_set/nsfw/greek-porn-655253.jpg')])))
