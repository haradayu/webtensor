from __future__ import print_function
import numpy as np
np.random.seed(114514)  # for reproducibility
#from keras.datasets import mnist

#from keras.utils import np_utils
from keras import backend as K
import pickle
import model_file
def lead_datasets(pkl_path, image_size):
    # deserialize
    with open(pkl_path, 'rb') as f:
      data = pickle.load(f)
    X_train = data[0]
    Y_train = data[1]
    X_test = data[2]
    Y_test = data[3]
    # input image dimensions
    output_classes = Y_train.shape[1]
    img_rows, img_cols = image_size
    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    return X_train, Y_train, X_test, Y_test

batch_size = 512
nb_epoch = 12

X_train, Y_train, X_test, Y_test = lead_datasets('mnist/out.pkl', (28, 28))
model = model_file.return_model()
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])