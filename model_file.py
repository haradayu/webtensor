from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

def return_model():
    input_shape = (28, 28, 1)
    output_classes = 10
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_classes))
    model.add(Activation('softmax'))
    return model
if __name__ == "__main__":
    # # execute only if run as a script
    from keras.utils.visualize_util import plot
    # model = return_model((28, 28, 1), 10)
    # plot(model, to_file='model.png', show_shapes = True)
    from keras.applications.inception_v3 import InceptionV3
    model = InceptionV3(include_top=True, weights='imagenet', input_tensor=None)
    plot(model, to_file='model.png', show_shapes = True)
    