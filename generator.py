
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D, Conv2D

def build_generator():
    gen_model = Sequential()
    gen_model.add(Dense(input_dim=100, units = 2048))
    gen_model.add(LeakyReLU(alpha = 0.2))
    gen_model.add(Dense(256 * 8 * 8))
    gen_model.add(BatchNormalization())
    gen_model.add(LeakyReLU(alpha=0.2))

    gen_model.add(Reshape((8, 8, 256), input_shape=(256 * 8 * 8,)))
    gen_model.add(UpSampling2D(size=(2, 2)))

    gen_model.add(Conv2D(128, (5, 5), padding='same'))
    gen_model.add(LeakyReLU(alpha=0.2))

    gen_model.add(UpSampling2D(size=(2, 2)))

    gen_model.add(Conv2D(64, (5, 5), padding='same'))
    gen_model.add(LeakyReLU(alpha=0.2))

    gen_model.add(UpSampling2D(size=(2, 2)))
    gen_model.add(Conv2D(3, (5, 5), padding='same'))
    gen_model.add(LeakyReLU(alpha=0.2))

    return gen_model
