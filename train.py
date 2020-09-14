from generator import build_generator
from discriminator import build_discriminator
import glob
import io
import math
import time

import keras.backend as K
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import Sequential, Input, Model
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.callbacks import TensorBoard
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import ReLU
from keras.layers import Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from scipy.misc import imread, imsave
from scipy.stats import entropy

def write_log(callback, name, loss, batch_no):
    """
    Write training summary to TensorBoard
    """
    # for name, value in zip(names, logs):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = loss
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()

def write_log2(callback, name, loss, batch_no):
    writer = tf.summary.create_file_writer("/Users/aashaysharma/Desktop/Generative-Adversarial-Networks-Projects-master/Chapter04/logs")
    with writer.as_default():

    # other model code would go here
        tf.summary.scalar("my_metric", loss, step=batch_no)
        writer.flush()

def save_rgb_img(img, path):
    """
    Save a rgb image
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("RGB Image")

    plt.savefig(path)
    plt.close()

start_time = time.time()
dataset_dir = "/Users/aashaysharma/Desktop/Generative-Adversarial-Networks-Projects-master/Chapter04/gallery-dl/danbooru/final_face/*.*"
batch_size = 128
z_shape = 100
epochs = 5000
dis_learning_rate = 0.0005
gen_learning_rate = 0.0005
dis_momentum = 0.9
gen_momentum = 0.9
dis_nesterov = True
gen_nesterov = True

all_images = []
for index, filename in enumerate(glob.glob(dataset_dir)):
    image = imread(filename, flatten=False)
    all_images.append(image)

X = np.array(all_images)
X = (X - 127.5) / 127.5
X = X.astype(np.float32)

dis_optimizer = SGD(lr=dis_learning_rate, momentum=dis_momentum, nesterov=dis_nesterov)
gen_optimizer = SGD(lr=gen_learning_rate, momentum=gen_momentum, nesterov=gen_nesterov)

gen_model = build_generator()
gen_model.compile(loss='binary_crossentropy', optimizer=gen_optimizer)

dis_model = build_discriminator()
dis_model.compile(loss='binary_crossentropy', optimizer=dis_optimizer)

adversarial_model = Sequential()
adversarial_model.add(gen_model)
dis_model.trainable = False
adversarial_model.add(dis_model)

adversarial_model.compile(loss='binary_crossentropy', optimizer=gen_optimizer)

tensorboard = TensorBoard(log_dir="logd/{}".format(time.time()), write_images=True, write_grads=True, write_graph=True)

tensorboard.set_model(gen_model)
tensorboard.set_model(dis_model)

for epoch in range(epochs):
    print("--------------------------")
    print("Epoch is", epoch)

    dis_losses = []
    gen_losses = []

    number_of_batches = int(X.shape[0]/batch_size)
    print("Number of Batches : ", number_of_batches)

    for index in range(number_of_batches):
        print("Batch:{}".format(index))

        z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

        image_batch = X[index * batch_size:(index + 1) * batch_size]

        generated_images = gen_model.predict_on_batch(z_noise)

        y_real = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
        y_fake = np.random.random_sample(batch_size) * 0.2

        dis_loss_real = dis_model.train_on_batch(image_batch, y_real)
        dis_loss_fake = dis_model.train_on_batch(generated_images, y_fake)

        d_loss = (dis_loss_real + dis_loss_fake)/2
        print("d_loss : ",d_loss)

        z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))
        g_loss = adversarial_model.train_on_batch(z_noise, y_real)
        print("g_loss:", g_loss)

        dis_losses.append(d_loss)
        gen_losses.append(g_loss)

        if epoch % 100 == 0:
            z_noise = np.random.normal(0,1,size=(batch_size, z_shape))
            gen_images1 = gen_model.predict_on_batch(z_noise)

            for img in gen_images1[:2]:
                save_rgb_img(img, "/Users/aashaysharma/Desktop/Generative-Adversarial-Networks-Projects-master/Chapter04/results/one_{}.png".format(epoch))
        print("Epoch:{}, dis_loss:{}".format(epoch, np.mean(dis_losses)))
        print("Epoch:{}, gen_loss: {}".format(epoch, np.mean(gen_losses)))

        """
        Save losses to Tensorboard after each epoch
        """
        write_log(tensorboard, 'discriminator_loss', np.mean(dis_losses), epoch)
        write_log(tensorboard, 'generator_loss', np.mean(gen_losses), epoch)

gen_model.save("generator_model.h5")
dis_model.save("discriminator_model.h5")

print("Time:", (time.time() - start_time))
