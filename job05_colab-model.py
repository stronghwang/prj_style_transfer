import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import time
from tensorflow.keras.layers import *
# pip install git+https://www.github.com/keras-team/keras-contrib.git 터미널에 입력시 다운로드된다.
train_x, test_x = np.load('./dataset/mone_image_data.npy', allow_pickle = True) # pickle 은 객체의 형태를 그대로 유지하며 저장
train_y, test_y = np.load('./datasets/picture_image_data.npy', allow_pickle = True) # pickle 은 객체의 형태를 그대로 유지하며 저장

# x = next(iter(test_x.shuffle(1000)))
# print(x)

epochs = 2

LAMBDA = 10

img_rows,img_cols,channel = 256,256,3
weight_initializer = RandomNormal(stddev = 0.02)

gen_g_optimizer = gen_f_optimizer = Adam(lr = 0.0002,beta_1 = 0.5)
dis_x_optimizer = dis_y_optimizer = Adam(lr = 0.0002,beta_1 = 0.5)

# def preprocess_image(img,_): #이미지 전처리
#   return tf.reshape(tf.cast(tf.image.resize(img,(int(img_rows),int(img_cols))),tf.float32) / 127.5 - 1,(1,img_rows,img_cols,channel))
#
# train_x = train_x.map(preprocess_image)
# train_y = train_y.map(preprocess_image)
# test_x = test_x.map(preprocess_image)
# test_y = test_y.map(preprocess_image)



def Ck(input,k,use_instancenorm=True):
  block = Conv2D(k,(4,4),strides=2,padding='same',kernel_initializer=weight_initializer)(input)
  if use_instancenorm:
    block = BatchNormalization()(block)
  block = LeakyReLU(0.2)(block)

  return block



from tensorflow.keras.layers import Input,Conv2D
from tensorflow.keras.models import Model

def discriminator():
  dis_input = Input(shape=(img_rows,img_cols,channel))

  d = Ck(dis_input,64,False)
  d = Ck(d,128)
  d = Ck(d,256)
  d = Ck(d,512)

  d = Conv2D(1,(4,4),padding='same',kernel_initializer=weight_initializer)(d)

  dis = Model(dis_input,d)
  dis.compile(loss='mse',optimizer=dis_x_optimizer)
  return dis

def dk(k,use_instancenorm=True):
  block = Sequential()
  block.add(Conv2D(k,(3,3),2,padding='same',kernel_initializer=weight_initializer))
  if use_instancenorm:
    block.add(BatchNormalization())
  block.add(Activation('relu'))

  return block


def uk(k): # resblock
  block = Sequential()
  block.add(Conv2DTranspose(k,(3,3),2,padding='same',kernel_initializer=weight_initializer))
  block.add(BatchNormalization())
  block.add(Activation('relu'))

  return block


def generator():
  gen_input = Input(shape=(img_rows,img_cols,channel))

  encoder_layers = [
    dk(64,False),
    dk(128),
    dk(256),
    dk(512),
    dk(512),
    dk(512),
    dk(512),
    dk(512)
  ]

  decoder_layers = [
    uk(512),
    uk(512),
    uk(512),
    uk(512),
    uk(256),
    uk(128),
    uk(64)
  ]

  gen = gen_input

  skips = []

  for layer in encoder_layers:
    gen = layer(gen)
    skips.append(gen)

  # Reverse for looping and get rid of the layer that directly connects to decoder
  skips = skips[::-1][1:]

  for skip_layer,layer in zip(skips,decoder_layers):
    gen = layer(gen)
    gen = Concatenate()([gen,skip_layer])

  gen = Conv2DTranspose(channel,(3,3),2,padding='same',kernel_initializer=weight_initializer,activation='tanh')(gen)

  return Model(gen_input,gen)

generator_g = generator()
generator_f = generator()

discriminator_x = discriminator()
discriminator_y = discriminator()

from tensorflow.keras.losses import BinaryCrossentropy
BinaryCrossentropy
loss = BinaryCrossentropy(from_logits=True)

def discriminator_loss(real,generated):
  return (loss(tf.ones_like(real),real) + loss(tf.zeros_like(generated),generated)) * 0.5

def gen_loss(validity):
  return loss(tf.ones_like(validity),validity)

def image_similarity(image1,image2):
  return tf.reduce_mean(tf.abs(image1 - image2)) #the mean absolute error of image1-image2

#tracks all operations in a function and constructs a callable that can execute a TensorFlow graph
@tf.function
def step(real_x,real_y):
  with tf.GradientTape(persistent=True) as tape:
    fake_y = generator_g(real_x,training=True)
    gen_g_validity = discriminator_y(fake_y,training=True)
    dis_y_loss = discriminator_loss(discriminator_y(real_y,training=True),gen_g_validity)

    with tape.stop_recording():
      discriminator_y_gradients = tape.gradient(dis_y_loss,discriminator_y.trainable_variables)
      dis_y_optimizer.apply_gradients(zip(discriminator_y_gradients,discriminator_y.trainable_variables))

    fake_x = generator_f(real_y,training=True)
    gen_f_validity = discriminator_x(fake_x,training=True)
    dis_x_loss = discriminator_loss(discriminator_x(real_x,training=True),gen_f_validity)

    with tape.stop_recording():
      discriminator_x_gradients = tape.gradient(dis_x_loss,discriminator_x.trainable_variables)
      dis_x_optimizer.apply_gradients(zip(discriminator_x_gradients,discriminator_x.trainable_variables))

    gen_g_adv_loss = gen_loss(gen_g_validity)
    gen_f_adv_loss = gen_loss(gen_f_validity)

    cyc_x = generator_f(fake_y,training=True)
    cyc_x_loss = image_similarity(real_x,cyc_x)

    cyc_y = generator_g(fake_x,training=True)
    cyc_y_loss = image_similarity(real_y,cyc_y)

    id_x = generator_f(real_x,training=True)
    id_x_loss = image_similarity(real_x,id_x)

    id_y = generator_g(real_y,training=True)
    id_y_loss = image_similarity(real_y,id_y)

    gen_g_loss = gen_g_adv_loss + (cyc_x_loss + cyc_y_loss) * LAMBDA + id_y_loss * 0.5 * LAMBDA
    gen_f_loss = gen_f_adv_loss + (cyc_x_loss + cyc_y_loss) * LAMBDA + id_x_loss * 0.5 * LAMBDA

    with tape.stop_recording():
      generator_g_gradients = tape.gradient(gen_g_loss,generator_g.trainable_variables)
      gen_g_optimizer.apply_gradients(zip(generator_g_gradients,generator_g.trainable_variables))

      generator_f_gradients = tape.gradient(gen_f_loss,generator_f.trainable_variables)
      gen_f_optimizer.apply_gradients(zip(generator_f_gradients,generator_f.trainable_variables))

def generate_images():

  x = next(iter(test_x.shuffle(1000))).numpy()
  y = next(iter(test_y.shuffle(1000))).numpy()

  y_hat = generator_g.predict(x.reshape((1,img_rows,img_cols,channel)))
  x_hat = generator_f.predict(y.reshape((1,img_rows,img_cols,channel)))

  plt.figure(figsize=(12,12))

  images = [x[0],y_hat[0],y[0],x_hat[0]]

  for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i] * 0.5 + 0.5)
    plt.axis('off')

  plt.tight_layout()
  plt.show()

for epoch in range(20):
  print("Epoch - {}".format(epoch))
  start = time.time()

  for k, (real_x,real_y) in enumerate(zip(train_x,train_y)):
    if k % 100 == 0:
      print(k)
    step(tf.reshape(real_x,(1,img_rows,img_cols,channel)),tf.reshape(real_y,(1,img_rows,img_cols,channel)))

  generate_images()
  print("time taken - {}".format(time.time() - start))

