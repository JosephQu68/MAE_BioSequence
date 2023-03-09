from keras import layers, Model,activations
import keras
from keras import backend as K
import tensorflow as tf
import numpy as np

class Autoencoder(Model):
  def __init__(self, encoder_shapes, latent_dim = 64):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder_shapes = encoder_shapes

    self.encoder = keras.Sequential([])
    self.encoder.add(layers.Input(shape=self.encoder_shapes, name = 'input_layer'))

    self.encoder.add(layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', name='conv1'))
    self.encoder.add(layers.BatchNormalization(name='bn1'))
    self.encoder.add(layers.ReLU(name='relu1'))
    self.encoder.add(layers.Dropout(rate=0.5))

    self.encoder.add(layers.Conv1D(64,3,2,padding='same', name='conv2'))
    self.encoder.add(layers.BatchNormalization(name='bn2'))
    self.encoder.add(layers.ReLU(name='relu2'))
    self.encoder.add(layers.Dropout(rate=0.5))


    self.encoder.add(layers.Conv1D(64,3,2,padding='same', name='conv3'))
    self.encoder.add(layers.BatchNormalization(name='bn3'))
    self.encoder.add(layers.ReLU(name='relu3'))
    self.encoder.add(layers.Dropout(rate=0.5))


    self.encoder.add(layers.Conv1D(64,3,1,padding='same', name='conv4'))
    self.encoder.add(layers.BatchNormalization(name='bn4'))
    self.encoder.add(layers.ReLU(name='relu4'))
    self.encoder.add(layers.Dropout(rate=0.5))


    self.encoder.add(layers.LSTM(32, return_sequences=True))
    self.encoder.add(layers.BatchNormalization())

    self.encoder.add(layers.LSTM(64, recurrent_dropout=0.5, activation='tanh'))
    self.encoder.add(layers.BatchNormalization())

    self.encoder.add(layers.Flatten())
    self.encoder.add(layers.Dense(self.latent_dim, name='dense1'))


    self.decoder = keras.Sequential([])
    self.decoder.add(layers.Input(shape=(self.latent_dim),name = 'decoder_input'))
    self.decoder.add(layers.Dense(units=12416))
    self.decoder.add(layers.Reshape([194,64]))
    self.encoder.add(layers.Dropout(rate=0.1))


    self.decoder.add(layers.Conv1DTranspose(64,3,1,'same'))
    self.decoder.add(layers.BatchNormalization())
    self.decoder.add(layers.ReLU())
    self.encoder.add(layers.Dropout(rate=0.1))


    self.decoder.add(layers.Conv1DTranspose(64,3,2,'same'))
    self.decoder.add(layers.BatchNormalization())
    self.decoder.add(layers.ReLU())
    self.encoder.add(layers.Dropout(rate=0.1))


    # self.decoder.add(layers.Conv1DTranspose(64,3,2,'same'))
    # self.decoder.add(layers.BatchNormalization())
    # self.decoder.add(layers.ReLU())
    
    self.decoder.add(layers.Conv1DTranspose(32,3,1,'same'))
    self.decoder.add(layers.BatchNormalization())
    self.decoder.add(layers.ReLU())
    self.encoder.add(layers.Dropout(rate=0.1))

    # self.decoder.add(layers.Conv1DTranspose(self.encoder_shapes[1],3,1,'same'))
    # self.decoder.add(layers.BatchNormalization())
    # self.decoder.add(layers.Softmax())
    self.decoder.add(layers.Flatten())
    self.decoder.add(layers.Dense(units=self.encoder_shapes[0]*self.encoder_shapes[1]))
    self.decoder.add(layers.BatchNormalization())
    self.decoder.add(layers.Softmax())

    self.decoder.add(layers.Reshape(self.encoder_shapes))

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded



class AutoencoderGRU(Model):
  def __init__(self, encoder_shapes, latent_dim = 64, gru_layer_shape = [512,1024,2048]):
    super(AutoencoderGRU, self).__init__()
    self.latent_dim = latent_dim
    self.encoder_shapes = encoder_shapes
    self.gru_layer_shape = gru_layer_shape

    self.encoder = self.create_encoder()

    self.decoder = self.create_decoder()

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

  def create_encoder(self):
    input_layer = layers.Input(shape=self.encoder_shapes)

    gru_0 = layers.GRU(self.gru_layer_shape[0],return_sequences=True)(input_layer)
    gru_0 = layers.BatchNormalization()(gru_0)

    gru_1 = layers.GRU(self.gru_layer_shape[1],return_sequences=True)(gru_0)
    gru_1 = layers.BatchNormalization()(gru_1)

    gru_2 = layers.GRU(self.gru_layer_shape[2],return_sequences=True)(gru_1)
    gru_2 = layers.BatchNormalization()(gru_2)

    conc = layers.concatenate([gru_0, gru_1,gru_2], axis=2)

    latent = layers.Dense(units=self.latent_dim)(conc)
    latent = layers.BatchNormalization()(latent)

    noise = layers.GaussianNoise(0.1)(latent)

    output_latent = activations.tanh(noise)

    encoder = Model(inputs = input_layer, outputs = output_latent)
    return encoder

  def create_decoder(self):
    input_layer = layers.Input(shape=(self.encoder_shapes[0], self.latent_dim))

    densed = layers.Dense(sum(self.gru_layer_shape))(input_layer)
    densed = layers.BatchNormalization()(densed)

    split = tf.split(densed, self.gru_layer_shape, axis=2)

    # cddd此处引入长度为32的helper
    gru_0 = layers.GRU(self.gru_layer_shape[0], return_sequences= True, name = 'decoder_gru_0')(split[0])
    gru_0 = layers.BatchNormalization()(gru_0)

    concate_for_gru1 = layers.concatenate([gru_0, split[1]], axis=2)
    gru_1 = layers.GRU(self.gru_layer_shape[1], return_sequences= True, name = 'decoder_gru_1')(concate_for_gru1)
    gru_1 = layers.BatchNormalization()(gru_1)

    concate_for_gru2 = layers.concatenate([gru_1,split[2]],axis=2)
    gru_2 = layers.GRU(self.gru_layer_shape[2],return_sequences=True,name = 'decoder_gru_2')(concate_for_gru2)
    gru_2 = layers.BatchNormalization()(gru_2)

    densed = layers.Dense(self.encoder_shapes[1])(gru_2)
    densed = layers.BatchNormalization()(densed)
    output_layer = layers.Softmax()(densed)

    decoder = Model(inputs = input_layer, outputs = output_layer)
    return decoder


class AutoencoderGRU_withMaskLoss(keras.Model):
  def __init__(self, encoder_shapes, latent_dim = 64, gru_layer_shape = [512,1024,2048], mask_rate = 0.15):
    super(AutoencoderGRU_withMaskLoss, self).__init__()
    self.latent_dim = latent_dim
    self.encoder_shapes = encoder_shapes
    self.gru_layer_shape = gru_layer_shape
    self.mask_rate = mask_rate

    self.encoder = self.create_encoder()

    self.decoder = self.create_decoder()

  # def cal_loss(self, input_data):
  #   masked_data, masked_range = self.mask_onehot_matrix(input_data, self.mask_rate)
  #   encoded = self.encoder(masked_data)
  #   encoded = self.encoder(input_data)
  #   output = self.decoder(encoded)


  #   masked_idx = np.zeros(masked_data.shape[:-1])
  #   for _ in range(masked_range.shape[0]):
  #     masked_idx[_,masked_range[_,0]:masked_range[_,1]] = 1
  #   masked_idx = tf.constant(masked_idx)
  #   masked_idx_expand = masked_idx[:,:,tf.newaxis]
  #   masked_idx_expand = tf.cast(masked_idx_expand, tf.float32)
  #   masked_idx = tf.cast(masked_idx, tf.bool)


  #   # loss = self.compiled_loss(output[masked_idx], input_data[masked_idx])
  #   loss = self.compiled_loss(output, input_data+1e-8)

  #   return loss, masked_idx_expand, output, masked_data

  # def call(self, inputs, training=None, mask=None):
  #   encoded = self.encoder(inputs)
  #   decoded = self.decoder(encoded)
  #   return decoded


  def train_step(self, data):
    # masked_data, masked_range = self.mask_onehot_matrix(data, self.mask_rate)
    with tf.GradientTape() as tape:
      # loss,masked_idx_expand, output, masked_data= self.cal_loss(data)
      encoded = self.encoder(data[0])
      output = self.decoder(encoded)
      loss = self.compiled_loss(data[1], output)
    
    # train_vars = [
    #     self.encoder.trainable_variables,
    #     self.decoder.trainable_variables
    #   ]
    # grads = tape.gradient(loss, train_vars)
    # tv_list = []
    # for (grad,var) in zip(grads, train_vars):
    #   for g,v in zip(grad,var):
    #     tv_list.append((g,v))
    # self.optimizer.apply_gradients(tv_list)
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # self._validate_target_and_loss(output,loss)
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    # self.compiled_metrics.update_state(data, output*masked_idx_expand+masked_data)
    self.compiled_metrics.update_state(data, output)
    return{m.name: m.result() for m in self.metrics}
  
  # def test_step(self, data):
  #   loss, masked_idx_expand, output,masked_data = self.cal_loss(data)

  #   self.compiled_metrics.update_state(data, output*masked_idx_expand+masked_data)
  #   return {m.name:m.result() for m in self.metrics}
    
    ###

  # def mask_onehot_matrix(self, onehot_data, mask_rate):
  #   # To input the whole data
  #   onehot_data = onehot_data.numpy()
  #   len_data = onehot_data.shape[0]
  #   res = onehot_data.copy()

  #   val_len = np.sum(np.sum(onehot_data, axis=2),axis=1)
  #   mask_len = (val_len * mask_rate).astype(np.int32)
  #   right_limit = val_len-mask_len

  #   start_point = np.random.randint(right_limit)
  #   end_point = start_point+mask_len
    
  #   for _ in range(len_data):
  #       res[_,start_point[_]:end_point[_],:] = 0.
  #   return tf.Variable(res),np.transpose([start_point,end_point])

  def create_encoder(self):
    input_layer = layers.Input(shape=self.encoder_shapes)

# kernel_regularizer=keras.regularizers.l2()
    gru_0 = layers.GRU(self.gru_layer_shape[0],return_sequences=True)(input_layer)
    gru_0 = layers.BatchNormalization()(gru_0)

    gru_1 = layers.GRU(self.gru_layer_shape[1],return_sequences=True)(gru_0)
    gru_1 = layers.BatchNormalization()(gru_1)

    gru_2 = layers.GRU(self.gru_layer_shape[2],return_sequences=True)(gru_1)
    gru_2 = layers.BatchNormalization()(gru_2)

    conc = layers.concatenate([gru_0, gru_1,gru_2], axis=2)

    latent = layers.Dense(units=self.latent_dim)(conc)
    latent = layers.BatchNormalization()(latent)

    noise = layers.GaussianNoise(0.1)(latent)

    output_latent = activations.tanh(noise)

    encoder = Model(inputs = input_layer, outputs = output_latent)
    return encoder

  def create_decoder(self):
    input_layer = layers.Input(shape=(self.encoder_shapes[0], self.latent_dim))

    densed = layers.Dense(sum(self.gru_layer_shape))(input_layer)
    densed = layers.BatchNormalization()(densed)

    split = tf.split(densed, self.gru_layer_shape, axis=2)

    # cddd此处引入长度为32的helper
    gru_0 = layers.GRU(self.gru_layer_shape[0], return_sequences= True, name = 'decoder_gru_0')(split[0])
    gru_0 = layers.BatchNormalization()(gru_0)

    concate_for_gru1 = layers.concatenate([gru_0, split[1]], axis=2)
    gru_1 = layers.GRU(self.gru_layer_shape[1], return_sequences= True, name = 'decoder_gru_1')(concate_for_gru1)
    gru_1 = layers.BatchNormalization()(gru_1)

    concate_for_gru2 = layers.concatenate([gru_1,split[2]],axis=2)
    gru_2 = layers.GRU(self.gru_layer_shape[2],return_sequences=True,name = 'decoder_gru_2')(concate_for_gru2)
    gru_2 = layers.BatchNormalization()(gru_2)

    densed = layers.Dense(self.encoder_shapes[1])(gru_2)
    densed = layers.BatchNormalization()(densed)
    output_layer = layers.Softmax()(densed)

    decoder = Model(inputs = input_layer, outputs = output_layer)
    return decoder

def ReconstructRateVaried(y_true,y_pred):
  y_true = tf.convert_to_tensor(y_true)
  y_pred = tf.convert_to_tensor(y_pred)

  val_len = tf.reduce_sum(y_true, axis=2)
  val_len = tf.cast(val_len, tf.int32)

  out = tf.equal(tf.argmax(y_true, axis=2),tf.argmax(y_pred,axis=2))
  out = tf.cast(out, tf.int32)
  sum = tf.reduce_sum(out*val_len)

  return tf.reduce_sum(sum)/tf.reduce_sum(val_len)


def my_loss_entropy(y_true, y_pred):
    list_loss = K.categorical_crossentropy(y_true, y_pred)
    return K.mean(list_loss)

# def my_loss_entropy_masked(masked_onehot_data):
#     masked_idx = tf.reduce_sum(masked_onehot_data,axis=2)
#     print(masked_idx.shape)
#     def loss(y_true, y_pred):
#       return K.mean(K.categorical_crossentropy(y_true*masked_idx, y_pred*masked_idx))
#     return loss
