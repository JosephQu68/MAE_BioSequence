from keras import layers, Model,activations
import keras
from keras import backend as K
import tensorflow as tf

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

BATCH_SIZE = 64
def calReconstructRate(y_true,y_pred):
        out = tf.equal(tf.argmax(y_true, axis=2),tf.argmax(y_pred,axis=2))
        out = tf.cast(out, tf.int32)
        sum = tf.reduce_sum(out)
        return sum / (y_true.shape[1]*BATCH_SIZE)  

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
