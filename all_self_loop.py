import keras
class CustomReconstructRate(keras.metrics.Metric):
    def __init__(self, name='custom_reconstruct_rate', dtype=None, **kwargs):
        super().__init__(name, dtype, **kwargs)
        self.res = self.add_weight(name='reconst', initializer='zeros')

    def update_state(self, y_true, y_pred):
        val_len = tf.reduce_sum(y_true, axis=2)
        val_len = tf.cast(val_len, tf.int32)

        out = tf.equal(tf.argmax(y_true, axis=2),tf.argmax(y_pred,axis=2))
        out = tf.cast(out, tf.int32)
        sum = tf.reduce_sum(out*val_len)

        rate = tf.reduce_sum(sum)/tf.reduce_sum(val_len)
        rate = tf.cast(rate, tf.float32)
        self.res.assign_add(rate)
    
    def result(self):
        return self.res

    def reset_states(self):
        self.res.assign(0.0)

import time
_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
_loss_fn = my_loss_entropy
_train_metric = CustomReconstructRate()
_val_metric = CustomReconstructRate()

autoencoder = AutoencoderGRU_withMaskLoss(latent_dim=512, encoder_shapes=(max_len, dimension))

for epoch in range(EPOCHS):
    print('\nStart of epoch %d'%epoch)
    start_time = time.time()

    for step, (x_train, y_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            encoded = autoencoder.encoder(x_train)
            output = autoencoder.decoder(encoded)

            _tmp_target = tf.cast(tf.reduce_sum(y_train,axis=2),tf.int32)
            _tmp_mask = tf.cast(tf.reduce_sum(x_train,axis=2), tf.int32)
            mask_idx = tf.bitwise.bitwise_xor(_tmp_target, _tmp_mask)
            masked_idx_expand = tf.cast(mask_idx[:,:,tf.newaxis], tf.float32)
            mask_idx = tf.cast(mask_idx, tf.bool)

            loss = _loss_fn(y_train[mask_idx],output[mask_idx])
        
        trainable_vars = autoencoder.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        _optimizer.apply_gradients(zip(gradients, trainable_vars)) 

        _train_metric.update_state(y_train,x_train + output*masked_idx_expand)

    train_rc_rate = _train_metric.result()
    print('Training loss at epoch %d : %.4f'%(epoch, loss))
    print('Training reconstruct rate at epoch %d : %.4f'%(epoch, train_rc_rate))
    _train_metric.reset_states()

    for x_val, y_val in val_dataset:
        output = autoencoder.decoder(autoencoder.encoder(x_val))
        _tmp_target = tf.cast(tf.reduce_sum(y_val,axis=2),tf.int32)
        _tmp_mask = tf.cast(tf.reduce_sum(x_val,axis=2), tf.int32)
        mask_idx = tf.bitwise.bitwise_xor(_tmp_target, _tmp_mask)
        masked_idx_expand = tf.cast(mask_idx[:,:,tf.newaxis], tf.float32)
        mask_idx = tf.cast(mask_idx, tf.bool)
        _val_metric.update_state(y_val,x_val + output*masked_idx_expand)
        loss_val = _loss_fn(y_val[mask_idx],output[mask_idx])
    
    val_rc_rate = _val_metric.result()
    print('Validation loss at epoch %d : %.4f'%(epoch, loss_val))
    print('Validation reconstruct rate at epoch %d : %.4f'%(epoch, val_rc_rate))
    _val_metric.reset_states()
    print('Time taken: %.2fs' % (time.time()-start_time))
