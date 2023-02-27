from keras import callbacks,backend
import tensorflow as tf
class ReconstructCallbacks(callbacks.Callback):
    def __init__(self, training_data, validation_data):
        super().__init__()
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.reconst_rates = []
        self.reconst_rates_val = []
        self.reconst_degrees = []
        self.reconst_degrees_val = []
    
    def on_batch_begin(self, batch, logs=None):
        return 
    
    def on_batch_end(self, batch, logs=None):
        return
    
    def on_train_begin(self, logs=None):
        return 
    
    def on_train_end(self, logs=None):
        return

    def on_epoch_begin(self, epoch, logs=None):
        return 
    

    def on_epoch_end(self, epoch, logs=None):
        def calReconstructRate(y_true,y_pred):
        #     res = []
        #     for _ in range(y_true.shape[0]):
        #         val_len = 0
        #         hit_len = 0
        #         for __ in range(y_true.shape[1]):
        #             if tf.reduce_sum(y_true[_,__,:]) == 1:
        #                 val_len += 1
        #                 if(tf.argmax(y_true[_,__,:]) == tf.argmax(y_pred[_,__,:])):
        #                     hit_len += 1
        #             else:
        #                 break
        #         res.append(hit_len/val_len)
        #     return sum(res)/len(res)
            res = []
            out = tf.equal(tf.argmax(y_true, axis=1),tf.argmax(y_pred,axis=1))
            out = tf.cast(out, tf.int32)
            sum = tf.reduce_sum(out)
            return int(sum) / (out.shape[0]*out.shape[1])  
        
        y_pred = self.model.predict(self.x)
        y_pred_val = self.model.predict(self.x_val)

        reconst_rate = calReconstructRate(self.y, y_pred)
        reconst_rate_val = calReconstructRate(self.y_val, y_pred_val)
        reconst_degree = float(tf.reduce_sum(self.y*y_pred)/tf.reduce_sum(self.y))
        reconst_degree_val = float(tf.reduce_sum(self.y_val*y_pred_val)/tf.reduce_sum(self.y_val))

        self.reconst_rates.append(reconst_rate)
        self.reconst_rates_val.append(reconst_rate_val)
        self.reconst_degrees.append(reconst_degree)
        self.reconst_degrees_val.append(reconst_degree_val)
        print('\nRecontruct Rate : {}\tReconstruct Rate for Val :{}'.format(
            reconst_rate,reconst_rate_val
        ))
        print('Reconstruct Degree : {}\tReconstruct Degree for Val : {}'.format(
            reconst_degree,reconst_degree_val
        ))