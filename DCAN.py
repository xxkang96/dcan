import os
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow import keras
from sklearn.model_selection import train_test_split,KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Layer,Reshape,Embedding,Dropout,BatchNormalization
from tensorflow.keras import layers
from tqdm import tqdm
from sklearn import metrics
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from keras import losses
tqdm.pandas()
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
from sklearn.preprocessing import MinMaxScaler

noise_factor = 0.5


class AutoEncoderLayer(Layer):
    '''
    自动编码器层
    '''
    def __init__(self, unit=640, video_num=20,dropout_rate=0.):
        super(AutoEncoderLayer, self).__init__()
        self.unit = unit
        self.video_num = video_num
        self._dropout_rate = dropout_rate

    def build(self, input_shape):
        #free videos and paid videos 维度
        self.free_dim = input_shape[0][-1]
        self.paid_dim = input_shape[1][-1]

        #从上一层free传到下一层
        self.free_weight = self.add_weight(shape=(self.free_dim, self.unit),
                                           initializer=keras.initializers.RandomNormal(),
                                           trainable=True)
        #从上一层paid传到下一层
        self.paid_weight = self.add_weight(shape=(self.paid_dim, self.unit),
                                           initializer=keras.initializers.RandomNormal(),
                                           trainable=True)
        #从paid传给free的下一层
        self.paid_free = self.add_weight(shape=(self.paid_dim, self.unit),
                                         initializer=keras.initializers.RandomNormal(),
                                         trainable=True)
        #从free传给paid的下一层
        self.free_paid = self.add_weight(shape=(self.free_dim, self.unit),
                                         initializer=keras.initializers.RandomNormal(),
                                         trainable=True)

        #free的偏置
        self.free_bias = self.add_weight(shape=(self.unit,),
                                         initializer=keras.initializers.Zeros(),
                                         trainable=True)
        #paid的偏置
        self.paid_bias = self.add_weight(shape=(self.unit,),
                                         initializer=keras.initializers.Zeros(),
                                         trainable=True)
        super(AutoEncoderLayer,self).build(input_shape)
    def attention(self,queries,keys,values):

        matmul = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1])) # MatMul
        scaled_matmul = matmul / int(queries.shape[-1]) ** 0.5  # Scale

        softmax_out = K.softmax(scaled_matmul) # SoftMax
        # Dropout
        out = K.dropout(softmax_out, self._dropout_rate)
        outputs = K.batch_dot(out, values)

        return outputs

    def call(self, inputs):
        free,paid = inputs


        #free = Reshape([self.free_dim])(free)
        #paid = Reshape([self.paid_dim])(paid)
        free_matrix = Reshape([self.video_num,self.free_dim//self.video_num])(free)
        paid_matrix = Reshape([self.video_num,self.paid_dim//self.video_num])(paid)

        if K.dtype(free_matrix) != 'float32':  free_matrix = K.cast(free_matrix, 'float32')
        if K.dtype(free_matrix) != 'float32':  paid_matrix = K.cast(paid_matrix, 'float32')

        free_next = tf.matmul(free,self.free_weight)
        paid_next = tf.matmul(paid,self.paid_weight)

        paid_free_next = self.attention(free_matrix,paid_matrix,paid_matrix)
        free_paid_next = self.attention(paid_matrix,free_matrix,free_matrix)

        paid_free_next = Reshape([self.free_dim])(paid_free_next)
        free_paid_next = Reshape([self.paid_dim])(free_paid_next)

        paid_free_next = tf.matmul(paid_free_next,self.paid_free)
        free_paid_next = tf.matmul(free_paid_next,self.free_paid)

        free_output = free_next + paid_next + self.free_bias
        paid_output = paid_next + free_next + self.paid_bias

        free_output = tf.nn.tanh(free_output)
        paid_output = tf.nn.tanh(paid_output)

        return free_output,paid_output
    #def compute_output_shape(self,input_shape):
    #assert isinstance(input_shape,list)
    #shape_a,shape_b = input_shape
    #return [(shape_a[0],s)]

def add_noise(x,noise_factor):
    '''
    add noise
    给输入数据添加噪声
    '''
    x_noisy = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)

    x_noisy = np.clip(x_noisy, 0., 1.)     # limit into [0, 1]

    return x_noisy

def my_loss(y_true,y_pred):
    y_true_1,y_true_2 = y_true
    y_pred_1,y_pred_2 = y_pred
    loss_1 = losses.mean_squared_error(y_true_1,y_pred_1)
    loss_2 = losses.mean_squared_error(y_true_2,y_pred_2)
    return loss_1+loss_2

free_input = Input(shape=(64*20,))
paid_input = Input(shape=(64*20,))
free_layer_1,paid_layer_1 = AutoEncoderLayer(64*20)((free_input,paid_input))
free_layer_2,paid_layer_2 = AutoEncoderLayer(64*20)((free_layer_1,paid_layer_1))
free_layer_3,paid_layer_3 = AutoEncoderLayer(64*20)((free_layer_2,paid_layer_2))
free_layer_4,paid_layer_4 = AutoEncoderLayer(64*20)((free_layer_3,paid_layer_3))
free_paid_encoder = Model(inputs=[free_input,paid_input],outputs=[free_layer_4,paid_layer_4])


free_layer_5,paid_layer_5 = AutoEncoderLayer(64*20)((free_layer_4,paid_layer_4))
free_layer_6,paid_layer_6 = AutoEncoderLayer(64*20)((free_layer_5,paid_layer_5))
free_layer_7,paid_layer_7 = AutoEncoderLayer(64*20)((free_layer_6,paid_layer_6))
free_layer_8,paid_layer_8 = AutoEncoderLayer(64*20)((free_layer_7,paid_layer_7))

free_output,paid_output = Dense(64*20,activation='tanh',name='free')(free_layer_4),Dense(64*20,activation='tanh',name='paid')(paid_layer_4)
autoencoder = Model(inputs=[free_input,paid_input],outputs=[free_output,paid_output])
autoencoder.compile(optimizer='adam',loss={'free':'mse','paid':'mse'},loss_weights={'free':0.9,'paid':0.1})
autoencoder.summary()

dense_input = Input(shape=(76, ), dtype=tf.float32)
sparse_input = Input(shape=(15, ), dtype=tf.float32)
embed_reg=1e-4
embed_sparse_layers = [Embedding(input_dim=feat['feat_num'],
                                 input_length=1,
                                 output_dim=feat['embed_dim'],
                                 embeddings_initializer='random_uniform'
                                 )
                       for feat in sparse_features]
#embeddings_regularizer=l2(embed_reg)
userprofile_info = dense_input
for i in range(15):
    userprofile_info = tf.concat([userprofile_info,embed_sparse_layers[i](sparse_input[:, i])], axis=-1)

dcan_free_input = Input(shape=(64*20,))
dcan_paid_input = Input(shape=(64*20,))

dcan_free_preference,dcan_paid_preference = free_paid_encoder((dcan_free_input,dcan_paid_input))

userprofile_info = Dropout(0.5)(Dense(128,activation='tanh')(userprofile_info))
userprofile_info = Dropout(0.5)(Dense(128,activation='tanh')(userprofile_info))
dcan_free_preference = Dropout(0.5)(Dense(128,activation='tanh')(dcan_free_preference))
dcan_paid_preference = Dropout(0.5)(Dense(128,activation='tanh')(dcan_paid_preference))
#output = userprofile_info
output = layers.concatenate([dcan_free_preference,dcan_paid_preference,userprofile_info])
output = Dropout(0.5)(Dense(256,activation='tanh')(output))
output = Dropout(0.5)(Dense(256,activation='tanh')(output))
output = Dropout(0.5)(Dense(256,activation='tanh')(output))
#output = Dropout(0.5)(Dense(64,activation='tanh')(output))
output = Dense(1,activation='sigmoid')(output)
dcan = Model(inputs=[dcan_free_input,dcan_paid_input,dense_input,sparse_input],outputs=output)
dcan.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.01),
              metrics=[AUC()])

dcan.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
callbacks_list = [ tf.keras.callbacks.EarlyStopping( monitor='val_loss',
                                                     patience=3,mode='min'),
                   tf.keras.callbacks.ModelCheckpoint( filepath='rank_din.h5',
                                                       monitor='val_auc',
                                                       save_best_only=True,mod='max') ]
history = dcan.fit([x_train_free_input,x_train_paid_input,x_train_dense_input,x_train_sparse_input],y_train_label,epochs=100,batch_size=256,validation_data=([x_val_free_input,x_val_paid_input,x_val_dense_input,x_val_sparse_input],y_val_label),callbacks=[early_stopping])


pre_label = dcan.predict([x_test_free_input,x_test_paid_input,x_test_dense_input,x_test_sparse_input])
pre_prob = []
pre_is = []
for i in range(len(pre_label)):
    pre_prob.append(pre_label[i][0])
    if pre_label[i][0]>0.5:
        pre_is.append(1)
    else:
        pre_is.append(0)
print(metrics.classification_report(y_test_label, pre_is))
print(metrics.roc_auc_score(y_test_label, pre_prob))