import tensorflow as tf 
import numpy as np
import sys,os,glob

ahlf_dir = "./AHLF"

sys.path.append(ahlf_dir)

from network_removed_dropout import network
#from network import network

ch = 64
net = network([ch,ch,ch,ch,ch,ch,ch,ch,ch,ch,ch,ch,ch],kernel_size=2,padding='same',dropout=.2) 
inp = tf.keras.layers.Input((3600,2))
sigm = net(inp)
model = tf.keras.Model(inputs=inp,outputs=sigm)
model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.BinaryCrossentropy(from_logits=False))

#####################################################################

batch_size=64
model_weights = 'alpha'
model.load_weights(os.path.join(ahlf_dir,'./model/%s_model_weights.hdf5'%model_weights))
