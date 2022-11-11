import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # #Silences unnecessary spam from TensorFlow libraries. Set to 0 for full output
import gpusetter
import json
#from helper_functions import balance_true_and_fakes, balance_iterations
from timer import Timer
from network import ModelManager
import config as cfg
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import tensorflow as tf
from functools import partial
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime

device_name = tf.test.gpu_device_name()
if not device_name:
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

nfea=29

def to_input(arg, x):
    wgts=arg
    x = tf.io.parse_tensor(x, tf.float32)

    wgt1=1.
    wgt2=1.
    iterates=[4, 5, 6, 7, 8, 9, 10, 11, 22, 23, 24]
    for i in iterates:
        a=float(wgts['T'][i])
        b=float(wgts['F'][i])
        wgt1=tf.cond(tf.math.equal(x[-2],i), lambda: a, lambda: wgt1)
        wgt2=tf.cond(tf.math.equal(x[-2],i), lambda: b, lambda: wgt2)
        
    wgt=tf.cond(tf.math.less(0.5,x[-1]), lambda: wgt1, lambda: wgt2)
 
    return ({'regular_input_layer':x[:nfea],'categorical_input_layer':x[-2]}, x[-1], tf.reshape(wgt, [1]))

def pre_filter_input(x):
    
    x = tf.io.parse_tensor(x, tf.float32)
    bool3=tf.cond(tf.less(x[22],0), lambda: False, lambda: True)
    xn=tf.math.is_nan(x)

    bool_=tf.cond(tf.math.reduce_any(xn), lambda: False, lambda: True)
    bool1=tf.cond(tf.math.equal(x[-2],13), lambda: False, lambda: True)
    bool2=tf.cond(tf.math.equal(x[-2],14), lambda: False, lambda: True)

    bool_ = bool_ and bool1
    bool_ = bool_ and bool2
    bool_ = bool_ and bool3
    return bool_

def to_input2(arg, x):
    wgts=arg
    x = tf.io.parse_tensor(x, tf.float32)   
 
    wgt1=1.
    wgt2=1.
    iterates=[4, 5, 6, 7, 8, 9, 10, 11, 22, 23, 24]
    for i in iterates:
        a=float(wgts['T'][i])
        b=float(wgts['F'][i])
        wgt1=tf.cond(tf.math.equal(x[-2],i), lambda: a, lambda: wgt1)
        wgt2=tf.cond(tf.math.equal(x[-2],i), lambda: b, lambda: wgt2)

    wgt=tf.cond(tf.math.less(0.5,x[-1]), lambda: wgt1, lambda: wgt2)

    return ({'regular_input_layer':x[:nfea],'categorical_input_layer':x[-2]}, x[-1], tf.reshape(wgt, [1]))

def filter_input(x,y,z):
    x1=x["regular_input_layer"]
    bool3=tf.cond(tf.less(x1[22],0), lambda: False, lambda: True)
    x1=tf.math.is_nan(x1)

    bool_=tf.cond(tf.math.reduce_any(x1), lambda: False, lambda: True)
    bool1=tf.cond(tf.math.equal(x['categorical_input_layer'],13), lambda: False, lambda: True)
    bool2=tf.cond(tf.math.equal(x['categorical_input_layer'],14), lambda: False, lambda: True)
    
    bool_ = bool_ and bool1
    bool_ = bool_ and bool2
    bool_ = bool_ and bool3
    return bool_ 

def mixing(*args):
    x=tf.concat(args,axis=0)
    x=tf.random.shuffle(x) 
    return x


path='/data2/legianni/TrainingFrameworkTrackDNN/tfrecord/'
processes=['TT']
def main():
    
    #load train dataset
    datasets={}
    stats_fact={} 
    batch_size=cfg.network_fit_param['batch_size']
    print("load TFRecord")
    
    for process in processes:
        
        files=glob.glob(path+process+"/train/*")
        files=tuple(files)
        datasets[process]=tf.data.TFRecordDataset(files)
        stat=open(path+process+'.txt')
        stats_fact[process]=json.loads(stat.read())

    print("produce weight sheet")

    wgts={}
    wgts['T']=[0]*26
    wgts['F']=[0]*26
    for process in processes:

        stat=stats_fact[process]
        for i in range(25):

            wgts['T'][25]+=stat['ntrk_t'][i]
            wgts['F'][25]+=stat['ntrk_f'][i]
            wgts['T'][i]+=stat['ntrk_t'][i]
            wgts['F'][i]+=stat['ntrk_f'][i]

    print ('true tracks list')
    print (wgts['T'][:])
    print ('false tracks list')
    print (wgts['F'][:])

    tot_sample_size=wgts['T'][25]+wgts['F'][25]
    for i in range(25):
        try:
            wgts['T'][i]=0.1*float(tot_sample_size)/wgts['T'][i]
            wgts['F'][i]=0.1*float(tot_sample_size)/wgts['F'][i]
        except Exception:
            wgts['T'][i]=0
            wgts['F'][i]=0

    print('split dataset into train and val')
    val_datasets=[]
    train_datasets=[]
    sample_ratio=[]
    for process in processes:

        sample_size = sum(stats_fact[process]['ntrk_t'])+sum(stats_fact[process]['ntrk_f'])
        sample_ratio.append(round(batch_size*float(sample_size)/float(tot_sample_size)))
        # use a subset to test
        # datasets[process]=datasets[process].take(round(0.0001*sample_size))
        train_size = round(0.95*sample_size)
        print ('full size is %i'%sample_size)
        print ('train size is %i'%train_size)
        train_datasets.append(datasets[process].take(train_size))
        val_datasets.append(datasets[process].skip(train_size))

    #sample_ratio[0]=sample_ratio[0]-1
    print ('concat and mixing dataset')
    print (sample_ratio)
    for i in range(len(sample_ratio)):
        train_datasets[i]=train_datasets[i].batch(batch_size=sample_ratio[i], drop_remainder=True).prefetch(sample_ratio[i]*10)
        val_datasets[i]=val_datasets[i].batch(batch_size=sample_ratio[i], drop_remainder=True).prefetch(sample_ratio[i]*10)

    train_data=tf.data.Dataset.zip(tuple(train_datasets))

    train_data=train_data.map(mixing)
    train_data=train_data.unbatch()
    val_data=tf.data.Dataset.zip(tuple(val_datasets))
    val_data=val_data.map(mixing)
    val_data=val_data.unbatch()
    
    #train_data=train_data.filter(pre_filter_input)
    #print (train_data)
    #train_data=train_data.map(partial(to_input2, wgts), num_parallel_calls=8)
    #print (train_data)
    train_data=train_data.map(partial(to_input, wgts), num_parallel_calls=8)           
    #print (train_data)
    #quit()
    #train_data=train_data.filter(filter_input)
    #print (train_data)

    val_data=val_data.map(partial(to_input, wgts), num_parallel_calls=8)
    #val_data=val_data.filter(filter_input)
    
    wt=20*[0]
    wf=20*[0]

    print("load max of the input features")
    train_data = train_data.batch(batch_size, drop_remainder=True).prefetch(batch_size*10)
    val_data = val_data.batch(batch_size, drop_remainder=True,).prefetch(batch_size*10)
    
    max_=[0]*nfea
    min_=[0]*nfea
    for process in processes:

        stat=stats_fact[process]
        min_=[(stat['min'][i] if stat['min'][i]<min_[i] else min_[i]) for i in range(nfea)]
        max_=[(stat['max'][i] if stat['max'][i]>max_[i] else max_[i]) for i in range(nfea)]

    min_abs_log=np.array(min_)
    max_abs_log=np.array(max_)     

    train_data.shuffle(buffer_size=batch_size*10)
    val_data.shuffle(buffer_size=batch_size*10)
    #train_data = train_data.prefetch(batch_size*10)
    #val_data = val_data.prefetch(batch_size*10) 
    model_manager = ModelManager(n_regular_inputs=nfea,
                                 min_values=min_abs_log,
                                 max_values=max_abs_log)

    model_manager.initalize_model(reinitialize_model=True)

    model = model_manager.get_model()
    
    print ('start train')
    class LossHistory(tf.keras.callbacks.Callback): 
        def on_train_begin(self, logs={}): 
            self.losses = [] 
            self.auc = []
        def on_batch_end(self, batch, logs={}): 
            self.losses.append(logs.get('loss'))
            self.auc.append(logs.get('auc'))
            #print ("and end batch", len(self.losses))
            if len(self.losses)%10000 ==0: print("loss %f, auc %f" %(logs.get('loss'), logs.get('auc')))
        def on_epoch_end(self, epoch, logs=None):
            model.save(f'./REDOTB/modelTEST-{epoch:02d}')

    loss_history = LossHistory()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,profile_batch = '500,520')
    #https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
   
    with Timer("Training"):
        history = model.fit(train_data,
                  #sample_weight=train_weights,
                  validation_data=val_data,
                  **cfg.network_fit_param,
                  callbacks=[loss_history, tensorboard_callback], 
                  #batch_size=512
                  #epochs=20,
        )
    print ('plot loss')
    plt.plot(range(1,cfg.network_fit_param['epochs']+1),history.history['auc'])
    plt.xlabel('epoch')
    plt.ylabel('auc')
    plt.legend(['train','validation'],loc='upper right')
    plt.savefig('plots/auc.pdf')
    plt.clf()
    plt.plot(range(1,len(loss_history.losses)+1),loss_history.losses) 
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig('plots/loss_iteration.pdf')
    plt.clf()
    with Timer("Save"):
        model_manager.save_model(model)


if __name__ == "__main__":
  main()
