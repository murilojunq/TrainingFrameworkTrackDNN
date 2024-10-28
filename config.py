import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Add, Concatenate, Dropout
from network.custom import InputSanitizerLayer, OneHotLayer

'''
Collects parameters and variables one might often change while testing different training setups into one place
'''

'''
Commonly tuned fit parameters when training a network. NOTE: Batch size is multiplied with the number of logical gpus,
because each minibatch is split evenly for each gpu when running distributed workloads with many GPUs available
'''
n_gpus = len(tf.config.list_logical_devices('GPU'))
minibatch_multiplier = n_gpus if n_gpus > 0 else 1

network_callbacks =[]
network_fit_param = {
    "batch_size":       512*minibatch_multiplier,
    "epochs":           5,
    #"callbacks":        network_callbacks,
    #"callbacks":        [TestCallback()],
}

'''
Commonly tuned parameters when training network
'''
network_param_dict = {
    "lr":                       1e-4,
    "decay":                    3e-6,                   
    #"loss":                     tf.keras.losses.MSE,
    "loss":                     tf.keras.losses.binary_crossentropy,
    #"loss":                     tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
    "kernel_regularization":    tf.keras.regularizers.l2(0.0),
    "kernel_initializer":       tf.keras.initializers.lecun_normal()
    
}

'''
Model architecture
'''
def make_model(n_regular_inputs, min_values, max_values):
    _kernel_regularization = network_param_dict["kernel_regularization"]
    _kernel_initializer = network_param_dict["kernel_initializer"]

    reg_inp = Input((n_regular_inputs,), name="regular_input_layer")
    cat_inp = Input((1,), name="categorical_input_layer")
    
    active='elu'
    hidden_wid=32

    sanitized_inp = InputSanitizerLayer(min_values, max_values)(reg_inp)
    one_hot_inp = OneHotLayer()(cat_inp)

    x = Concatenate()([sanitized_inp, one_hot_inp])

    x = Dense(units=256,
              activation=active,
              kernel_regularizer=_kernel_regularization,
              kernel_initializer=_kernel_initializer)(x)
    x = Dense(units=128,
              activation=active,
              kernel_regularizer=_kernel_regularization,
              kernel_initializer=_kernel_initializer)(x)
    x = Dense(units=64,
              activation=active,
              kernel_regularizer=_kernel_regularization,
              kernel_initializer=_kernel_initializer)(x)
    x = Dense(units=32,
              activation=active,
              kernel_regularizer=_kernel_regularization,
              kernel_initializer=_kernel_initializer)(x)
    x_in = Dense(units=hidden_wid,
              activation=active,
              kernel_regularizer=_kernel_regularization,
              kernel_initializer=_kernel_initializer)(x)
    for i in range(5):
        
         x = Dense(units=hidden_wid,
               activation=active,
               kernel_regularizer=_kernel_regularization,
               kernel_initializer=_kernel_initializer)(x_in) 
         x = Dense(units=hidden_wid,
               activation=active,
               kernel_regularizer=_kernel_regularization,
               kernel_initializer=_kernel_initializer)(x)    
         x = Dense(units=hidden_wid,
               activation=active,
               kernel_regularizer=_kernel_regularization,
               kernel_initializer=_kernel_initializer)(x)   

         x_in = Add()([x_in, x])


    # Bias initializer is the "good initializer" giving the expectation value for two
    # equally balanced classes. See "init well" from A.Karpathy https://karpathy.github.io/2019/04/25/recipe/
    out = Dense(units=1,
                activation='sigmoid',
                kernel_initializer=_kernel_initializer,
                kernel_regularizer=_kernel_regularization,
                bias_initializer=tf.keras.initializers.Constant(0.5))(x_in)

    model = tf.keras.Model([reg_inp, cat_inp], out)
    return model


