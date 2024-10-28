import os
import shutil
import sys
import tensorflow as tf

from config import network_param_dict, make_model
from helper_functions import get_timestamp

class ModelManager():
    '''
    Class for handling the neural network. It can reinitialize model from stored initial weights, save it to a folder,
    freeze it as .pb for deployment in CMSSW and produce plots for monitoring performance of the model.
    If you have asked gpusetter.py to use more than one GPU, model is initialized with MirroredStrategy to leverage
    the additional computing power.
    '''

    def __init__(self, n_regular_inputs, min_values, max_values):
        self.n_regular_inputs = n_regular_inputs
        self._min_values = min_values
        self._max_values = max_values
        self._initialized_network_storage_path = "network_initialization"
        self._model = None
        self._result_dir = None
        self._hypertuned_model = None
        self._strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())if len(tf.config.list_logical_devices('GPU')) > 1 else tf.distribute.get_strategy()



    def initalize_model(self, reinitialize_model=False):
        """
        Checks if this network architecture already has initialized weights stored and uses them, unless
        the desired architecture has been changed, or new initialization is explicitly asked for.

        Currently check is done from the number of parameters in the model. Its unlikely that two
        different architectures end up having the exact same number of parameters so this shouldn't become
        a problem.

        :param
        n_regular_inputs: int, number of non-categorical inputs to the network
        n_categories: int, number of categories for the categorical input (trk_originalAlgo)
        min_values: ndarray of length n_regular_inputs, minimum values non-categorical inputs are clipped to
        max_values: ndarray of length n_regular_inputs, maximum values non-categorical inputs are clipped to

        :return:
        """
        # if len(tf.config.list_logical_devices('GPU')) > 1:
        #     self._strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()).scope()
        # else:
        #     self._strategy = tf.distribute.get_strategy().scope()
        with self._strategy.scope():
            #Check if a tuned model exists and use that
            if os.path.exists("hypertuned_model") and not reinitialize_model:
                print("Found it")
                stored_model = tf.keras.models.load_model("hypertuned_model")
            else:
                current_model = make_model(self.n_regular_inputs, self._min_values, self._max_values)
                if not reinitialize_model:
                    #Check if initialized weights already exist or should new ones be created
                    if os.path.exists(f"{self._initialized_network_storage_path}"):
                        stored_model = tf.keras.models.load_model(f"{self._initialized_network_storage_path}")

                    #Checks if the currently desired model differs from the stored model initialization from number of parameters
                    #if (stored_model.count_params() != current_model.count_params()):
                    #    print("Creating new model")
                    #    stored_model = current_model
                    #    shutil.rmtree(f"{self._initialized_network_storage_path}")
                    #    stored_model.save(f"{self._initialized_network_storage_path}")
                    else:
                        print("Using saved initialization.")

                else:
                    print("Creating new model")
                    stored_model = current_model
                    stored_model.save(f"{self._initialized_network_storage_path}.keras")
                #==========================================================================

            #Compile model.
            metrics = [tf.metrics.AUC()]
            stored_model.compile(optimizer=tf.keras.optimizers.Adam(lr=network_param_dict["lr"], decay=network_param_dict["decay"]),
                               #optimizer=tf.keras.optimizers.SGD(lr=network_param_dict["lr"], decay=network_param_dict["decay"]),
                               loss=network_param_dict["loss"],
                               #loss="binary_crossentropy",
                               metrics=metrics)
            print(stored_model.summary())
            self._model = stored_model
            #==========================================================================

    def get_model(self):
        if self._model is None:
            raise Exception("Initialize model first by calling .initialize_model()")
        return self._model

    def save_model(self, model):
        #Create directory to store results
        time = get_timestamp()
        self._result_dir = f"results/training_run_{time}"
        [os.mkdir(x) for x in [f"{self._result_dir}/{subdir}" for subdir in ["", "model", "plots"]]]

        #Save in TF savedModel format
        model.save(f'{self._result_dir}/model')

        #Freeze model as protobuf (.pb), launches another clean process to take care of it. See model_freeze.py for
        #more explanation. If sys.prefix points to wrong python executable, this will probably fail in an odd manner.
        python_exec = f'{sys.prefix}/bin/python'
        os.system(f'{python_exec} model_freeze.py {self._result_dir}')

