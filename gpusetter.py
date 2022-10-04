from os import environ
from gpustat import GPUStatCollection
from numpy import argsort
from tensorflow import config as tf_config

'''
Notes on GPU settings:
On a multi-gpu machine where the GPUs are interconnected at least (single host, multi device), there is a scaling when using TensorFlow MirroredStrategy
and multiple GPUs. For my case (four 1080 GTX GPUs) increasing the number of GPUs gives somewhat of a linear scaling as expected.
MirroredStrategy splits the minibatch among the participating devices and synchronizes the updates calculated by each model. Hence
the "mini_batch" fit parameter is known as the global minibatch, which is split evenly among the devices, so you want to multiply
the batch size with the number of devices.

Additional benefits can be derived from partitioning each physical GPUs to a number of logical GPUs, since our data is not very
intense (only 30 or so floating point numbers per example) and the models are relatively small. This is more GPU dependent what
the optimal partitioning is, since you want some memory buffer available for efficient tensorflow usage, but at least for me
it seems that four partitions with ~1700MB of VRAM each gives a ~15% additional improvement in epoch training times. But this
may also depend on your CPU having enough cores to keep all logical GPUs fed, so feel free to experiment or if you want to play
it safe, set the "split_gpu_into" parameter as 1

The model_manager class inspects if there are more than one visible GPU (including logical ones) and automatically uses the
MirroredStrategy if that is the case.

However when freezing the model, use of multiGPU strategies makes things difficult. To get around this, ModelManager class
will save the model and reload it from storage outside of the Strategy scope to avoid these. This is done if more than one
visible GPU is detected. The underlying reason stems from the tensorflow control_ops and the fact that 
convert_variables_to_constants_v2 cannot handle those gracefully.

If you dont want to play around with using multiple GPUs, just set n_gpus=1 and split_gpu_into=1 and it'll give you the
standard one GPU treatment.
'''
gpu_settings = {
    "n_gpus": 1,
    "min_vram": 1800,
    "split_gpu_into": 1
}

def set_gpus(n_gpus=gpu_settings["n_gpus"], min_vram=gpu_settings["min_vram"], split_gpu_into=gpu_settings["split_gpu_into"]):
    '''
    Configures the GPUs to be allocated for training, preferring the GPUs with most free VRAM.

    :param
        n_gpus: How many physical GPUs to allocate for this training process. Set to 0 to run on CPU.
        min_memory: How much free VRAM each physical GPU has to have. Too low value causes an error if the GPU runs out of memory when training.
                    This prevents TensorFlow from allocating all of the memory on GPU to the process.
        split_into: How many logical GPUs to split each physical GPU into. This can speed up the training due to distributed training.
                    Each physical GPU has to have min_memory * split_into VRAM available or an error is raised.

    :return
        None
    '''

    if n_gpus==0:
        environ['CUDA_VISIBLE_DEVICES'] = ''
    gpu_stats = GPUStatCollection.new_query()
    print(gpu_stats)
    print(n_gpus)
    gpu_ids = map(lambda gpu: int(gpu.entry['index']), gpu_stats)
    print(gpu_ids)
    gpu_freemem = map(lambda gpu: float(gpu.entry['memory.total']-gpu.entry['memory.used']), gpu_stats)
    pairs = list(zip(gpu_ids, gpu_freemem))
    valid_pairs = [pair for pair in pairs if pair[1] >= min_vram * split_gpu_into]
    print (valid_pairs)
    if len(valid_pairs) < n_gpus:
        raise ValueError("Not enough valid GPUs detected. Check if the machine has at least {n_gpus} GPUs with at least {min_vram * split_gpu_into}MB free VRAM or set a lower --n_gpus value")
    sorted_indices = list(argsort([mem[1] for mem in valid_pairs]))[::-1]
    sorted_pairs = [valid_pairs[i] for i in sorted_indices]
    if n_gpus != 0:
        print("Setting {n_gpus} physical GPUs split into {n_gpus * split_gpu_into} logical GPUs with {min_vram}MB VRAM each for this training")
    else:
        print("Training on CPU")
    environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    devices = ",".join([str(pair[0]) for pair in sorted_pairs[:n_gpus]])
    environ['CUDA_VISIBLE_DEVICES'] = devices
    if split_gpu_into > 1:
        physical_devices = tf_config.list_physical_devices('GPU')
        for device in physical_devices:
            tf_config.set_logical_device_configuration(
                device,
                [tf_config.LogicalDeviceConfiguration(memory_limit=min_vram) for _ in range(split_gpu_into)]
            )
    else:
      physical_devices = tf_config.list_physical_devices('GPU')
      #line added not to fill the entire gpu for no reason...
      tf_config.experimental.set_memory_growth(physical_devices[0], True)

set_gpus()
