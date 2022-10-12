import tensorflow as tf
from tensorflow import keras
import os
import datetime
import random


def create_model_checkpoint(model_name, save_dir, monitor:str = 'val_loss',verbose: int = 0, save_best_only: bool = True, save_weights_only: bool = False,
                            mode: str = 'auto', save_freq='epoch', options=None, initial_value_threshold=None, **kwargs):
    model_name = model_name+'-'+ str(datetime.datetime.now())
    dir = os.path.join(save_dir, model_name)

    if not os.path.exists(dir):
        os.makedirs(dir)

    return tf.keras.callbacks.ModelCheckpoint(
                                                dir,
                                                monitor = monitor,
                                                verbose = verbose,
                                                save_best_only = save_best_only,
                                                save_weights_only = save_weights_only,
                                                mode = mode,
                                                save_freq = save_freq,
                                                options=options,
                                                initial_value_threshold = initial_value_threshold,
                                                **kwargs)










def main():
    ...

    
if __name__ == '__main__':
    main()
