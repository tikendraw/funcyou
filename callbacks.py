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





def create_tensorboard_callback(dir_name, experiment_name):
	"""
	Creates a TensorBoard callback instand to store log files.
	Stores log files with the filepath:
	"dir_name/experiment_name/current_datetime/"
	
	Args
	###########################################
	dir_name: target directory to store TensorBoard log files
	experiment_name: name of experiment directory (e.g. efficientnet_model_1)
	
	Returns
	##########################################
	A tensorboard callback
	"""
	log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	tensorboard_callback = tf.keras.callbacks.TensorBoard(
	  log_dir=log_dir
	)
	print(f"Saving TensorBoard log files to: {log_dir}")
	return tensorboard_callback


class DisplayCallback(tf.keras.callbacks.Callback):
	'''
	Prints model prediction and true values for dataset specified
	
	Args
	###########################################
	print_on_epoch:int : after how many epochs you want to print the results
	x:np.array  : input dataset (np.array)
	y:np.array  : output dataset (np.array)
	
	Returns
	##########################################
	A Display callback
	
	'''
	def __init__(self,print_on_epoch:int, x:np.array, y:np.array):
		self.print_on_epoch = print_on_epoch
		self.x = x
		self.y = y
		
	def predict(self):
		ypred = tf.squeeze(model.predict(self.x))
		print('ypred: ',ypred[0].numpy().astype('int'),'\nytrue: ',self.y[0])

	def on_epoch_end(self, epoch, logs=None):
		if (epoch%self.print_on_epoch==0):
			return self.predict()



def main():
    ...

    
if __name__ == '__main__':
	main()
