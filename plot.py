import os , random, shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

def plot_random_dataset(main_path, row:int = 2, col:int = 5, figsize:tuple= (15,6), recursive = True):

    '''
    Plots random images from the Path provided
    
    Args:
      main_path:str = path to the directory
      row:int       = number of rows needed
      col:int		  = number of columns needed
      figsize		 = size of the figure 

      Returns:
          A figure of row*cols of random images from dataset

    '''
    main_path = Path(main_path)
    all_images = []
    image_extension = ['jpg','jpeg','png','tiff','webp','svg','pjpeg']  
    
    if recursive:# walk through all dir and get images
          for dir, folders, files in os.walk(main_path):
              filles = [dir+'/'+i for i in files if i.split('.')[-1] in image_extension]
              all_images += filles
    else:
      for i in main_path.iterdir():
        if i.suffix in image_extension:
          all_images.append(i)
    #sampling random images 
    sample_images = random.sample(all_images, row*col)

    # plotting figure
    fig = plt.figure(figsize = figsize)

    if len(all_images) <= 0:
      print('No Image Found', len(all_images))

    for i,image in enumerate(sample_images):
        title = str(image).split('/')[-2]
        img = mpimg.imread(image)
        plt.subplot(row, col, i+1)
        plt.title(title)
        plt.imshow(img)
    
    plt.show()

def pairplot(df, figsize = (20,20), hue:str = None):
    num_cols = df.select_dtypes('number').columns
    print(num_cols)

    total_cols = len(num_cols)
    plot_num = 1

    plt.figure(figsize=(20,20))
    for col in num_cols:
        for col2 in num_cols:
            plt.subplot(total_cols, total_cols, plot_num)
            plot_num += 1

            if col == col2:
                sns.histplot(x = df[col], hue = df[hue])

            else:
                sns.scatterplot(x = df[col], y = df[col2], hue = df[hue])
				
				
from sklearn.metrics import confusion_matrix
import itertools
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 

  """Makes a labelled confusion matrix comparing predictions and ground truth labels.
  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.
  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.
  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")




















def main():
	pass

if __name__=="__main__":
	
	main()
