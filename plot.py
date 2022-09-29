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
    
    fig.suptitle(PATH)
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





















def main():
	pass

if __name__=="__main__":
	
	main()
