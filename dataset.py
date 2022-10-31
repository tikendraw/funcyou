import os , random, shutil
from pathlib import Path

#make data splitstr
def make_data_split(main_path, test = True, test_split_ratio:float  = .2 , val = True, val_split_ratio:float = .1, shuffle = True, unzip = True ):
    '''
    Make data split Train/test/val
    Note: keep a copy of the ORIGINAL DATASET
    
    Args:
        main_path = Dataset directory that has class subdirectory
        test_split_ratio = float value between 0 to 1
        val = bool to split for validation data
        val_split_ratio = float value for split
        shuffle = shuffle
    
    Returns:
        Splited DATASET (train/test/validation)
        
    '''
    main  = Path(main_path)
    class_names = sorted([i.name for i in main.iterdir()])
    print(class_names)
    total_files = len(os.listdir(main/class_names[0]))
    try:
        # create train/test/validation
        os.makedirs(main/'train')

        # creating test set
        if test :
            os.makedirs(main/'test')
            test_image_num = 0
            test_image_num = int(total_files*test_split_ratio)
            print('test images:',test_image_num)
            for class_name in class_names:
                class_path = main/class_name
                if shuffle:
                    sample_images = random.sample(os.listdir(class_path), test_image_num)

                else:
                    sample_images = sorted(os.listdir(class_path))[:test_image_num]

                sample_images = [class_path/i for i in sample_images]
                test_class = str(main/'test'/class_name)
                os.makedirs(test_class)
                for file in sample_images:
                    shutil.move(str(file), test_class)

        #creating validation dataset
        val_image_num = 0
        if val:
            os.makedirs(main/'validation')
            val_image_num = int(total_files*val_split_ratio)
            print('val images:',val_image_num)
            for class_name in class_names:
                class_path = main/class_name
                if shuffle:
                    sample_images = random.sample(os.listdir(class_path), val_image_num)

                else:
                    sample_images = sorted(os.listdir(class_path))[:val_image_num]

                sample_images = [class_path/i for i in sample_images]
                test_class = str(main/'validation'/class_name)
                os.makedirs(test_class)
                for file in sample_images:
                    shutil.move(str(file), test_class)

        for class_name in class_names:
            shutil.move(str(main/class_name), str(main/'train'/class_name))

        print('train images:', total_files - test_image_num - val_image_num)
    except Exception as e:
        print('Exception Occured: ',e)


#download_kaggle_dataset
def download_kaggle_dataset(api_command:str = None, url:str = None, unzip = False):

    '''
    This function downloads kaggle dataset
    Note: keep kaggle.json in curent dir or your google drive 

    args :
            data_link   = link to the dataset  user/datasetname
            api_command = copied command from the kaggle dataset page (easiest way, just copy and paste)
            kaggle      = place of kaggle.json ['current','gdrive']
            kind        = [datasets, compititions]
    
    '''
    import os

    kaggle_path = '/root/.kaggle/'

    #create the kaggle path
    if not os.path.exists(kaggle_path):
        os.makedirs(kaggle_path)
    
    #mount gdrive and copy kaggle.json

    os.system(f'cp kaggle.json {kaggle_path}')

    #giving permissino to file
    os.system('chmod 600 /root/.kaggle/kaggle.json')

    #downloadin dataset
    if api_command is not None :
        try:
            if unzip == True:
                os.system(api_command + ' --unzip')
            else :
                os.system(api_command )
        except Exception as e:
            print(e)

    elif data_link is not None:
        os.system(f'kaggle {kind} download {data_link} --unzip') 



def main():

    if __name__=="__main__":	
	    main()
