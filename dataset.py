import os , random, shutil
from pathlib import Path

def make_data_split(main_path, test = True, test_split_ratio:float  = .2 , val = True, val_split_ratio:float = .1, shuffle = True):
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

def main():
	pass
if __name__=="__main__":
	
	main()
