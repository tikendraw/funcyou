import os, random, shutil
from pathlib import Path
import os
import sys

# make data splitstr
def make_data_split(
    main_path,
    test=True,
    test_split_ratio: float = 0.2,
    val=True,
    val_split_ratio: float = 0.1,
    shuffle=True,
    unzip=True,
):
    """
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

    """
    main = Path(main_path)
    class_names = sorted([i.name for i in main.iterdir()])
    print(class_names)
    total_files = len(os.listdir(main / class_names[0]))
    try:
        # create train/test/validation
        os.makedirs(main / "train")

        # creating test set
        if test:
            os.makedirs(main / "test")
            test_image_num = 0
            test_image_num = int(total_files * test_split_ratio)
            print("test images:", test_image_num)
            for class_name in class_names:
                class_path = main / class_name
                if shuffle:
                    sample_images = random.sample(
                        os.listdir(class_path), test_image_num
                    )

                else:
                    sample_images = sorted(os.listdir(class_path))[:test_image_num]

                sample_images = [class_path / i for i in sample_images]
                test_class = str(main / "test" / class_name)
                os.makedirs(test_class)
                for file in sample_images:
                    shutil.move(str(file), test_class)

        # creating validation dataset
        val_image_num = 0
        if val:
            os.makedirs(main / "validation")
            val_image_num = int(total_files * val_split_ratio)
            print("val images:", val_image_num)
            for class_name in class_names:
                class_path = main / class_name
                if shuffle:
                    sample_images = random.sample(os.listdir(class_path), val_image_num)

                else:
                    sample_images = sorted(os.listdir(class_path))[:val_image_num]

                sample_images = [class_path / i for i in sample_images]
                test_class = str(main / "validation" / class_name)
                os.makedirs(test_class)
                for file in sample_images:
                    shutil.move(str(file), test_class)

        for class_name in class_names:
            shutil.move(str(main / class_name), str(main / "train" / class_name))

        print("train images:", total_files - test_image_num - val_image_num)
    except Exception as e:
        print("Exception Occured: ", e)


import os, random, shutil
from pathlib import Path
import os
import sys

# download_kaggle_dataset
def download_kaggle_dataset(api_command: str = None, url: str = None, unzip=False):

    """
    This function downloads kaggle dataset

    Note: keep kaggle.json in curent dir or your google drive
    args :

            url  = link to the dataset  https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset

            api_command = copied command from the kaggle dataset page (easiest way, just copy and paste)

            unzip = True , unzip the downloaded zip file
    """

    IN_COLAB = "google.colab" in sys.modules

    if IN_COLAB:
        kaggle_path = "/root/.kaggle/"

    else:
        kaggle_path = "~/.kaggle"

    # create the kaggle path
    if not os.path.exists(kaggle_path):
        os.makedirs(kaggle_path)

    # mount gdrive and copy kaggle.json
    try:
        os.system(f"cp kaggle.json {kaggle_path}")
    except Exception as e:
        raise ValueError(e)

    # #giving permissino to file
    kaggle_file_path = os.path.join(kaggle_path, "kaggle.json")
    os.system(f"chmod 600 {kaggle_file_path}")

    if url:
        url = url.split("/")
        idx = url.index("www.kaggle.com")
        kind = url[idx + 1]

        if kind == "datasets":
            person = url[idx + 2]
            dataname = url[idx + 3]

            api_command = f"kaggle {kind} download -d {person}/{dataname}"
        else:
            dataname = url[idx + 2]
            print(
                "Note: Make Sure you have agreed to Competition Rules. Else we can't download it"
            )
            api_command = f"kaggle {kind} download -c {dataname}"

    print(api_command)

    try:
        if unzip == True:
            print(os.system(api_command + " --unzip"), flush=True)
        else:
            print(os.system(api_command), flush=True)
    except Exception as e:
        print("Error Occured: ", e)


def main():

    if __name__ == "__main__":
        main()
