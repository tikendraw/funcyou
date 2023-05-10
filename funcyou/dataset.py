
import random
from pathlib import Path
import sys
import os
import pathlib
import subprocess
import shutil
from tqdm import tqdm

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




def download_kaggle_dataset(api_command: str = None, url: str = None, unzip=False):
    """
    This function downloads a Kaggle dataset.

    Note: keep kaggle.json in the current directory or your Google Drive.
    Args:
        url: Link to the dataset, e.g., https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset
        api_command: Copied command from the Kaggle dataset page (easiest way, just copy and paste).
        unzip: True to unzip the downloaded zip file.
    """

    IN_COLAB = "google.colab" in sys.modules

    if IN_COLAB:
        kaggle_path = pathlib.Path("/root/.kaggle/")
    else:
        kaggle_path = pathlib.Path("~/.kaggle").expanduser()

    # Create the Kaggle path
    kaggle_path.mkdir(parents=True, exist_ok=True)

    # Copy kaggle.json
    kaggle_json_path = kaggle_path / "kaggle.json"
    shutil.copy("kaggle.json", kaggle_json_path)

    # Set permissions for kaggle.json
    kaggle_json_path.chmod(0o600)

    if url:
        url_parts = url.split("/")
        idx = url_parts.index("www.kaggle.com")
        kind = url_parts[idx + 1]

        if kind == "datasets":
            person = url_parts[idx + 2]
            dataname = url_parts[idx + 3]
            api_command = f"kaggle {kind} download -d {person}/{dataname}"
        else:
            dataname = url_parts[idx + 2]
            print("Note: Make sure you have agreed to Competition Rules. Else we can't download it.")

            api_command = f"kaggle {kind} download -c {dataname}"

    print(api_command)

    try:
        _extracted_from_download_kaggle_dataset_(api_command, unzip)
    except Exception as e:
        print("Error Occurred:", e)

def download_kaggle_dataset(api_command: str = None, url: str = None, unzip=False):
    """
    This function downloads a Kaggle dataset.

    Note: keep kaggle.json in the current directory or your Google Drive.
    Args:
        url: Link to the dataset, e.g., https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset
        api_command: Copied command from the Kaggle dataset page (easiest way, just copy and paste).
        unzip: True to unzip the downloaded zip file.
    """

    IN_COLAB = "google.colab" in sys.modules

    if IN_COLAB:
        kaggle_path = pathlib.Path("/root/.kaggle/")
    else:
        kaggle_path = pathlib.Path("~/.kaggle").expanduser()

    # Create the Kaggle path
    kaggle_path.mkdir(parents=True, exist_ok=True)

    # Copy kaggle.json
    kaggle_json_path = kaggle_path / "kaggle.json"
    shutil.copy("kaggle.json", kaggle_json_path)

    # Set permissions for kaggle.json
    kaggle_json_path.chmod(0o600)

    if url:
        url_parts = url.split("/")
        idx = url_parts.index("www.kaggle.com")
        kind = url_parts[idx + 1]

        if kind == "datasets":
            person = url_parts[idx + 2]
            dataname = url_parts[idx + 3]
            api_command = f"kaggle {kind} download -d {person}/{dataname}"
        else:
            dataname = url_parts[idx + 2]
            print("Note: Make sure you have agreed to Competition Rules. Else we can't download it.")

            api_command = f"kaggle {kind} download -c {dataname}"

    print(api_command)

    try:
        _extracted_from_download_kaggle_dataset_(api_command, unzip)
    except Exception as e:
        print("Error Occurred:", e)


# TODO Rename this here and in `download_kaggle_dataset`
def _extracted_from_download_kaggle_dataset_(api_command, unzip):
    command = f"{api_command} --unzip" if unzip else api_command
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    progress_bar = tqdm(total=None, unit='B', unit_scale=True)

    for line in process.stdout:
        if progress_match := re.search(r"(\d+)%", line):
            progress = float(progress_match.group(1))
            progress_bar.update(progress - progress_bar.n)

    process.stdout.close()
    return_code = process.wait()

    progress_bar.close()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


# TODO Rename this here and in `download_kaggle_dataset`
def _extracted_from_download_kaggle_dataset_(api_command, unzip):
    command = f"{api_command} --unzip" if unzip else api_command
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    progress_bar = tqdm(total=None, unit='B', unit_scale=True)

    for line in process.stdout:
        if progress_match := re.search(r"(\d+)%", line):
            progress = float(progress_match.group(1))
            progress_bar.update(progress - progress_bar.n)

    process.stdout.close()
    return_code = process.wait()

    progress_bar.close()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main():

    if __name__ == "__main__":
        main()
