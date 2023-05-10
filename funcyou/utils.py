import os
import tarfile
import wget
import sklearn
import warnings
import pandas as pd
import mimetypes
from pathlib import Path
import pandas as pd
import mimetypes
import sys


def variable_memory():
    def sizeof_fmt(num, suffix="B"):
        """by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified"""
        for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, "Yi", suffix)

    print('''
    def sizeof_fmt(num, suffix="B"):
        """by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified"""
        for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, "Yi", suffix)
        
    
    for name, size in sorted(
        ((name, sys.getsizeof(value)) for name, value in locals().items()),
        key=lambda x: -x[1],
    )[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        '''

    )

    for name, size in sorted(
        ((name, sys.getsizeof(value)) for name, value in locals().items()),
        key=lambda x: -x[1],
    )[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


def download_USEncoder():
    try:
        print("downloading universal sentence encoder...")
        use_filename = wget.download(use_url)

        print("Downloaded!")
        # Extracting
        os.makedirs("universal_sentence_encoder", exist_ok=True)
        print("Extracting universal sentence encoder....")
        # open file
        file = tarfile.open(use_filename)

        # extracting file
        file.extractall("./universal_sentence_encoder")

        file.close()
        print("Extracted.")
    except Exception as e:
        print(e)


def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    # check_is_fitted(column_transformer)

    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == "drop" or (hasattr(column, "__len__") and not len(column)):
            return []
        if trans == "passthrough":
            if hasattr(column_transformer, "_df_columns"):
                return (
                    column
                    if (
                        (not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)
                    )
                    else column_transformer._df_columns[column]
                )
            indices = np.arange(column_transformer._n_features)
            return ["x%d" % i for i in indices[column]]
        if not hasattr(trans, "get_feature_names"):
            # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn(
                "Transformer %s (type %s) does not "
                "provide get_feature_names. "
                "Will return input column names if available"
                % (str(name), type(trans).__name__)
            )
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            return [] if column is None else [name + "__" + f for f in column]
        return [name + "__" + f for f in trans.get_feature_names()]

    ### Start of processing
    feature_names = []

    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [
            (name, trans, None, None)
            for step, name, trans in column_transformer._iter()
        ]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))

    for name, trans, column, _ in l_transformers:
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names) == 0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))

    return feature_names


def dir_walkthrough(directory, show_hidden_dir=False, exclude=None, include=None):
    """
    Walks through a directory and its subdirectories recursively and counts the number of files of each MIME type.

    Args:
        directory (str): The path to the directory.
        show_hidden_dir (bool): Whether to show hidden directories or not.
        exclude (list): A list of folder names to exclude.
        include (list): A list of folder names to include.

    Returns:
        pd.DataFrame: A dataframe containing the file counts.
    """
    if exclude is None:
        exclude = []
    if include is None:
        include = []
    final = []

    for root, dirs, files in os.walk(directory):
        dir_name = os.path.basename(root)

        # Exclude hidden directories
        if not show_hidden_dir:
            dirs[:] = [d for d in dirs if not d.startswith(".")]

        # Exclude directories in the exclude list
        if dir_name in exclude:
            dirs.clear()
            continue

        # Include only directories in the include list
        if include and dir_name not in include:
            # dirs.clear()
            continue

        file_counts = {
            "directory": root,
            "base": dir_name,
            "folders": len(dirs),
            "video": 0,
            "music": 0,
            "photos": 0,
            "application/zip": 0,
            "documents": 0,
            "others": 0,
            "total files": len(files),
        }

        for file in files:
            file_path = os.path.join(root, file)
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type is None:
                file_counts["others"] += 1

            elif mime_type.startswith("video/"):
                file_counts["video"] += 1
            elif mime_type.startswith("audio/"):
                file_counts["music"] += 1
            elif mime_type.startswith("image/"):
                file_counts["photos"] += 1
            elif mime_type == "application/zip":
                file_counts["application/zip"] += 1
            elif (
                mime_type.startswith("text/")
                or mime_type.endswith("msword")
                or mime_type.endswith("pdf")
            ):
                file_counts["documents"] += 1
        final.append(file_counts)

    return pd.DataFrame(final)


def main():
    ...


if __name__ == "__main__":

    main()
