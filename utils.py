import os
import tarfile
import wget
import sklearn
import warnings

# CONTENTS OF DRIVE
def dir_walkthrough(path):

    '''
    shows the contents of the file directory
    '''
    import pandas as pd
    columns = ['dirname','folders','images','videos','others','TOTAL']
    big_list = []
    for dirname, folders, files in os.walk(path):
        img_extension = ['jpg','jpeg','png','webp','tiff','tif','bmp','gif']
        video_extension = ['mp4','m4a','3gp','mkv','xvid','vob','mov','wmv','avi']
        image_files = []
        video_files = []
        other_files = []

        for file in files:
            extension = file.split('.')[-1]
            if extension in img_extension:
                image_files.append(file)
            elif extension in video_extension:
                video_files.append(file)
            else:
                other_files.append(file)
            
        
        big_list.append({'Directory': dirname,
             'Folders': len(folders),
             'Images': len(image_files),
             'Videos': len(video_files),
             'Others':len(other_files),
             'Total Files': len(files)})
    
    df = pd.DataFrame(big_list)
    return df


import sys

def variable_memory():
	def sizeof_fmt(num, suffix='B'):
		''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
		for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
			if abs(num) < 1024.0:
				return "%3.1f %s%s" % (num, unit, suffix)
			num /= 1024.0
		return "%.1f %s%s" % (num, 'Yi', suffix)

	for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
							 key= lambda x: -x[1])[:10]:
		print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

		
def download_USEncoder():
    try:
        print('downloading universal sentence encoder...')
        use_filename = wget.download(use_url)

        print('Downloaded!')
        # Extracting
        os.makedirs('universal_sentence_encoder', exist_ok = True)
        print('Extracting universal sentence encoder....')
        # open file
        file = tarfile.open(use_filename)
        
        # extracting file
        file.extractall('./universal_sentence_encoder')
        
        file.close()
        print('Extracted.')
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
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names


def main():
	...
	
if __name__=="__main__":
	
	main()
