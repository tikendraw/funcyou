import os

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

	
def main():
	...
	
if __name__=="__main__":
	
	main()
