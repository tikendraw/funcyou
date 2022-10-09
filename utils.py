import os

def dir_walkthrough(path):

    '''
    shows the contents of the file directory
    '''
	print('		dirname			folders			images			videos			others			TOTAL')
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
            
        print(f'''{dirname}			{len(folders)}			{len(image_files)}			{len(video_files)}			{len(other_files)}			{len(files)}''')


def main():
	...
	
if __name__=="__main__":
	
	main()
