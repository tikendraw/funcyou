import os 
from tqdm import tqdm
from PIL import Image

def resize_func( file_path:str, dest_path:str = None ,px:int  = 512, keep_aspect_ratio = True, quality:int = 100):
    '''
    Resizes photos in a directory with given paramters
    
    -----------------------------
    
    Paramters-----------------
        file_path:str =  folder which contains files
        
        dest_path:str =  folder where to keep the resized folder (if not provided
                        resized files will be kept in file_path)
                        
        px:int = max pixel range to resize upto
        
        quality:int = quality range between 1 to 100  
    '''
    
    if dest_path == None:
        dest_path = file_path
    
    for item in tqdm(os.listdir(file_path)):
        img_path = os.path.join(file_path, item)
        img_name = item.split('.')[:-1]
        
        img = Image.open(img_path)

        # print(img_path)
        w, h = img. size
        print('height:', h, 'width: ', w)

        ratio = h/w
        print(ratio)
        if keep_aspect_ratio:        
            if ratio < 1:
                h = px * ratio
                w = px 
                
                
            else:
                w = px/ratio
                h = px 
        else:
            h = px
            w = px
            
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        print('new height:', h, 'new width:',w)

        img_resized = img.resize((int(w),int(h)), Image.ANTIALIAS)
        img_resized.save(dest_path + f'/{img_name}_resized.jpg', quality=100)



def main(): 
    ...


if __name__ == '__main__':
    main()
