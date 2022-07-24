

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import shutil


# ## download data from kaggle

class Datasets:
    def __init__(self) -> None:
        pass

    class Kaggle:
        def __init__(self) -> None:
            os.system('pip install kaggle')
            os.system('pip install wheel')
            os.system('pip install kaggle-cli')

            def get_dataset(dataset:str = None):
                ''' First Download your Kaggle api json to your Download folder'''
                try:
                    
                    os.system('mkdir  ~/.kaggle')
                    os.system('cp Downloads/kaggle.json ~/.kaggle')
                    os.system('chmod 600 ~/.kaggle/kaggle.json')
                    
                    if dataset is not None:
                        try:
                            os.system("kaggle datasets download -d $dataset -p './Dataset/' --unzip")
                #             shutil.unpack_archive(dataset.split('/')[-1]+'.zip','./Dataset')
                        except Exception as e:
                            print( 'Exeption occured :', e)
                    else:
                        print('Provide the parameters')
                except Exception as E:
                    print('Exception Occured : ', E)



            def kaggle_dataset_download(dataset:str = None):
                src_file = '~/Downloads/kaggle.json'
                dest_file = '~/.kaggle'
                '''Download your kaggle api json file to Downloads directory,
                
                data_address looks like username/datasetname
                e.g : zusmani/uberdrives 
                
                '''
                if dataset is None:
                    print('Provide dataset address')
                else:
                    try:
                        shutil.copy2(src_file, dest_file)
                        os.system(' chmod 600 ~/.kaggle/kaggle.json')
                        os.system("kaggle datasets download -d $dataset -p './Dataset/' --unzip")
                    except Exception as e:
                        print('Exception Occured: Kaggle dataset download function :', e)




def get_factors(x):
    factors = set()
    for i in range(1,round((x+1)/2)):
        if x%i==0:
            factors.add(i)
            factors.add(round(x/i))
    return list(factors)

def main():
    print('nothing here')


if __name__ == '__main__':
    main()
