import matplotlib as mpl

def plot_history(history, plot = ['loss','accuracy'], split = ['train','val'], epoch:int = None, figsize = (20,10),colors = ['r','b'], **plot_kwargs ):
    
    ''' Plots History

    Arguments:
    ###############
    histroy 	:	History to plot
    plot:list	:   what to plot (what metrics you want to compare)  -> ['loss', 'accuracy']  
    split:list  :   what split to compare -> ['train', 'val']
    epoch:int   :   for how many epochs to comapre (cannot be greater than highest epoch of histories)
    figsize:tuple:  size of plot
    plot_kwargs :   kwargs to plt.plot to customize plot

    Returns:
    ##############
    Plots history 

    '''

    try:
        mpl.rcParams['figure.dpi'] = 500
        
        if not len(colors) == len(split):
            raise ValueError('not enogh colors')
        
        cols = []
        for i in plot:
            for j in split:
                if j == 'val':
                    cols.append(j+'_'+i)
                else:
                    cols.append(i)
        
        #compare to epoch
        if epoch is None:
            epoch = history.epoch

        def display(col, plot_num, history, epoch:int = None,label = None, **plot_kwargs):
            plt.subplot(len(plot),len(split),plot_num)
            plt.grid(True)
            
            if epoch == None:
                epoch = history.epoch
            
            if label is None:
                label=history.model.name
                
            plt.plot(epoch, pd.DataFrame(history.history)[col], label=label, **plot_kwargs)
            plt.title((' '.join(col.split('_'))).upper())
            plt.xlabel('epochs')
            plt.legend()
        
        plt.figure(figsize = figsize)
        plot_title = " ".join(plot).upper()+" PLOT"
        plt.suptitle(plot_title)

        for plot_num,col in enumerate(plot,1):
            display(col, plot_num, history, epoch, label = 'train',color = colors[0], **plot_kwargs)
            if 'val' in split:
                display('val_'+col, plot_num, history, epoch,label = 'val' ,color = colors[1])
    except Exception as e:
        print('Error Occured: ',e)

		
def create_bert(trainable=False):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    encoder_inputs = preprocessor(text_input)
    encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/2",
        trainable=trainable)
    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"]      
    sequence_output = outputs["sequence_output"]  
    embedding_model = tf.keras.Model(text_input, pooled_output)
    return embedding_model
		
		
if __name__=='__main__':
	print('hello')
