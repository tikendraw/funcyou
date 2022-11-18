import re
import json

with open('contractions.json') as f:
   contractions_dict = json.load(f)

# very necessary function you can't tell differece between "’" and "'", I don't even have that symbol in my keyboard
def text_cleaning_apos(text):
    text = str(text)
    text = text.lower()
    text = re.sub("’", "'", text) # removing punctuation
    return text


# FUNCTIONS TO EXPAND CONTRACTIONS
def cont_to_exp(x):
    x = str(x).lower()
    xsplited = x.split(' ')
    exp_sentence = []
    for s in x.split():
        if s in contractions_dict.keys():
            
            s = contractions_dict.get(s)
        exp_sentence.append(s)
        
    x = ' '.join(exp_sentence)
    return x

def text_cleaning(text):
    text = str(text)
    text = text.lower()
    text = re.sub("[^a-zA-Z]", " ", text) # removing punctuation
    # remove special characters from text column
    text = re.sub('[#,@,&]', '',text)
    # Remove digits
    text = re.sub('\d*','', text)
    # remove "'s"
    text = re.sub("'s",'', text)
    #Remove www
    text = re.sub('w{3}','', text)
    # remove urls
    text = re.sub("http\S+", "", text)
    # remove multiple spaces with single space
    text = re.sub('\s+', ' ', text)
    #remove all single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    return text






def main():
	...
	
if __name__ == '__main__':
	main()
