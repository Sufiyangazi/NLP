# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import nltk
import re
import string
from nltk.stem  import WordNetLemmatizer
import pandas as pd
from nltk.stem.porter import PorterStemmer
# %%
file_path = 'C:\\Users\\Admin\\Data Science\\NLP\\NLP Tutorials\\IMDB Dataset.csv'
data = pd.read_csv(file_path)
# %%
data
# %%
data.head()
# %%
data.values[0]
# %%
print(data.value_counts('sentiment'))
data.value_counts('sentiment').plot(kind='bar')
# %%
### Preprocessing Function
ps = PorterStemmer()
corpus = set()
def preprocess(text):
    
    ## removing unwanted space
    text = text.strip()
    
    ## removing html tags 
    text = re.sub("<[^>]*>", "",text)
    
    ## removing any numerical values
    text = re.sub('[^a-zA-Z]', ' ',text)
    
    ## lower case the word
    text = text.lower()
    
    text = text.split()
    
    ## stemming the word for sentiment analysis do not remove the stop word
    text = [ps.stem(word) for word in text]
    text = ' '.join(text)
    return text
# %%
data['Preprocessed_review'] = data.review.apply(preprocess)
# %%
data.head()
# %%
data.shape
# %%
map_dict = {'positive':1,
           'negative':0}
data['sentiment_numeric'] = data.sentiment.map(map_dict)
data.head()
# %%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data.Preprocessed_review,
                                                 data.sentiment_numeric,
                                                 test_size=0.2,
                                                 random_state=1234,
                                                 stratify=data.sentiment_numeric)
# %%
X_train.shape,X_test.shape
# %%
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer()
tf_idf
# %%
len(tf_idf.vocabulary_)
# %%
