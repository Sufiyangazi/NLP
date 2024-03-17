# %%
from nltk.corpus import stopwords
sentences = ['sky is nice','cloud is nice','sky is nice and cloud is nice']
cleaned_sentence = []
for sent in sentences:
    word = sent.lower() # lower all sent so it dosnot treat upper case and lowercase differently
    worwd = word.split()
    # remove the stop words
    words = [w for w in word if w not in set(stopwords.words('english'))]
    word = "".join(word)
    cleaned_sentence.append(word)

print(cleaned_sentence)
# %%
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=3)
bagofwords=cv.fit_transform(cleaned_sentence)

bagofwords.toarray()
# %%
import pandas as pd
pd.DataFrame(bagofwords.toarray(),columns=['cloud','nice','sky'])
# %%
cv.vocabulary_
# %%
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
sentences = ['Game of Thrones is an amazing tv series!', 
             'Game of Thrones is the best tv series!', 
             'Game of Thrones is so great']
cleaned_sentence = []
for sent in sentences:
    word = sent.lower()
    word = sent.split()
    word = [w for w in word if w not in set(stopwords.words('english'))]
    word = "".join(word)
    cleaned_sentence.append(word)
print(cleaned_sentence)

# feature extracrion
cv = CountVectorizer()
bagofwords = cv.fit_transform(cleaned_sentence).toarray()
print(cv.vocabulary_)
print(bagofwords)
# %%
################################ With logic##################################
sen=' '.join(cleaned_sentence)
l=list(set(sen.split()))
print("vocabulary:",l)
d={}
l1=[]
for sentence in cleaned_sentence:
    for i in l:
        if i in sentence:
            d[i]=1
        else:
            d[i]=0
    myKeys = list(d.keys())
    myKeys.sort()
    sorted_dict = {i: d[i] for i in myKeys}
    l1.append(sorted_dict)

print(l1)
l2=[i.values() for i in l1]
l2
# %%
# Creating word histogram
import nltk
word2count = {}
for data in sentences:
    words = nltk.word_tokenize(data) # we are split into words
    for word in words:               # we are calling each word
        if word not in word2count.keys(): # if the word not in dictionary, we are 
            word2count[word] = 1
        else:
            word2count[word] += 1
print(word2count)
# %%
import matplotlib.pyplot as plt
plt.hist(word2count,bins=80)
# %%
