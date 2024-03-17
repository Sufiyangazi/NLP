# %%
import nltk
# %%
dir(nltk)
# %%
# Open the file
filename = 'C:\\Users\\Admin\\Downloads\\metamorphosis clean.txt'
with open(filename,'rt') as file:
    text = file.read()
# %%
text
# %%
print(text)
# %%
# Split the text using white space
words = text.split()
# %%
words
# %%
# Punctuations
import string
string.punctuation
# %%
len(string.punctuation)
# %%
string.printable
# %%
len(string.printable)
# %%
string.digits
# %%
string.ascii_letters
# %%
string.ascii_lowercase
# %%
string.ascii_uppercase
# %%
string.capwords('nltk')
# %%
# re
# regex pacakge
import re
# %%
text = 'The muslims are very smart'
text.count('m')
# %%
text = 'The muslims are very smart'
re.findall('ar',text)
# %%
dir(re)
# %%
# Remove punctuations
text1 = 'Mehmad the conqurer'
words = text1.split()
words
# %%
re.escape(string.punctuation)
# %%
re_punct = re.compile('[%s]'%re.escape(string.punctuation))
# %%
for w in words:
    print(w)
    print(re_punct.sub('',w))
# %%
re_punct
# %%
import string,re
re_punct = re.compile('[%s]'%re.escape(string.punctuation))
w = [re_punct.sub('',w) for w in words]
w
# %%
import re
text1 = 'either i conquer you or you will conqurer me$$'
words = text1.split()
print(words)
re_punct = re.compile('[%s]'%re.escape(string.punctuation))
w = [re_punct.sub('',w) for w in words]
print(w)
# %%
import re
text1 = 'I will the one to kill ##you$$'
words = text1.split()
print(words)
re_punct = re.compile('[%s]'%re.escape(string.punctuation))
w = [re_punct.sub('',w) for w in words]
print(w)
# %%
def re_exp(text):
    print('orginal text',text)
    words = text.split()
    print(words)
    re_punct = re.compile('[%s]'%re.escape(string.punctuation))
    w = [re_punct.sub('',w) for w in words]
    sent = ' '.join(w)
    print(w)
    print('preprocessed text',sent)
re_exp('A new age is going to start it is starting $$$is $$ starting tonight..#####we #$will %take the ci$ty')
# %%
def text_cleaning(text1):
    words = text1.split()
    re_punct = re.compile('[%s]'%re.escape(string.punctuation))
    w = [re_punct.sub('',w) for w in words]
    return (w)
# %%
file_name = 'C:\\Users\\Admin\\Downloads\\metamorphosis clean.txt'
file = open(file_name,'rt')
text = file.read()
w =text_cleaning(text)
w
# %%
# Normalization
# convert to lowercase
def text_cleaning(text1):
    words = text1.split()
    re_punct = re.compile('[%s]'%re.escape(string.punctuation))
    w = [re_punct.sub('',w) for w in words]
    w = [i.lower() for i in w]
    return (w)
# %%
# Filter and Punctuations
from nltk.tokenize import word_tokenize,sent_tokenize
file_name = 'C:\\Users\\Admin\\Downloads\\metamorphosis clean.txt'
file = open(file_name,'rt',encoding='utf-8-sig')
text = file.read()
tokens = word_tokenize(text)
print(tokens[:15])
# %%
print(text)
# %%
from nltk.tokenize import word_tokenize,sent_tokenize
file_name = 'C:\\Users\\Admin\\Downloads\\metamorphosis clean.txt'
file = open(file_name,'rt',encoding='utf-8-sig')
text = file.read()
tokens = word_tokenize(text)
print(tokens[:15])
words = [words for word in tokens if word.isalnum()]
words[:10]
# %%
from nltk.tokenize import word_tokenize,sent_tokenize
files = 'C:\\Users\\Admin\\Downloads\\metamorphosis clean.txt'
file = open(files,mode='rt',encoding='utf-8-sig')
text = file.read()
token = word_tokenize(text)
print(token[:5])
wordss = [word for word in token if word .isalnum()]
wordss[:10]
# %%
from nltk.tokenize import word_tokenize,sent_tokenize
file_loc = 'C:\\Users\\Admin\\Downloads\\metamorphosis clean.txt'
file = open(file_loc,mode='rt',encoding='utf-8-sig')
text = file.read()
tokenize = word_tokenize(text)
print(tokenize[:10])
# re
repunct = re.compile('[%s]'%re.escape(string.punctuation))
w = [repunct.sub('',w) for w in tokenize]
print(w[:4])




# %%
from nltk.tokenize import sent_tokenize
filename = 'C:\\Users\\Admin\\Downloads\\metamorphosis clean.txt'
file = open(filename,mode='rt',encoding='utf-8-sig')
text = file.read()
sent_tokens = sent_tokenize(text)
sent_tokens[:10]
# re
repunct = re.compile('[%s]'%re.escape(string.punctuation))
t = (repunct.sub('',t) for t in sent_tokens)
modifed_sent = list(t)
print(modifed_sent[:10])
# %%
# removing stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
filename = 'C:\\Users\\Admin\\Downloads\\metamorphosis clean.txt'
file = open(filename,mode='rt',encoding='utf-8-sig')
text = file.read()
# split into words
tokens = word_tokenize(text)
# Remove all tokens that are not alphabetic
words = [word for word in token if word.isalnum()]
# Remove stopwords
stop_words = stopwords.words('english')
words = [w for w in words if not w in stop_words]
words[:10]
# %%
# Stem words
# - reducing of each word to rot or base
# - ex:fishing,fished,fisher ===== fish
# - in NLP making a dictonary or a vocablary based on your case will create our own vocablary
# - we create our dictionary to do stemming in NLTK =======> Porter Stemmer
import string
import re
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Step-1 : Load the data

file_name = 'C:\\Users\\Admin\\Downloads\\metamorphosis clean.txt'
file = open(file_name,mode='rt',encoding='utf-8-sig')
text = file.read()

# Step -2 : Split into tokens
tokens = word_tokenize(text)

# step-3 : convert into lower case
tokens = [w.lower() for w in token]

# prepare regax to remove punctuations
repunct = re.compile('[%s]'%re.escape(string.punctuation))

# step - 4 : remove the punctuations
stripped = [repunct.sub('',w) for w in tokens]

# step - 5 : remove all the tokens that are not alphabetic
words = [word for word in stripped if word.isalnum()]

# step - 6 : remove the stopwords
stop_words = stopwords.words('english')
words = [w for w in words if not w in stop_words]
print(words[:10])

# Step - 7 : Apply stemming
porter = PorterStemmer()
stem_words = [porter.stem(word) for word in words]
print(stem_words[:10])
# %%
word1 = ['python','pythoner','pythonly','pythoned']
ps = PorterStemmer()
stem_words=[ps.stem(word) for word in word1]
print(stem_words)
# %%
text='All pythoners should be pythoned will with python,every one atleast pythones very poorly with python'
# tokens = word_tokenize(text)
words1 = [ps.stem(w) for w in word_tokenize(text)]
print(words1)
# %%
# PARTS OF SPEECH
# - for each word it will provide the POS
from nltk import pos_tag
text = 'He recived the best actor award'
pos_tag(word_tokenize(text)) # first we apply word_tokenize then apply parsts of speech
# it will give if the word in noun,pronoun,verb,adverb,adjective
# %%
nltk.help.upenn_tagset('PRP')
# %%
nltk.help.upenn_tagset()
# %%
# Lemmatistaion
# - Lemmatisation is similar to stemming , as it produces a normalised version of the input word

# - The output is lemma i.e.Proper word

# - The input word is lemmataised according to its Part-Of-Speech(POS) tag
# %%
from nltk.stem import WordNetLemmatizer
s=WordNetLemmatizer()
print(s.lemmatize('having',pos='v'))
print(s.lemmatize('have',pos='v'))
print(s.lemmatize('had',pos='v'))

print(s.lemmatize('fishing',pos='v'))
print(s.lemmatize('fish',pos='v'))
print(s.lemmatize('fisher',pos='v'))
print(s.lemmatize('fishes',pos='v'))
print(s.lemmatize('fished',pos='v'))

# %%
from nltk.stem import PorterStemmer
s=PorterStemmer()
print(s.stem("having"))
print(s.stem("have"))
print(s.stem("had"))
# %%
from nltk.corpus import wordnet
syns1 = wordnet.synsets('good')
syns1
# %%
word1=wordnet.synset('good.n.01')
print(word1.definition())
print(word1.examples())
# %%
word1=wordnet.synset('good.n.02')
print(word1.definition())
print(word1.examples())
# %%
word1=wordnet.synset('good.s.06')
print(word1.definition())
print(word1.examples())
# %%
