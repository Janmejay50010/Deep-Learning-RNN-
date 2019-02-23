#!/usr/bin/env python
# coding: utf-8

# In[2]:


import string
import re
from numpy import array, argmax, random, take
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[29]:


def read_text(filename):
        # open the file
        file = open(filename, mode='rt')
        
        # read all text
        text = file.read()
        file.close()
        return text


# In[40]:


exstring = 'abc\ndefgh\nijk'
exlist = exstring.split('\n')
print(exlist)
#for i in exstring:
    #print(i)


# In[45]:


exstr = [['abc','def'],['ghi','jkl']]
Dict = {}
for i in exstr:
    Dict[i[0]]=i[1]
Dict


# In[ ]:


# split a text into sentences
def to_lines(text):
        Dict = {}
        sents = text.split('\n')
        sents1 = [i.split('\t') for i in sents]
        for j in sents1:
            print(j[1])
            #Dict[j[0]] = j[1]
        
        return Dict


# In[46]:


# split a text into sentences
def to_lines(text):
        Dict = {}
        sents = text.split('\n')
        sents1 = [i.split('\t') for i in sents]
        for j in sents1:
            print(j[1])
            #Dict[j[0]] = j[1]
        
        return Dict


# In[ ]:





# In[47]:


data = read_text('spa.txt')


# In[48]:


#print(data)
print(type(data))


# In[49]:


spa_eng = to_lines(data)


# In[14]:


import numpy as np
##
x = [[1,2],[2,3],[3,4]]
y = np.array(x)
y.shape


# In[8]:


print(type(spa_eng))
print(len(spa_eng))
spa_eng[:9]


# In[12]:


import numpy as np
#print(spa_eng[:4])
spa_eng1=np.array(spa_eng)
#print(type(np.array(spa_eng1[0]))
print(type(spa_eng1[0]))
spa_eng10=np.array(spa_eng1[0])
print(spa_eng10.shape)


# In[19]:


spa_eng1


# In[10]:


#Only the first 50000 sentence pairs to reduce the training time of the model
spa_eng = spa_eng1[:50000,:]


# In[13]:


# Remove punctuation
#spa_eng[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in spa_eng1[:,0]]
#spa_eng[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in spa_eng1[:,1]]


# In[ ]:


# convert text to lowercase
for i in range(len(deu_eng)):
    deu_eng[i,0] = deu_eng[i,0].lower()
    deu_eng[i,1] = deu_eng[i,1].lower()

deu_eng


# In[ ]:


# empty lists
eng_l = []
deu_l = []

# populate the lists with sentence lengths
for i in deu_eng[:,0]:
      eng_l.append(len(i.split()))

for i in deu_eng[:,1]:
      deu_l.append(len(i.split()))

length_df = pd.DataFrame({'eng':eng_l, 'deu':deu_l})

length_df.hist(bins = 30)
plt.show()


# In[ ]:


# function to build a tokenizer
def tokenization(lines):
      tokenizer = Tokenizer()
      tokenizer.fit_on_texts(lines)
      return tokenizer


# In[ ]:


# prepare english tokenizer
eng_tokenizer = tokenization(deu_eng[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1

eng_length = 8
print('English Vocabulary Size: %d' % eng_vocab_size)


# In[ ]:


# prepare Deutch tokenizer
deu_tokenizer = tokenization(deu_eng[:, 1])
deu_vocab_size = len(deu_tokenizer.word_index) + 1

deu_length = 8
print('Deutch Vocabulary Size: %d' % deu_vocab_size


# In[ ]:


# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
         # integer encode sequences
         seq = tokenizer.texts_to_sequences(lines)
         # pad sequences with 0 values
         seq = pad_sequences(seq, maxlen=length, padding='post')
         return seq


# In[ ]:


from sklearn.model_selection import train_test_split

# split data into train and test set
train, test = train_test_split(deu_eng, test_size=0.2, random_state = 12)


# In[ ]:


# prepare training data
trainX = encode_sequences(deu_tokenizer, deu_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])

# prepare validation data
testX = encode_sequences(deu_tokenizer, deu_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])


# In[ ]:


# build NMT model
def define_model(in_vocab,out_vocab, in_timesteps,out_timesteps,units):
      model = Sequential()
      model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
      model.add(LSTM(units))
      model.add(RepeatVector(out_timesteps))
      model.add(LSTM(units, return_sequences=True))
      model.add(Dense(out_vocab, activation='softmax'))
      return model


# In[ ]:


# model compilation
model = define_model(deu_vocab_size, eng_vocab_size, deu_length, eng_length, 512)


# In[ ]:


rms = optimizers.RMSprop(lr=0.001)
model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')


# In[ ]:


filename = 'model.h1.24_jan_19'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# train model
history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1),
                    epochs=30, batch_size=512, validation_split = 0.2,callbacks=[checkpoint], 
                    verbose=1)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','validation'])
plt.show()


# In[ ]:


model = load_model('model.h1.24_jan_19')
preds = model.predict_classes(testX.reshape((testX.shape[0],testX.shape[1])))


# In[ ]:


def get_word(n, tokenizer):
      for word, index in tokenizer.word_index.items():
          if index == n:
              return word
      return None


# In[ ]:


preds_text = []
for i in preds:
       temp = []
       for j in range(len(i)):
            t = get_word(i[j], eng_tokenizer)
            if j > 0:
                if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):
                     temp.append('')
                else:
                     temp.append(t)
            else:
                   if(t == None):
                          temp.append('')
                   else:
                          temp.append(t) 

       preds_text.append(' '.join(temp))


# In[ ]:


pred_df = pd.DataFrame({'actual' : test[:,0], 'predicted' : preds_text})


# In[ ]:


# print 15 rows randomly
pred_df.sample(15)


# In[ ]:





# In[ ]:




