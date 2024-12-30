#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
# import gensim

from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

nltk.download('punkt') # biaodian dataset
nltk.download('stopwords') # meaningless words dataset


# In[2]:


# Load data into dataframe

df = pd.read_csv(r"C:\Users\Ravi Chauhan\Downloads\amazon_reviews_us_Watches_v1_00.tsv", sep='\t', error_bad_lines=False)


# In[3]:


df.head(5)


# In[4]:


# Remove missing value
df.dropna(subset=['review_body'],inplace=True)


# In[5]:


df.reset_index(inplace=True, drop=True)


# In[6]:


df.info()


# In[7]:


# use the first 1000 data as our training data
data = df.loc[:999, 'review_body'].tolist() # focus on first 1000 reviews


# # Part 2: Tokenizing and Stemming

# In[8]:


# Use nltk's English stopwords.
stopwords = nltk.corpus.stopwords.words('english') #stopwords.append("n't")
stopwords.append("'s")
stopwords.append("'m")
stopwords.append("br") #html <br>
stopwords.append("watch") # all comments this time will be related to watches, so it will exist and will be meaningless

print ("We use " + str(len(stopwords)) + " stop-words from nltk library.")
print (stopwords[:10])


# In[35]:


from nltk.stem.snowball import SnowballStemmer
# from nltk.stem import WordNetLemmatizer 

stemmer = SnowballStemmer("english") # stemming process, convert words with modifications to original words

# tokenization and stemming
def tokenization_and_stemming(text):
    tokens = []
    # exclude stop words and tokenize the document, generate a list of string 
    for word in nltk.word_tokenize(text):
        if word.lower() not in stopwords:
            tokens.append(word.lower())

    filtered_tokens = []
    
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if token.isalpha(): # filter out non alphabet words like emoji
            filtered_tokens.append(token)
            
    # stemming
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# In[36]:


tokenization_and_stemming(data[0])


# In[37]:


data[0]


# # Part 3: TF-IDF

# In[38]:


from sklearn.feature_extraction.text import TfidfVectorizer
# define vectorizer parameters
# TfidfVectorizer will help us to create tf-idf matrix
# max_df : maximum document frequency for the given word
# min_df : minimum document frequency for the given word
# max_features: maximum number of words
# use_idf: if not true, we only calculate tf
# stop_words : built-in stop words
# tokenizer: how to tokenize the document
# ngram_range: (min_value, max_value), eg. (1, 3) means the result will include 1-gram, 2-gram, 3-gram
tfidf_model = TfidfVectorizer(max_df=0.99, max_features=1000,
                                 min_df=0.01, stop_words=stopwords,#stop_words='english',
                                 use_idf=True, tokenizer=tokenization_and_stemming, ngram_range=(1,1))

tfidf_matrix = tfidf_model.fit_transform(data) #fit the vectorizer to synopses

print ("In total, there are " + str(tfidf_matrix.shape[0]) +       " reviews and " + str(tfidf_matrix.shape[1]) + " terms.")


# In[39]:


tfidf_matrix


# In[40]:


tfidf_matrix.toarray() #todense()


# In[41]:


tfidf_matrix.todense()


# In[42]:


print(type(tfidf_matrix.toarray()))


# In[43]:


print(type(tfidf_matrix.todense()))


# In[44]:


# words
tf_selected_words = tfidf_model.get_feature_names()


# In[45]:


# print out words
tf_selected_words[0:100:5]


# # Part 4: K-means clustering

# In[46]:


# k-means clustering
from sklearn.cluster import KMeans

num_clusters = 5

# number of clusters
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)

clusters = km.labels_.tolist()


# # 4.1. Analyze K-means Result
#  

# In[47]:


# create DataFrame films from all of the input files.
product = { 'review': df[:1000].review_body, 'cluster': clusters}
frame = pd.DataFrame(product, columns = ['review', 'cluster'])


# In[48]:


frame.head(10)


# In[49]:


print ("Number of reviews included in each cluster:")
frame['cluster'].value_counts().to_frame()


# In[50]:


km.cluster_centers_


# In[51]:


km.cluster_centers_.shape


# # Document clustering results

# In[52]:


print ("<Document clustering result by K-means>")

#km.cluster_centers_ denotes the importances of each items in centroid.
#We need to sort it in decreasing-order and get the top k items.
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

Cluster_keywords_summary = {}
for i in range(num_clusters):
    print ("Cluster " + str(i) + " words:", end='')
    Cluster_keywords_summary[i] = []
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        Cluster_keywords_summary[i].append(tf_selected_words[ind])
        print (tf_selected_words[ind] + ",", end='')
    print ()
    
    cluster_reviews = frame[frame.cluster==i].review.tolist()
    print ("Cluster " + str(i) + " reviews (" + str(len(cluster_reviews)) + " reviews): ")
    print ("Example1: "+cluster_reviews[0])
    print ("Example2: "+cluster_reviews[1])
    print ("Example3: "+cluster_reviews[2])
    print ()


# # Part 5: Topic Modeling - Latent Dirichlet Allocation

# In[53]:


# Use LDA for clustering
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=5)


# In[54]:


# document topic matrix for tfidf_matrix_lda
lda_output = lda.fit_transform(tfidf_matrix)
print(lda_output.shape)
print(lda_output)


# In[55]:


# topics and words matrix
topic_word = lda.components_
print(topic_word.shape)
print(topic_word)


# In[56]:


# column names
topic_names = ["Topic" + str(i) for i in range(lda.n_components)]

# index names
doc_names = ["Doc" + str(i) for i in range(len(data))]

df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topic_names, index=doc_names)

# get dominant topic for each document
topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['topic'] = topic

df_document_topic.head(10)


# In[57]:


df_document_topic['topic'].value_counts().to_frame()


# In[58]:


# topic word matrix
print(lda.components_)
# topic-word matrix
df_topic_words = pd.DataFrame(lda.components_)

# column and index
df_topic_words.columns = tfidf_model.get_feature_names()
df_topic_words.index = topic_names

df_topic_words.head()


# In[59]:


# print top n keywords for each topic
def print_topic_words(tfidf_model, lda_model, n_words):
    words = np.array(tfidf_model.get_feature_names())
    topic_words = []
    # for each topic, we have words weight
    for topic_words_weights in lda_model.components_:
        top_words = topic_words_weights.argsort()[::-1][:n_words]
        topic_words.append(words.take(top_words))
    return topic_words

topic_keywords = print_topic_words(tfidf_model=tfidf_model, lda_model=lda, n_words=15)        

df_topic_words = pd.DataFrame(topic_keywords)
df_topic_words.columns = ['Word '+str(i) for i in range(df_topic_words.shape[1])]
df_topic_words.index = ['Topic '+str(i) for i in range(df_topic_words.shape[0])]
df_topic_words


# In[ ]:




