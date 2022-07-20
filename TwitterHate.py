#!/usr/bin/env python
# coding: utf-8

# # Help Twitter Combat Hate Speech Using NLP and Machine Learning.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


# ## 1. Load the tweets file using read_csv function from Pandas package

# In[2]:


df = pd.read_csv('TwitterHate.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df['label'].value_counts()


# In[6]:


sns.countplot(df['label'])


# In[7]:


df[df['label'] == 1].values[:10]


# In[8]:


df['label'].value_counts(normalize=True)


# 0 is positive tweet and 1 is negative tweet. There is a class imbalance problem here. 93 % of the tweets are non hate and 7% is hate.

# ## 2. Get the tweets into a list for easy text cleanup and manipulation.

# In[9]:



tweets = df['tweet'].values
tweets[:5]


# ## 3.To cleanup: 

# ### 1.Normalize the casing.

# In[10]:


tweets = [tweet.lower() for tweet in tweets]
tweets[:5]


# ### 2. Using regular expressions, remove user handles. These begin with '@’.

# In[11]:


pattern = re.compile(re.escape('@'))


# In[12]:


tweets = [pattern.sub('',tweet) for tweet in tweets]


# In[13]:


tweets[:10]


# ### 3.Using regular expressions, remove URLs.

# In[14]:


pattern = re.compile(r'((ftp|http|https):\/\/)?www\.([A-z]+)\.([A-z]{2,})')


# In[15]:


#pattern = re.compile('user')


# In[16]:


pattern


# In[17]:


#tweets = [pattern.sub('',tweet) for tweet in tweets]


# In[18]:


matches = []
for tweet in tweets:
  matches.append(pattern.finditer(tweet))


# In[19]:


for match in matches:
   for m in match:
    print(m)


# In[20]:


tweets = [pattern.sub('',tweet) for tweet in tweets]


# In[21]:


matches = []
for tweet in tweets:
  matches.append(pattern.finditer(tweet))


# In[22]:


for match in matches:
   for m in match:
    print(m)


# ### 4. Using TweetTokenizer from NLTK, tokenize the tweets into individual terms.

# In[23]:


tokenizer = TweetTokenizer()
tokens = []
for tweet in tweets:
    tokens.append(tokenizer.tokenize(tweet))


# In[24]:


tokens[0]


# ### 5. Remove stop words.

# In[25]:


# filter out stop words
stop_words = set(stopwords.words('english'))
stopword_removed = []
for sent in tokens:    
    stopword_removed.append([word for word in sent if not word in stop_words])


# In[26]:


stopword_removed[0]


# ### 6.  Remove redundant terms like ‘amp’, ‘rt’, etc.

# In[27]:


pattern = re.compile(r'&')


# In[28]:


matches = []
for sent in stopword_removed:
    for word in sent:
        matches.append(pattern.finditer(word))


# In[29]:


for match in matches:
    for m in match:
        print(m)


# In[30]:


amp_removed = []
for sent in stopword_removed:
    amp_removed.append([pattern.sub("",word) for word in sent])


# In[31]:


amp_removed


# In[32]:


# Re-check if the '&" is removed.
matches = []
for sent in amp_removed:
    for word in sent:
        matches.append(pattern.finditer(word))


# In[33]:


for match in matches:
    for m in match:
        print(m)


# The match returns no results, which means '&' is removed.

# ### 7. Remove ‘#’ symbols from the tweet while retaining the term.

# In[34]:


amp_removed


# In[35]:


pattern = re.compile('#')


# In[36]:


matches = []
for sent in amp_removed:
    for word in sent:
        matches.append(pattern.finditer(word))
    


# In[37]:


for match in matches:
    for m in match:
        print(m)


# In[38]:


hashtag_removed = []
for sent in amp_removed:
    hashtag_removed.append([pattern.sub("",word) for word in sent])


# In[39]:


hashtag_removed


# In[40]:


# Re-check if '#' is removed.
matches = []
for sent in hashtag_removed:
    for word in sent:
        matches.append(pattern.finditer(word))


# In[41]:


for match in matches:
    for m in match:
        print(m)


# The match returns no results, which means '#' is removed.

# ## 4. Extra cleanup by removing terms with a length of 1.

# In[42]:


smallword_removed = []
for sent in hashtag_removed:
    smallword_removed.append([word for word in sent if len(word) > 1])


# In[43]:


smallword_removed


# In[44]:


# Also remove any words that are not alpha
only_alpha = []
for sent in smallword_removed:
    only_alpha.append([word for word in sent if word.isalpha()])


# In[45]:


only_alpha


# ## 5.Check out the top terms in the tweets:

# ### 1. First, get all the tokenized terms into one large list.

# In[46]:


only_alpha


# In[47]:


allwords = []
for sent in only_alpha:
    for word in sent:
        allwords.append(word)


# In[48]:


allwords


# ### 2.Use the counter and find the 10 most common terms.

# In[49]:


top = Counter(allwords)
temp = pd.DataFrame(top.most_common(10))
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='Blues')


# # 6. Data formatting for predictive modeling:

# ### 1. Join the tokens back to form strings. This will be required for the vectorizers.

# In[50]:


only_alpha


# In[51]:


df['tweet'] = only_alpha


# In[52]:


df


# ### 2. Assign x and y.

# In[53]:


X = df['tweet']
y = df['label']


# In[54]:


print(X)
print(y)


# ### 3. Perform train_test_split using sklearn.
# 
#  

# In[55]:


# Split the data into training and testing data - 90% train and 10% test

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.1, random_state = 100)


# In[56]:


print(X_train[:5])
print(X_test[:5])
print(y_train[:5])
print(y_test[:5])


# In[57]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[58]:


y_train.value_counts(normalize=True)


# In[59]:


y_test.value_counts(normalize=True)


# ## 7. We’ll use TF-IDF values for the terms as a feature to get into a vector space model.

# ### 1. Import TF-IDF  vectorizer from sklearn

# In[60]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[61]:


X_train


# In[62]:


X_train = X_train.map(' '.join)


# In[63]:


X_test = X_test.map(' '.join)


# ### 2. Instantiate with a maximum of 5000 terms in your vocabulary. 

# In[64]:


X_train


# In[65]:


X_test


# ### 3.Fit and apply on the train set.

# In[66]:


vectorizer = TfidfVectorizer(max_features=5000)

# tokenize and build vocab
vectorizer.fit(X_train)
# encode document
training_features = vectorizer.transform(X_train)


# ### 4. Apply on test set 

# In[67]:


test_features = vectorizer.transform(X_test)


# In[68]:


training_features.toarray()


# In[69]:


test_features.toarray()


# ## 8. Model building: Ordinary Logistic Regression

# ### 1. Instantiate Logistic Regression from sklearn with default parameters.

# In[70]:


logReg = LogisticRegression()


# ### 2. Fit into  the train data. 

# In[71]:


logReg.fit(training_features,y_train)


# ### 3. Make predictions for the train and the test set. 

# In[72]:


# predict the y based on X_test.
prediction = logReg.predict(test_features)


# In[73]:


prediction


# ## 9. Model evaluation: Accuracy, recall, and f_1 score. 

# ### 1. Report the accuracy on the test set. 

# In[74]:


# Accuracy score for the model
from sklearn import metrics
print(metrics.accuracy_score(y_test,prediction))


# In[75]:


# Confusion matrix for the model
cm = metrics.confusion_matrix(y_test, prediction)


# In[76]:


cm


# ### 2. Report the recall and f1 score 

# In[77]:


print("\n" + classification_report(y_test,prediction))


# The F1 score is 0.98 for class 0 (majority) and 0.44 for class 1 (minority).

# ## 10. Looks like you need to adjust the class imbalance, as the model seems to focus on the 0s.

# ### 1. Adjust the appropriate class in the LogisticRegression model.

# Our majority class(0) value count is 0.929854 and minority class(1) value count is 0.070146. 
# to correct the class imbalance, We set a higher weight for minority class and reduce the weight for the majority class.
# Here we can set weights such that minority class is 13 times more than majority class.
# 

# ## 11. Train again with the adjustment and evaluate.

# In[78]:


logReg = LogisticRegression(class_weight={0:0.070146 , 1: 0.929854})
logReg.fit(training_features,y_train)
# predict the y based on X_test.
prediction = logReg.predict(test_features)


# In[79]:


# Confusion matrix for the model
cm = metrics.confusion_matrix(y_test, prediction)
cm


# In[80]:


print("\n" + classification_report(y_test,prediction))


# The F1 score is 0.95 for class 0 (majority) and 0.50 for class 1 (minority).

# ## 12. Use a balanced class weight while instantiating the logistic regression.

# In[81]:


logReg = LogisticRegression(class_weight='balanced')
logReg.fit(training_features,y_train)
# predict the y based on X_test.
prediction = logReg.predict(test_features)


# In[82]:


# Confusion matrix for the model
cm = metrics.confusion_matrix(y_test, prediction)
cm


# In[83]:


print("\n" + classification_report(y_test,prediction))


# The F1 score is 0.96 for class 0 (majority) and 0.54 for class 1 (minority).

# ## 13. Regularization and Hyperparameter tuning: 

# ### 1. Import GridSearch and StratifiedKFold because of class imbalance. 

# In[84]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold
lr = LogisticRegression()

#Setting the range for class weights
weights = np.linspace(0.0,0.99,200)

#Creating a dictionary grid for grid search
param_grid = {'class_weight': [{0:x, 1:1.0-x} for x in weights]}

#Fitting grid search to the train data with 5 folds
gridsearch = GridSearchCV(estimator= lr, 
                          param_grid= param_grid,
                          cv=StratifiedKFold(), 
                          n_jobs=-1, 
                          scoring='f1', 
                          verbose=2).fit(training_features, y_train)

#Ploting the score for different values of weight
sns.set_style('whitegrid')
plt.figure(figsize=(12,8))
weigh_data = pd.DataFrame({ 'score': gridsearch.cv_results_['mean_test_score'], 'weight': (1- weights)})
sns.lineplot(weigh_data['weight'], weigh_data['score'])
plt.xlabel('Weight for class 1')
plt.ylabel('F1 score')
plt.xticks([round(i/10,1) for i in range(0,11,1)])
plt.title('Scoring for different class weights', fontsize=24)


# In[85]:


weigh_data.sort_values(by="score",ascending=False)


# using gridsearch and stratified kfold cv, we found the optimal weights which give the highest f1 score. 
# Weight for class 1(minority class) is 0.880603. Weight for class 0(majority class) is 1-0.880603 which is 0.119397.
# Lets apply these weights and run the code once more.

# In[86]:


logReg = LogisticRegression(class_weight={0:0.119397,1:0.880603})
logReg.fit(training_features,y_train)
# predict the y based on X_test.
prediction = logReg.predict(test_features)


# In[87]:


# Confusion matrix for the model
cm = metrics.confusion_matrix(y_test, prediction)
cm


# In[88]:


print("\n" + classification_report(y_test,prediction))


# The F1 score is 0.97 for class 0 (majority) and 0.61 for class 1 (minority). We tried to create a balance by getting decent f1 scores of both class 0 and class 1 using the gridsearch and stratified k-fold.

# In[89]:


X_test


# In[90]:


prediction.shape


# In[92]:


df['prediction'] = prediction


# In[93]:


df.sort_values(by="prediction",ascending=False)[:10]


# We can see that the model has indeed identified the racist and hate tweets correctly.

# ## Conclusion

# In this project, we worked on classfying the twitter tweets as non-hate and hate tweets. 
# We did text processing by removing &,#,stop words,non alpha etc and we converted the tweets to tokens.
# Also, we did feature engineering by converting the words to vectors using tf-idf. 
# Next we assigned X and y and split our data. We used logistic regression to train our model. We dealt with class imbalance by adjusting weights for the majority and minority classes. We tried using balanced weights to improve the f1 score. 
# Lastly, we used gridsearch cv and stratified k-fold to find the optimal weights for the 0 and 1 classes.
# We were able to build a decent model with a f1 score of 97 for majority class and 61 for minority class.
