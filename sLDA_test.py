#%%
import tomotopy as tp
import os
import os.path
import utils
import prepocessing as pp
import text_processing as _tp
import random
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
#%%
"""

Initialise sLDA model, tw: term weighting scheme k: the number of topics vars: response value type, l means linear

"""
k_topic = 5
mdl = tp.SLDAModel(tw=tp.TermWeight.IDF,k=k_topic,vars='l')
#%%
"""

Loading local dataset

"""
docs_list = []

for i in range(len(pp.topic_label_map)):
        topic_path = os.path.join(pp.LOCAL_DIR, pp.topic_label_map[i][0])
        for doc_name in os.listdir(topic_path):
            doc_path = os.path.join(topic_path, doc_name)
            if os.path.isfile(doc_path):
                lines_set = utils.load_local_dataset(doc_path)
                doc_token_list = []
                for line in lines_set:
                    doc_token_list.extend(_tp.get_list_filtered_tokens(line))
            docs_list.append((doc_token_list,pp.topic_label_map[i][1]))
#%%
"""
Splitting into train set and test set
"""
random.shuffle(docs_list)

ratio = int(len(docs_list)*0.9)
train_docs_list = docs_list[:ratio]
test_docs_list = docs_list[ratio+1:len(docs_list)]
#%%
"""
sLDA model training
"""
for doc in train_docs_list:
    mdl.add_doc(doc[0],[doc[1]])
for i in range(0, len(mdl.docs), k_topic):
    mdl.train(5)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))
#%%
"""
Summary of sLDA topic modelling
"""

for k in range(mdl.k):
    print('Top 5 words of topic #{}'.format(k))
    print(mdl.get_topic_words(k, top_n=k_topic))

mdl.summary()
#%%
"""
Get topic distribution of each document
"""
train_dist = []
for doc in mdl.docs:
    train_dist.append((doc.get_topic_dist(),doc.vars[0]))

test_dist = []

for doc in test_docs_list:
  mdl_doc = mdl.make_doc(doc[0],[doc[1]])
  test_dist.append(
      ((mdl.infer(mdl_doc)),doc[1]))

# %%
"""
Vectorising topic distributions
"""
train_dist_X = []
train_dist_y = []
test_dist_X =[]
test_dist_y=[]

for t in train_dist:
    train_dist_X.append(t[0])
    train_dist_y.append(t[1])

for t in test_dist:
    test_dist_X.append(t[0][0])
    test_dist_y.append(t[1])
# %%
"""
Classifier training using linear SVC
"""
svm_clf = Pipeline([
                ("std_scaler", StandardScaler()),
                ("svm_clf",SVC(kernel='linear',gamma='auto'))])
svm_clf.fit(train_dist_X,train_dist_y)

#%%
y_pred = svm_clf.predict(test_dist_X)
print(classification_report(test_dist_y,y_pred))
# %%

"""
Predicting on up-to-date documents
"""
X_utd_docs = []
y = []
X_utd_vecs = []
utd_path = './unseen_dat/'
for doc_name in os.listdir(utd_path):
                doc_path = os.path.join(utd_path, doc_name)
                if os.path.isfile(doc_path):
                    lines_set = utils.load_local_dataset(doc_path)
                    token_list = []
                    for line in lines_set:
                        token_list.extend(_tp.get_list_filtered_tokens(line))
                    response = int(doc_name.split('_')[0])
                    X_utd_vecs.append(mdl.infer(
                        mdl.make_doc(token_list,[response])
                        )[0])
                    y.append(response)

#%%
y_pred_utd = svm_clf.predict(X_utd_vecs)
print(classification_report(y,y_pred_utd))
# %%
