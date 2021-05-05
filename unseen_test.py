import text_processing as tp
import prepocessing as pp
import os
import os.path
import utils
import numpy as np
import fs_n_training as ft
from sklearn.metrics import classification_report


path = './unseen_dat/'

X_unseen_raw = []
y = []

for doc_name in os.listdir(path):
                doc_path = os.path.join(path, doc_name)
                if os.path.isfile(doc_path):
                    lines_set = utils.load_local_dataset(doc_path)
                    X_unseen_raw.append(lines_set)
                    y.append(int(doc_name.split('_')[0]))
title_voc = utils.load('vocab_title.pickle')
voc = utils.load('vocab.pickle')


X_unseen_vecs = []
for i,doc in enumerate(X_unseen_raw):
    title = np.asarray(tp.get_freq_vecs([doc[0]], title_voc)[0])
    content = np.asarray(tp.get_freq_vecs(doc[1:],  voc))
    content_vec =np.zeros(len(content[0]))
    for j in range(len(content)) :
        content_vec  = np.add(content_vec,content[j])
    pos = np.asarray(tp.get_pos_tag_list(doc))
    # vectorised general X
    x = np.concatenate((title,content_vec,pos), axis=0)
    #x = np.concatenate((x,pos),axis=0)
    X_unseen_vecs.append(x)

svm_test,soft_test = ft.get_grid_instance()
y_gold = np.asarray(y)
ft.print_test_report("SVC Best ",pp.topic_list,svm_test,X_unseen_vecs,y_gold)
ft.print_test_report("Softmax Best ",pp.topic_list,soft_test,X_unseen_vecs,np.asarray(y_gold))



 