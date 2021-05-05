# %%
import numpy as np
import os
import os.path
import utils
import text_processing as tp
import collections
import functools
import operator
import random

LOCAL_DIR = './dataset/bbc/'

BUSINESS = 'business'
ENTERTAIN = 'entertainment'
POLITICS = 'politics'
SPORT = 'sport'
TECH = 'tech'

topic_label_list = [0,1,2,3,4]
topic_list = [BUSINESS,ENTERTAIN,POLITICS,SPORT,TECH]

topic_label_map = [(BUSINESS, 0), (ENTERTAIN, 1),
                   (POLITICS, 2), (SPORT, 3), (TECH, 4)]

def processing(flag, c_top_num=1000, t_top_num=300):
    X_SERIAL = f"X_{str(c_top_num)}_{str(t_top_num)}.pickle"
    y_SERIAL = f"y_{str(c_top_num)}_{str(t_top_num)}.pickle"
    if not utils.does_file_exist(X_SERIAL) or not utils.does_file_exist(y_SERIAL) or not flag:
        print("Cached dataset not found, processing ...")

        """
        1. Loading dataset and meta information of documents

        """
        # raw_data -> {'topic',[document[lines[]]]}
        raw_data = {}
        topic_docs_map = []

        # iterating directories
        for i in range(len(topic_label_map)):
            file_count = 0
            topic_path = os.path.join(LOCAL_DIR, topic_label_map[i][0])
            for doc_name in os.listdir(topic_path):
                doc_path = os.path.join(topic_path, doc_name)
                if os.path.isfile(doc_path):
                    file_count += 1
                    lines_set = utils.load_local_dataset(doc_path)
                    label_key = topic_label_map[i][1]
                    if label_key in raw_data:
                        raw_data[label_key].append(lines_set)
                    else:
                        raw_data[label_key] = []
                        raw_data[label_key].append(lines_set)
            # save the number of documents in each topic
            topic_docs_map.append((topic_label_map[i][0], file_count))

        print("count:", topic_docs_map)
        # %%
        """

        2. For headlines and contents of news, computing words frequency in each topic respectivly (topic) 
        merging the frequency dictionaries from all topics and creating the corups freq dictionary

        """
        # catergorised frequency dictionaries appear in news contents
        topic_freq_list = []
        # general frequency dic in news contents
        collection_freq = {}

        # number of token in each topic in news contents
        topic_token_no = []
        # total number of tokens in news contents
        data_token_no = 0

        # catergorised token frequency dictionaries appear in titles
        topic_title_freq_list = []
        # general  frequency dic appear in titles
        collection_title_freq = {}

        for topic in raw_data:
            documents = raw_data[topic]
            topic_content_freq = {}
            topic_title_freq = {}
            for i in range(len(documents)):
                # title
                topic_title_freq = tp.get_topic_freq([documents[i][0]], topic_title_freq)
                # contents
                topic_content_freq = tp.get_topic_freq(documents[i][1:], topic_content_freq)

            topic_token_no.append((topic, len(topic_content_freq)))
            # topic labelled frequency list
            topic_freq_list.append((topic, topic_content_freq))
            topic_title_freq_list.append((topic, topic_title_freq))

        # merging all topic_freq_dic's into collection_freq_dic
        collection_freq = dict(functools.reduce(operator.add,
                                                map(collections.Counter, [t[1] for t in topic_freq_list])))

        collection_title_freq = dict(functools.reduce(operator.add,
                                                map(collections.Counter, [t[1] for t in topic_title_freq_list])))

        # %%
        """

        3. Scaling tokens in each topic by tf-idf, choosing top tokens from each topic,
        combining as general vocabulary, sorted and ranked topic vocabulary

        """
        # TODO deleting
        if not os.path.exists(os.path.join('./','vocab.pickle')):
            vocab,topics_vocab =  tp.get_vocab(collection_freq, topic_freq_list)
            utils.serialisation('vocab.pickle',vocab)
        else: vocab = utils.load('vocab.pickle')
        if not os.path.exists(os.path.join('./','vocab_title.pickle')):
            title_vocab, topics_title_vocab = tp.get_vocab(collection_title_freq,topic_title_freq_list,300)
            utils.serialisation('vocab_title.pickle',title_vocab)
        else: title_vocab = utils.load('vocab_title.pickle')
        # %%
        """

        4. Extracting POS features and combining all features as vectors

        """
        # Feature 1: content vocabulary frequency vectors

        # Feature 2: title vocabulary frequency vectors

        # Feature 3: frequency distribution of Part of Speech Tags vectors

        in_X = []
        in_y = []

        for label in raw_data:
            pos_list = []
            content_list = []
            title_list = []
            for i in range(len(raw_data[label])):
                title = np.asarray(tp.get_freq_vecs([raw_data[label][i][0]], title_vocab)[0])
                content = np.asarray(tp.get_freq_vecs(raw_data[label][i][1:],  vocab))
                content_vec =np.zeros(len(content[0]))
                for j in range(len(content)) :
                    content_vec  = np.add(content_vec,content[j])
                pos = np.asarray(tp.get_pos_tag_list(raw_data[label][i]))
                title_list.append(title)
                content_list.append(content)
                pos_list.append(pos)
                # vectorised general X
                x = np.concatenate((title,content_vec,pos), axis=0)
                in_X.append(x)
                in_y.append(label)

        utils.serialisation(X_SERIAL,in_X)
        utils.serialisation(y_SERIAL,in_y)
    else:
        print("Cached dataset found ! Using existing cached data...")
        in_X = utils.load(X_SERIAL)
        in_y = utils.load(y_SERIAL)

    return in_X,in_y
# %%
