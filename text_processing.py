import nltk
import numpy as np
import operator
from math import log

nltk.download('stopwords') # If needed
nltk.download('punkt') # If needed
nltk.download('wordnet') # If needed
nltk.download('averaged_perceptron_tagger')# If needed

lemmatizer = nltk.stem.WordNetLemmatizer()
stopwords=set(nltk.corpus.stopwords.words('english'))
stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")
stopwords.add("!")
stopwords.add("''")
stopwords.add("'")
stopwords.add("(")
stopwords.add(")")

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# Function taken from Session 1
def get_list_tokens(string): # Function to retrieve the list of tokens from a string
  sentence_split=nltk.tokenize.sent_tokenize(string)
  list_tokens=[]
  for sentence in sentence_split:
    list_tokens_sentence=nltk.tokenize.word_tokenize(sentence)
    for token in list_tokens_sentence:
      list_tokens.append(lemmatizer.lemmatize(token).lower())
  return list_tokens

def get_list_filtered_tokens(string):
  sentence_split=nltk.tokenize.sent_tokenize(string)
  list_tokens=[]
  for sentence in sentence_split:
    list_tokens_sentence=nltk.tokenize.word_tokenize(sentence)
    for token in list_tokens_sentence:
      if token in stopwords: continue
      else: list_tokens.append(lemmatizer.lemmatize(token).lower())
  return list_tokens


# get topic frequency
def get_topic_freq(dataset, token_frequency={}):
  for instance in dataset:
    sentence_tokens=get_list_tokens(instance)
    for word in sentence_tokens:
      if word in stopwords: continue
      if word not in token_frequency: token_frequency[word]=1
      else: token_frequency[word]+=1
  return token_frequency
  
# get document frequency
def get_df(topic_freq_list,topic_label,token):
    df = 1
    for i in range(len(topic_freq_list)):
        if topic_freq_list[i][0] == topic_label: continue
        if topic_label in topic_freq_list[i][1]: df+=1
    return df

""" 

TF-IDF scaling 

"""
# TODO data structure simplifying
def freq_scaler(collection_freq,topic_freq_list):
    # scaling token frequencies according to topic relevance
    for i in range(len(topic_freq_list)):
        freq_dic = topic_freq_list[i][1]
        for token in freq_dic:     
            i_tf = freq_dic[token]
            i_cf = collection_freq[token]
            if i_cf != 0:
                # cacluate weighting
                i_tf_idf = round((i_tf/i_cf)*log(len(topic_freq_list)/get_df(topic_freq_list,topic_freq_list[i][0],token)),5)
                # scaling
                topic_freq_list[i][1][token] = round(i_tf*i_tf_idf,5)
            else: topic_freq_list[i][1][token] = 0
    return topic_freq_list


def get_vocab(collection_freq,topic_freq_list,top_num=1000):
  topic_freq_list = freq_scaler(collection_freq,topic_freq_list)
  vocab = []
  topics_vocab = []
  for topic_tuple in topic_freq_list:
       sorted_list = sorted(topic_tuple[1].items(), key=lambda item: item[1],reverse=True)[:top_num]
       topics_vocab.append((topic_tuple[0],sorted_list))
       for word,frequency in sorted_list:
         vocab.append(word)
  return list(set(vocab)),topics_vocab

# Function taken from Session 2
def get_vector_text(list_vocab,string):
  vector_text=np.zeros(len(list_vocab))
  list_tokens_string=get_list_tokens(string)
  for i, word in enumerate(list_vocab):
    if word in list_tokens_string:
      vector_text[i]=list_tokens_string.count(word)
  return vector_text

def get_freq_vecs(any_set, vocab):
  X=[]
  for instance in any_set:
    vector_instance=get_vector_text(vocab,instance)
    X.append(vector_instance)
  return X

# get POS count of a single sentence
def check_pos_tag(string, pos_counter=[0,0,0,0,0]):
    line_tokens = get_list_tokens(string)
    sent_pos =  nltk.pos_tag(line_tokens)
    for i, tag in enumerate(pos_family):
        for pos_tuple in sent_pos:
            if pos_tuple[1] in pos_family[tag]:
                pos_counter[i]+=1
    return pos_counter

def get_pos_tag_list (sent_set):
    set_pos = [0,0,0,0,0]
    for sent in sent_set:
        set_pos = check_pos_tag(sent,set_pos)
    return set_pos


""" 

retrival tokens from news title (first line of document)

"""
def get_title_features(doc,topic_vocab_list):
 
    title_tokens = list_tokens_string=get_list_tokens(doc[0])
    title_topic_freq_list = []
    for topic_vocab in topic_vocab_list:
        title_topic_freq = 0
        for token in title_tokens:
            if token in topic_vocab[1]:
                title_topic_freq+=1
        title_topic_freq_list.append(title_topic_freq)
    return title_topic_freq_list
