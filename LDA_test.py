import tomotopy as tp
import os
import os.path
import utils
import prepocessing as pp
import text_processing as _tp

k_topic = 5

mdl = tp.LDAModel(tw=tp.TermWeight.IDF,k=k_topic)

for i in range(len(pp.topic_label_map)):
        topic_path = os.path.join(pp.LOCAL_DIR, pp.topic_label_map[i][0])
        for doc_name in os.listdir(topic_path):
            doc_path = os.path.join(topic_path, doc_name)
            if os.path.isfile(doc_path):
                lines_set = utils.load_local_dataset(doc_path)
                doc_token_list = []
                for line in lines_set:
                    doc_token_list.extend(_tp.get_list_filtered_tokens(line))
                mdl.add_doc(doc_token_list)

for i in range(0, len(mdl.docs), k_topic):
    mdl.train(5)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

for k in range(mdl.k):
    print('Top 5 words of topic #{}'.format(k))
    print(mdl.get_topic_words(k, top_n=k_topic))

mdl.summary()

