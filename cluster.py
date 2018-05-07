# -*- encoding:utf8 -*-
from __future__ import division
from gensim import corpora, models, similarities
from sklearn.cluster import KMeans
import jieba
import codecs
import json
from xlwt import Workbook
from strip_tags import strip_tags
from datetime import datetime

def get_stocks_dict():
    f = codecs.open('stocks_for_day/%d.json' % day, encoding='utf-8')
    f_d = json.load(f)
    return f_d

def get_news_list():
    newsList = []
    f = open('newsList/%d.txt' % day, 'r')
    for line in f.readlines():
        line = line.strip()
        newsList.append(int(line))
    return newsList

def segmentation(corpus):
    jieba.load_userdict('WordsDic/userdict.txt')
    seg_corpus = []
    for ele in corpus:
        seg_list = jieba.lcut(ele, cut_all=False)
        seg_corpus.append(seg_list)
    return seg_corpus

def computeSimilarity_lsm(X, query):
    index = similarities.MatrixSimilarity(X)
    sims = index[query]
    scoreList = list(enumerate(sims))
    rankList = [scoreList[i][1] for i in range(len(scoreList))]
    return rankList

def get_entity_corpus(news):
    t = []  # title entities lists
    f_t = codecs.open('nn_crf_title_ner/%d.json' % day, encoding='utf-8')
    for line in f_t.readlines():
        line_dict = json.loads(line)
        entities = line_dict["entities"]
        t_words = []
        for ele in entities:
            t_words.append(ele["word"])
        t.append(t_words)
    c = []  # content entities lists
    f_c = codecs.open('nn_crf_content_ner/%d.json' % day, encoding='utf-8')
    for line in f_c.readlines():
        line_dict = json.loads(line)
        entities = line_dict["entities"]
        c_words = []
        for ele in entities:
            c_words.append(ele["word"])
        c.append(c_words)
    entity_corpus = []
    for i in range(len(t)):
        t_w = t[i]
        c_w = c[i]
        c_w.extend(t_w)  # BILSTM+CRF entities
        # m = newsList[i]
        # f_t = codecs.open('../title_data/inputTitleData/%d.json' % m, encoding='utf-8')
        # f_t_d = json.load(f_t)
        # t_e = f_t_d["Entity"]
        # f_c = codecs.open('../content_data/inputContentData/%d.json' % m, encoding='utf-8')
        # f_c_d = json.load(f_c)
        # c_e = f_c_d["Entity"]
        # c_e.extend(t_e)  # StanfordNLP entities
        # s_e = []
        # for ele in c_e:
        #     if ele not in c_w:
        #         s_e.append(ele)
        # c_w.extend(s_e)
        entity_corpus.append(c_w)
    entity_corpus_ = []
    for ele in news:
        entity_corpus_.append(entity_corpus[newsList.index(ele)])
    return entity_corpus_

def cluster(myfile, corpus, num_topics):
    dictionary = corpora.Dictionary(corpus)
    length_of_dictionary = len(dictionary)
    print("length of dictionary: ", length_of_dictionary)
    doc_vectors = [dictionary.doc2bow(text) for text in corpus]
    # TF-IDF
    tfidf = models.TfidfModel(doc_vectors)
    tfidf_vectors = tfidf[doc_vectors]
    # LSI
    lsi = models.LsiModel(tfidf_vectors, id2word=dictionary, num_topics=num_topics)
    lsi_vectors = lsi[tfidf_vectors]

    fout = codecs.open(myfile + '/lsi_entity%d.txt' % day, 'w', encoding='utf-8')
    for ele in lsi_vectors:
        fout.write(str(ele))
        fout.write('\n')
    fout.close()

    # get lsi vector
    vec_ = []
    for _ in lsi_vectors:
        L = [ele[1] for ele in _]
        vec_.append(L)
    vec = []
    for ele in vec_:
        if len(ele) == 0:
            ele = [0] * num_topics
            vec.append(ele)
        else:
            vec.append(ele)
    num_of_clusters = num_topics
    km = KMeans(n_clusters=num_of_clusters)
    result = km.fit_predict(vec)
    return result

def get_cluster_result(myfile, result, num_topics):
    f = codecs.open(myfile+'/cluster%d.txt' % day, 'w', encoding='utf-8')
    for i in range(0, num_topics):
        f_text = codecs.open('text/text%d.txt' % day, 'r', encoding='utf-8')
        for linenumber, eachline in enumerate(f_text.readlines()):
            if linenumber >= len(result):
                break
            if result[linenumber] == i:
                f.write(str(newsList[linenumber]) + '#' + str(i) + '#' + eachline)
        f_text.close()
    f.close()

def get_result(Threshold, myfile, data, result_pair, stocks, del_news_id, s_result_pair):
    news_id = []
    type_list = []
    word_list = []
    for i in range(len(data)):
        id = data[i][0]
        type = data[i][1]
        word = data[i][2]
        news_id.append(id)
        type_list.append(type)
        word_list.append(word)

    event_list = [ele[0] for ele in result_pair]
    score_list = [ele[1] for ele in result_pair]
    s_score_list = [ele[1] for ele in s_result_pair]

    new_type_list = []
    new_word_list = []
    new_score_list = []
    new_s_score_list = []
    new_news_id = []
    for i in range(len(type_list)):
        if int(type_list[i]) in event_list and news_id[i] not in del_news_id:
            index = int(type_list[i])
            new_news_id.append(news_id[i])
            new_type_list.append(type_list[i])
            new_word_list.append(word_list[i])
            new_score_list.append(score_list[index])
            new_s_score_list.append(s_score_list[index])

    stocks_list = []
    new_news_id1 = [str(ele) for ele in new_news_id]
    for _ in new_news_id1:
        stocks_list.append(stocks[_])

    book = Workbook()
    sheet1 = book.add_sheet('sheet1')
    length2 = len(new_type_list)
    k = -1
    for i in range(length2):
        if new_score_list[i] >= Threshold and new_s_score_list[i] > 0:
            k = k + 1
            row = sheet1.row(k)
            row.write(0, new_news_id[i])
            row.write(1, new_type_list[i])
            row.write(2, new_score_list[i])  # 全文本相似度(lsi)
            row.write(3, new_s_score_list[i])  # stock相似度
            row.write(4, stocks_list[i])
            row.write(5, new_word_list[i][:150])
    book.save(myfile+'/%d.xls' % day)

def get_top_clusters(Threshold, myfile, num_topics):
    # feature：all text words
    f = codecs.open(myfile+'/cluster%d.txt' % day, 'r', encoding='utf-8')
    data = []
    for line in f.readlines():
        data_pair = line.split('#')
        data.append(data_pair)

    text = []
    for ele in data:
        text.append(ele[2].replace('\r\n', ''))
    text_corpus = segmentation(text)

    news = []
    for ele in data:
        news.append(ele[0])

    topic_list = []
    for i in range(num_topics):
        num = 0
        for ele in data:
            if int(ele[1]) == i:
                num += 1
        topic_list.append(num)

    dictionary = corpora.Dictionary(text_corpus)
    doc_vectors = [dictionary.doc2bow(text) for text in text_corpus]
    tfidf = models.TfidfModel(doc_vectors)
    tfidf_vectors = tfidf[doc_vectors]
    lsi = models.LsiModel(tfidf_vectors, id2word=dictionary, num_topics=num_topics)
    lsi_vectors = lsi[tfidf_vectors]

    fout = codecs.open(myfile+'/lsi%d.txt' % day, 'w', encoding='utf-8')
    for ele in lsi_vectors:
        fout.write(str(ele))
        fout.write('\n')
    fout.close()

    # compute the similarity
    result = []
    del_news_id = []
    del_news_type = []
    for i in range(len(topic_list)):
        l = sum(topic_list[:i])
        r = l + topic_list[i]
        X = lsi_vectors[l: r]
        if topic_list[i] >= 200:
            result.append(0.0)
        elif topic_list[i] <= 1:
            result.append(0.0)
        else:
            X_score = []
            for j in range(topic_list[i]):
                query = X[j]
                scoreList = computeSimilarity_lsm(X, query)
                X_score.append(scoreList)
            num_of_compute = topic_list[i] * topic_list[i] - topic_list[i]
            score = (sum([sum(ele) for ele in X_score]) - topic_list[i]) / num_of_compute

            every_list = []
            for ele in X_score:
                every_list.append((sum(ele) - 1) / (len(ele) - 1))
            every = score
            delete = []
            for k in range(len(every_list)):
                if every - every_list[k] > 0.1:
                    del_news_id.append(news[l + k])
                    del_news_type.append(i)
                    delete.append(k)

            new_X = []
            for _ in range(topic_list[i]):
                if _ not in delete:
                    new_X.append(X[_])
            new_X_score = []
            for j in range(len(new_X)):
                query = new_X[j]
                scoreList = computeSimilarity_lsm(new_X, query)
                new_X_score.append(scoreList)
            length = topic_list[i] - len(delete)
            new_num_of_compute = length * length - length
            new_score = (sum([sum(ele) for ele in new_X_score]) - length) / new_num_of_compute
            result.append(new_score)
    result_pair = []
    for index, s in enumerate(result):
        result_pair.append((index, s))

    # feature：stocks
    stocks = get_stocks_dict()
    new = [str(ele) for ele in news]
    stocks_corpus = []
    for ele in new:
        stocks_corpus.append(stocks[ele])
    s_dictionary = corpora.Dictionary(stocks_corpus)
    s_doc_vectors = [s_dictionary.doc2bow(text) for text in stocks_corpus]
    s_tfidf = models.TfidfModel(s_doc_vectors)
    s_tfidf_vectors = s_tfidf[s_doc_vectors]
    # if num_topics >= 10:
    #     num_of_stocks_topics = num_topics // 10
    # else:
    #     num_of_stocks_topics = num_topics
    # s_lsi = models.LsiModel(s_tfidf_vectors, id2word=s_dictionary, num_topics=num_of_stocks_topics)
    # s_lsi_vectors = s_lsi[s_tfidf_vectors]
    # compute the similarity
    s_result = []
    for i in range(len(topic_list)):
        l = sum(topic_list[:i])
        r = l + topic_list[i]
        X = s_tfidf_vectors[l: r]
        X_ = []
        for _ in X:
            if len(_) == 0:
                ele = []
                for j in range(5):
                    ele.append((j, 1.0))
                X_.append(ele)
            else:
                X_.append(_)
        if len(X_) > 200:
            s_result.append(0.0)
        else:
            X_score = []
            for j in range(topic_list[i]):
                query = X_[j]
                scoreList = computeSimilarity_lsm(X_, query)
                X_score.append(scoreList)
            if len(X_score) > 1:
                num_of_compute = (topic_list[i] * topic_list[i] - topic_list[i]) * 0.5
                score = (sum([sum(ele) for ele in X_score if ele > 0]) - topic_list[i]) / (2 * num_of_compute)
                s_result.append(score)
            else:
                s_result.append(0.0)
    s_result_pair = []
    for index, s in enumerate(s_result):
        s_result_pair.append((index, s))

    get_result(Threshold, myfile, data, result_pair, stocks, del_news_id, s_result_pair)

if __name__ == '__main__':
    d0 = datetime.now()
    num_of_days = 302
    average_size_of_a_topic = 15
    for i in range(num_of_days):
        day = i + 1
        print("cluster for day %d" % day)
        newsList = get_news_list()
        num_of_topics = len(newsList) // average_size_of_a_topic
        print("number of topics for day %d: " % day, num_of_topics)
        entity_corpus = get_entity_corpus(newsList)
        filename = 'result'
        cluster_result = cluster(filename, entity_corpus, num_of_topics)
        get_cluster_result(filename, cluster_result, num_of_topics)
        cluster_Threshold = 0.75
        get_top_clusters(cluster_Threshold, filename, num_of_topics)
    d1 = datetime.now()
    print(d0, d1, d1-d0)






