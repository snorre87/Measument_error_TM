# TOPIC MODEL SIMULATION

# import dependencies
import nltk
import random
#nltk.download() # download nltk ressources
import pandas as pd
import gensim
import unicodecsv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import codecs
import re
from tmtoolkit.topicmod import tm_lda



import json
noise = []
for dat in json.load(open('trump+hillary_tp_noise.json','r')):

    try:
        page_id = dat['id']
    except:
        page_id = dat['PID']
    try:
        name = dat['name']
    except: # assumme same as last
        pass
    d = {'page_id':page_id,'page_name':name}
    try:
        data = dat['feed']['data']
    except:
        data = dat['data']
    for d1 in data:
        d1.update(d)
        noise.append(d1)

noise_df = pd.DataFrame(noise)
noise_df = noise_df.drop_duplicates('id')
noise_df = noise_df[noise_df.message.apply(lambda x: type(x)==str)]
import datetime
noise_df['date'] = noise_df.created_time.apply(lambda x: pd.to_datetime(x).date())


url_pattern = '(?:(?:https?:\/\/|www\.)\w+[\.\w]*\.\w+)|(?:\w+\.(?:com|dk))|(?:https?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
url_re = re.compile(url_pattern)
url_sub = ' __url__ '
digit_pattern = r"\d(?:[\.,\d]*\d+)|\d{2,}(?:[\.,\d]*\d+)?" # matching >1 figure digits
replace_regex = [(url_sub,url_re),(' __digit__ ',re.compile(digit_pattern))]



def filter_words(words,stopwords):
    # filter non signs
    words = [word for word in words if isalphanum(word)]
    # filter stopwords
    words = [word for word in words if word not in stopwords]
    return words
def isalphanum(string):
    if not string.isalnum():
        # find word with - between
        pattern = r'[a-zAZøæåÆØÅ]+\-[a-zAZøæåÆØÅ]*|__[a-zAZøæåÆØÅ]+_*'
        search = re.findall(pattern,string)
        if len(search)>0:
            return True
        else:
            return False
    return True
def process_documents(text,stopwords):
    # replace known patterns.
    for sub_pattern,regex in replace_regex:
        text = regex.sub(sub_pattern,text)
    text = nltk.tokenize.word_tokenize(text)
    # lower
    text = [i.lower() for i in text]
    # remove stopwords
    text = filter_words(text,stopwords)
    return text
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english')) | set(['__digit__'])# english # 'hillary','trump'
stopwords = set(nltk.corpus.stopwords.words('danish'))|set(open('danish_stopwords.txt','r').read().split('\n'))|set(['__digit__']) # danish

noise_docs = noise_df.message.apply(process_documents,**{'stopwords':stopWords}).values


df = pd.read_csv('data/topic_modelling_dataset.csv').drop('bow',axis=1)
len(df)


real_docs = df.message.apply(process_documents,**{'stopwords':stopwords}).values


def make_index(texts,min_thres=3,max_thres=0.5,max_n=25000):
    c = Counter()
    d = Counter()
    for text in texts:
        s = Counter(text)
        c.update(s)
        d.update(set(s))
    return {w:num for num,w in enumerate(sorted(c,key=lambda x: c[x],reverse=True)[:max_n]) if c[w]>min_thres and d[w]/len(texts)<max_thres}

def to_dense(corpus,vocab_size):
    X = np.zeros((len(corpus),vocab_size),dtype=np.int32)
    for num in range(len(corpus)):
        bow = corpus[num]
        for w,count in bow.items():
            try:
                X[num][w]=count
            except:
                pass
    return X
import scipy.sparse as sp
def to_sparse(corpus,vocab_size):
    X = sp.dok_matrix((len(corpus),vocab_size), dtype=np.int32)
    for num in range(len(corpus)):
        bow = corpus[num]
        for w,count in bow.items():
            try:
                X[num][w]=count
            except:
                pass
    X = X.tocsr()
    return X

def to_index(text,d):
    return [d[i] for i in text if i in d]
def to_bow(texts,d):
    global vocab_size
    bows = [Counter(to_index(text,d)) for text in texts]
    return to_dense(bows,vocab_size)


real_index = make_index(real_docs,max_n=30000,min_thres=2)
noise_index = make_index(noise_docs,min_thres=2)
Index = real_index.copy()
out = set(noise_index)-set(real_index)
print(len(out),len(noise_index))
for w in out:
    idx = noise_index[w]+len(real_index)
    Index[w] = idx
vocab_size = len(Index)


real_bows = to_bow(real_docs,Index)
noise_bows = to_bow(noise_docs,Index)
noise_start_d = len(real_bows)
noise_start_w = len(real_index)
# create docs lengths
real_length = real_bows.sum(axis=0)

print(sum(real_bows.sum(axis=1)>0),len(real_bows),sum(noise_bows.sum(axis=1)>0),len(noise_bows))



df = df[real_bows.sum(axis=1)>0]
noise_df = noise_df[noise_bows.sum(axis=1)>0]

noise_bows = noise_bows[noise_bows.sum(axis=1)>0]
real_bows = real_bows[real_bows.sum(axis=1)>0]

import pickle
print(pickle.compatible_formats)
pickle.dump([df,noise_df,sorted(Index,key=lambda x: Index[x])],open('topic_simulation_dependencies.pkl','wb'),protocol=3)

# Define Noise
import numpy as np
ratio = len(noise_bows)/(len(real_bows)+len(noise_bows))
noise_share = np.linspace(0,0.05,20)
import random
def get_noise(share):
    # calculate number of noise examples to be sampled
    n = len(real_bows)*share / (1-share)
    noise_idx = random.choices(range(len(noise_bows)),k=int(n))
    return noise_idx

metrics = ('loglikelihood',
 'cao_juan_2009',
 'arun_2010',
 'coherence_mimno_2011',
 'griffiths_2004')
import random
def run_simulation(X):
    # this will evaluate the models in parallel
    new_params = varying_params.copy()
    for d in new_params:
        d['random_state'] = random.choice(range(5000)) # random_state use to be set to 1 constantly.
    models = tm_lda.evaluate_topic_models(X, new_params, const_params,
                                          return_models=True,metric=metrics)
    return models
# Dump metrics, and models.
def dump_metrics():
    scores = []
    for num,model in enumerate(models):
        modelid = num+model_n
        d = {'noise':share,'simulationNumber':simulation,'modelid':modelid}
        for key,val in model[1].items():
            if key=='model':
                continue
            d[key] = val
        for key,val in model[0].items():
            d[key] = val
        scores.append(d)

    basename = 'simulation_results/metrics/index1_%d_%r_%d'%(simulation,round(share,3),modelid)
    scores = pd.DataFrame(scores)
    scores.to_csv(basename,index=False)
    basename = 'simulation_results/metrics/noise_index_%d_%r_%d'%(simulation,round(share,3),modelid) # added 10:12 25.10.2018
    json.dump(noise_index,open(basename,'w'))

# Dump distributions
def get_top(m):
    doc2topic = m.doc_topic_

    top50 = []
    top50vals = []
    top1000 = []
    top1000vals = []
    for i in range(len(doc2topic[0])):
        idx = np.argsort(doc2topic[:,i])[-50:][::-1]
        top50.append(idx)
        top50vals.append(doc2topic[idx][i])

        idx = np.argsort(m.topic_word_[i,:])[-1000:][::-1]
        top1000.append(idx)
        top1000vals.append(m.topic_word_[i][idx])
    return list(top50),list(top50vals),list(top1000),list(top1000vals)
import pickle
# reveight lambda factor

from tmtoolkit.topicmod.model_stats import  get_marginal_topic_distrib
from tmtoolkit.topicmod.model_stats import get_word_saliency
from tmtoolkit.topicmod.model_stats import get_topic_word_relevance

def dump_distribution(): # uses global iteration number as filename
    for num,model in enumerate(models):
        modelid = num+model_n
        m = model[1]['model']
        doc2topic = m.doc_topic_
        w2topic = m.topic_word_ # fix index problem?

        basename = '/mnt/b0c8e396-e5ba-4614-be6f-146c4c861ab3/data/topic_model_simulation/models/%d_%r_%d'%(simulation,round(share,3),modelid)
        sparse_matrix = sp.csr_matrix(doc2topic)
        sp.save_npz(basename+'_doc2topic',sparse_matrix)
        sparse_matrix = sp.csr_matrix(w2topic)
        sp.save_npz(basename+'_w2topic',sparse_matrix)

        # score words by relevance
        lambda_ = 0.6
        rel_mat = get_topic_word_relevance(m.topic_word_, m.doc_topic_, doc_lengths, lambda_=lambda_)
        np.save(basename+_'w2topic_relevance',rel_mat)
        # topic assignment counts of each word
        nzw = m.nzw_
        sparse_matrix = sp.csr_matrix(nzw)
        sp.save_npz(basename+'_w2count_topic',sparse_matrix)

        # dump basic data
        basename = 'simulation_results/distributions/%d_%r_%d'%(simulation,round(share,3),modelid)
        # top50 documents,top1000 words
        #pickle.dump(get_top(m),open(basename+'_top','wb'))

### document2topic,word2topic,
const_params = dict(n_iter=1000)#, random_state=1)
varying_params = [dict(n_topics=k, alpha=1.0/k) for k in range(10, 251, 5)]
import tqdm
from os import listdir
done = set(listdir('simulation_results/metrics/'))

n_simulations = 20
model2index = {}
try:
    model_n = max([int(i.split('_')[-1]) for i in done])
except:
    model_n = 0
import time
ftime = open('simulation_time','a')
ftime.write('%r,%d\n'%(time.time(),model_n))
for simulation in range(n_simulations):
    for share in tqdm.tqdm(noise_share):
        basename = 'simulation_results/metrics/index1_%d_%r_%d'%(simulation,round(share,3),model_n)
        if basename in done:
            continue
        if share!=0:
            noise_index = get_noise(share)
        else:
            noise_index = []

        data = []
        for row in real_bows:
            data.append(row)
        for idx in noise_index:
            data.append(noise_bows[idx])
        X = np.array(data)
        del data
        ## make sure that no rows are completely empty in the noise_idx
        sub = X[:,noise_start_w:]
        sub = sub[noise_start_d:]
        zero = sub.sum(axis=0)>0
        if len(noise_index)!=0:
            for col in np.arange(len(zero))[~zero]:
                rand_row = random.choice(np.arange(noise_start_d,len(X)))
                noise_col = noise_start_w+col
                X[rand_row,noise_col] = 1
            ##
        # get document lengths
        noise_length = X[-len(noise_index)].sum(axis=0)
        # dump document_length.
        doc_length = np.concatenate([real_length,noise_length])


        models = run_simulation(X)
        dump_metrics()
        dump_distribution()
        model_n +=len(models)
        print(model_n,end=' ')
        ftime.write('%r,%d\n'%(time.time(),model_n))
    ftime.write('%d,%d\n'%(simulation,model_n))
