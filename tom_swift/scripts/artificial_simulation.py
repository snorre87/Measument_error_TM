### Artificial simulation ###
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA
import matplotlib.pyplot as plt

import seaborn as sns
from collections import Counter
import pandas as pd
import pickle
from tqdm import tqdm as tqdm
import scipy.stats
def normalize_entropy(series):
    result = scipy.stats.entropy(((series.T/series.sum(axis=1)).T).T)
    return result[0]

def run_lda(docs):
    lda = LDA(n_components=3,learning_method='batch',verbose=1)
    doc2topics = lda.fit_transform(docs)
    return doc2topics,lda
def get_measures(doc2topic):
    dist2topic = make_dist2topic(doc2topics)
    dist2max = get_main_topic(dist2topic)
    loss = get_loss(dist2topic)
    proportions = get_proportions(doc2topics)
    return dist2topic,dist2max,loss

def get_proportions(doc2topic):
    proportions = doc2topics.sum(axis=0)
    return proportions

###

def generate_docs(proportion_variance=1,subtopic_overlap=1,ratio_pos=2,ratio=1,reverse_ratio=False,n_docs = 500,docsize = 1000):
    """
    proportion_variance: determines how much the propotion of topics vary and is expressed
    as a factor multiplying the alpha parameter (0.33).

    ratio_pos: determines which topic to be dominant: the low entropy, high entropy, or the subtopic one.

    ratio: determines the topic proportion
    reverse_ratio: determine if the ratio should be applied to the two other distributions.

    subtopic_overlap: determine the overlap between the first topic
    n_docs: determines the size of the two other

    docsize: determines the count of each document."""
    ## Determine topics
    ratio = int(ratio)

    topic_proportions = np.array([0.33,0.33,0.33])*proportion_variance
    if reverse_ratio:
        topic_proportions[0] *= ratio
        topic_proportions[1] *= ratio
    else:
        topic_proportions[-1] *= ratio
    doc_topic_proportions = np.random.dirichlet(topic_proportions,size=n_docs*ratio)
    # make documents

    X = np.zeros((n_docs*ratio,n_tokens*3))

    count_distributions = np.zeros(4)
    # make simple topics
    for num in range(2):
        dist = distributions[num]
        weights = doc_topic_proportions[:,num]
        for i in range(n_docs*ratio):
            weight = weights[i]
            count_distributions[num]+=weight
            word_counts = np.random.multinomial(int(docsize*weight),dist)
            X[i,num*n_tokens:(num+1)*n_tokens] = word_counts
    # make subtopic topics
    proportions = np.array([0.5,0.5])
    sub_sizes = np.random.dirichlet(proportions*subtopic_overlap,size=n_docs*ratio)

    for num in range(2,4):
        dist = distributions[num]
        weights = doc_topic_proportions[:,2]
        sub_weights = sub_sizes[:,num-2]
        for i in range(n_docs*ratio):
            weight = weights[i]
            sub_weight = sub_weights[i]
            count_distributions[num]+=(weight*sub_weight)
            word_counts = np.random.multinomial(int(docsize*weight*sub_weight),dist)
            X[i][2*n_tokens+(num-2)*(n_tokens//2):2*n_tokens+(num-1)*(n_tokens//2)] = word_counts
    return X,sub_sizes,doc_topic_proportions,count_distributions

###
import os
files = ['simulation_results/log/'+i for i in os.listdir('simulation_results/log/') if not i=='log.csv']
dist_count = int(max(files,key=lambda x: int(x.split('_')[-1])).split('_')[-1])

### generate distributions
## Configurations,
### N_tokens
### Entropies
#entropies = [{'low_ent':2,'high_ent':3,'sub_ent':2.5},{'low_ent':1.75,'high_ent':2}]
import random,pickle
filename = random.choice(['simulation_results/log/'+i for i in os.listdir('simulation_results/log/') if not i=='log.csv'])
dist_count = int(filename.split('_')[-1])
distributions,dist_params = pickle.load(open(filename,'rb'))
n_tokens = 100

def create_distribution():
    dist_count = max([int(i.split('_')[0]) for i in os.listdir('simulation_results/log/') if not i=='log.csv'])

    entropies = [2,3]
    dominant_ent = 2.5

    dist_params = {'n_tokens':n_tokens,'dominant_entropy':dominant_ent,'entropies':entropies}
    distributions = []
    for a in entropies:
        s = np.random.zipf(a, 100)
        s.sort()

        distributions.append(s/s.sum())
    for _ in range(2):
        s = np.random.zipf(dominant_ent, 100//2)
        s.sort()
        distributions.append(s/s.sum())
    # dump distributions
    pickle.dump([distributions,dist_params],open('simulation_results/log/distribution_%d'%dist_count,'wb'))


## analyze distributions
def make_topic_co_occur(X):
    co_occur = X.T.dot(X)
    starts = [0,100,200,250,300]
    startstops = list(zip(starts[:-1],starts[1:]))
    top_occur = np.zeros((len(starts)-1,len(starts)-1))

    for num in range(4):
        start,stop = startstops[num]
        for num2 in range(num,4):
            start2,stop2 = startstops[num2]
            count = co_occur[start:stop,start2:stop2].sum()
            top_occur[num,num2] = count
            top_occur[num2,num] = count
            #in_ = co_occur[start:stop,start:stop].sum()
            #out = co_occur[start:stop,:].sum()-in_
            #print(in_,out,in_/co_occur[start:stop,:].sum())
    starts = [0,100,200,300]
    startstops = list(zip(starts[:-1],starts[1:]))
    top_occur2 = np.zeros((len(starts)-1,len(starts)-1))

    for num in range(len(starts)-1):
        start,stop = startstops[num]
        for num2 in range(num,len(starts)-1):
            start2,stop2 = startstops[num2]
            count = co_occur[start:stop,start2:stop2].sum()
            top_occur2[num,num2] = count
            top_occur2[num2,num] = count
            #in_ = co_occur[start:stop,start:stop].sum()
            #out = co_occur[start:stop,:].sum()-in_
            #print(in_,out,in_/co_occur[start:stop,:].sum())


    starts = [0,100,200,250,300]
    startstops = list(zip(starts[:-1],starts[1:]))
    w2top = np.zeros((n_tokens*3,len(starts)-1))

    for num in range(len(starts)-1):
        start,stop = startstops[num]
        for w in range(n_tokens*3):
            count = co_occur[w:,start:stop].sum()
            w2top[w,num] = count
    starts = [0,100,200,300]
    startstops = list(zip(starts[:-1],starts[1:]))
    w2top2 = np.zeros((n_tokens*3,len(starts)-1))

    for num in range(len(starts)-1):
        start,stop = startstops[num]
        for w in range(n_tokens*3):
            count = co_occur[w:,start:stop].sum()
            w2top[w,num] = count
    w_sum = X.sum(axis=0)
    return top_occur,top_occur2,co_occur,w_sum,w2top,w2top2

import pickle
def run_simulation(params):
    global t,delta_t
    # generate docs
    X,sub_sizes,doc_dist_proportions,distribution_counts = generate_docs(**params)
    doc_dist_proportions_sub = np.zeros((len(X),4))
    doc_dist_proportions_sub[:,0:2] = doc_dist_proportions[:,0:2]
    doc_dist_proportions_sub[:,2] = doc_dist_proportions[:,2]*sub_sizes[:,0]
    doc_dist_proportions_sub[:,3] = doc_dist_proportions[:,2]*sub_sizes[:,1]
    ## run lda
    doc2topics,lda = run_lda(X)
    ##TODO # calculate perplexity and performance

    
    word2topic = lda.components_
    ### Calculate CoOccurence stats.
    top_occur,top_occur2,co_occur,w_sum,w2top,w2top2 = make_topic_co_occur(X)

    ## assign each distribition to topic.
    #assigned = doc_dist_proportions.T.dot(doc2topics).argsort().T[::-1].T[:,0]
    w2loss = w_sum.copy()
    assigned = np.zeros(3,dtype=np.int)
    for num in range(3):
        start,stop = num*n_tokens,(num+1)*n_tokens

        assigned[num] = int(word2topic[:,start:stop].sum(axis=1).argmax())

        top_num = assigned[num]
        w2loss[start:stop] = w2loss[start:stop] - word2topic[top_num,start:stop]
    loss_frac = np.absolute(w2loss).sum()/w_sum.sum()
    ## measure error
    loss = np.zeros(3)
    for num in range(3):
        a,a2 = doc_dist_proportions[:,num],doc2topics[:,assigned[num]]
        loss[num] = sum(abs(a-a2))
    ## measure individual doc error.
    binned_losses = []
    binned_stds = []
    doc2loss = np.zeros((len(X),3))
    for num in range(3):
        a,a2 = doc_dist_proportions[:,num],doc2topics[:,assigned[num]]
        temp_loss = a-a2
        doc2loss[:,num] = temp_loss
        sort_idx = a.argsort()
        ## Count 20 bins loss mean and std
        a,temp_loss = a[sort_idx],temp_loss[sort_idx]
        binned_loss = []
        binned_std = []
        for i,j in zip(np.linspace(0,1,20)[1:],np.linspace(0,1,20)[1:]):
            a_idx = (a<=i)&(a<j)
            temp_i = temp_loss[a_idx]
            binned_loss.append(temp_i.mean())
            binned_std.append(temp_i.std())
        binned_losses.append(binned_loss)
        binned_stds.append(binned_std)
    ## measure word loss
    w2loss = w_sum.copy()
    for num in range(3):
        top_num = assigned[num]
        w2loss[num*n_tokens:(num+1)*n_tokens] -=word2topic[top_num,num*n_tokens:(num+1)*n_tokens]
    #w2loss = w_sum-word2topic.max(axis=0)

    ## quantify the degree to which each occurence has high topicality
    ### w_dist_proportions
    w_dist_proportions = (X.T.dot(doc_dist_proportions_sub).T/X.sum(axis=0)).T

    # dump data
    ### word_properties:
    #### w_sum: word counts in corpus
    #### word2topic : topic assignment
    #### w_dist_proportions: average fractions of each distribution pr word.
    ######(X.T.dot(doc_dist_proportions_sub).T/X.sum(axis=0)).T

    word_properties = [w_sum,word2topic,w_dist_proportions,w2loss]

    ### doc_properties
    #### doc_dist_proportions: fraction of each distribution in each document
    #### doc_dist_proportions: fraction of each distribution in each document including subtopics
    #### doc2topics: document assignment based on LDA.
    #### doc2loss: document level difference between distribution proportions and assigned topic proportions
    doc_properties = [doc2loss,doc_dist_proportions,doc_dist_proportions_sub,doc2topics]

    ### co-occurence
    #### top_occur: co-occurence matrix between distributions.
    #### top_occur2: co-occurence matrix between distributions including sub distributions
    #### w2top: co_occurence stats of each word to each distribution
    #### w2top2: co_occurence stats of each word to each distribution including subdistribution
    ####co_occur: co occurence matrix between all words.
    co_occurence = [top_occur,top_occur2,w2top,w2top2,co_occur]

    ### performance
    #### loss: absolute deviation from real distribution proportions to assigned topic proportions.
    #### binned_losses: losses as a function of document proportions (small to high dist proportion)
    ##### binned into 20 different proportions.
    #### binned_stds: variance in the proportion bin.
    performance = [loss,loss_frac,binned_losses, binned_stds]

    ## meta
    ### params
    meta = [params,dist_count]

    for name,l in zip(['wprop','dprop','cooccur','performance','meta'],[word_properties,doc_properties,co_occurence,performance,meta]):
        filename = basename+'_%s'%name
        with open(filename,'wb') as f:
            pickle.dump(l,f)
    ### write log
    loss = loss.sum()
    ## write log
    ## header = ['t','delta_t','distribution_id','loss','loss_frac']+keys
    delta_t = time.time()-t
    row = [t,delta_t,dist_count,modelid,loss,loss_frac]+[params[i] for i in keys]
    logfile.write(','.join(map(str,row))+'\n')
    logfile.flush()
    ## reset time
    t = delta_t

## generate parameter space
"""
    proportion_variance: determines how much the propotion of topics vary and is expressed
    as a factor multiplying the alpha parameter (0.33).

    ratio_pos: determines which topic to be dominant: the low entropy, high entropy, or the subtopic one.

    ratio: determines the topic proportion
    reverse_ratio: determine if the ratio should be applied to the two other distributions.

    subtopic_overlap: determine the overlap between the first topic
    n_docs: determines the size of the two other

    docsize: determines the count of each document."""


docsizes = [100,200]
ndocs = [500]
proportion_variance = [0.1,1,3,5,10,15]
ratios = np.linspace(1,100,50)
reverse = [True,False]
## the higher the stronger overlap
subtopic_overlap = [0.1,1,3,5,10,15]
ratio_pos = [2]
import itertools
parameters = sorted(list(itertools.product(*[docsizes,ndocs,proportion_variance,ratios,reverse,subtopic_overlap,ratio_pos])),key=lambda x: x[4])
keys = 'docsize,n_docs,proportion_variance,ratio,reverse_ratio,subtopic_overlap,ratio_pos'.split(',')

if not os.path.isfile('simulation_results/log/log.csv'):
    logfile = open('simulation_results/log/log.csv','w')
    header = ['t','delta_t','distribution_id','model_id','loss']+keys
    logfile.write(','.join(header)+'\n')
    logfile.close()

#done_df = pd.read_csv('simulation_results/log/log.csv')
done_df = pd.read_csv('simulation_results/log/log.csv',skiprows=[0],names  = ['t','delta_t','distribution_id','model_id','loss','loss_frac']+keys)
done = set([tuple(row) for row in done_df[done_df.distribution_id==dist_count][keys].values])
print('%d already done'%len(done))

basepath = 'simulation_results/results/'
try:
    modelid = int(max(os.listdir(basepath),key=lambda x: int(x.split('_')[0])).split('_')[0])
except:
    modelid = 0
print(modelid)
modelid+=1

overwrite = False
if overwrite:
    modelid = 0

logfile = open('simulation_results/log/log.csv','a')
import time
t = time.time()
for params in tqdm(parameters):

    if tuple(params) in done:
        continue
    params = dict(zip(keys,params))
    basename = basepath+str(modelid)

    run_simulation(params)
    ## update modelid
    modelid+=1
