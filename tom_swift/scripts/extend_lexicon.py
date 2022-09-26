import time
import pandas as pd
import codecs
import json,os
Index = json.load(codecs.open('Word_Index','r','utf-8'))
import pickle
#topwords = pickle.load(open('Topword_Occurence.pkl','rb'))
topwords = pickle.load(open('Topword_Occurence_frex.pkl','rb'))
#suspect_df = pd.read_csv('lexicon_extension_suspects.csv')
suspect_df = pd.read_csv('lexicon_extension_suspects_frex.csv')
lex2idx = pickle.load(open('lexicon2idx.pkl','rb'))

def load_dataframe(filename,names = ['word','idx','include','top_occurence']): # include best_match category
    try:
        df = pd.read_csv(filename,names = names,encoding='latin1')
    except:
        df = pd.read_csv(filename,names = names,encoding='utf-8')
    return df
def get_never_include():
    import os
    files = [i for i in os.listdir('/') if 'lexicon_extension' in i and 'suspects' not in i]
    never = set()
    for filename in files:
        try:
            done_df = load_dataframe(filename)
            #done_df = pd.read_csv(filename,names=['word','idx','include','top_occurence'])
        except:
            continue
        never.update(set(done_df[done_df['include']<0].idx))
    return never
never = get_never_include()
print(len(never))
idx2lex = pickle.load(open('idx2lex_new.pkl','rb'))
last_t = time.time()
time_f = open('crime_timespent','a')
for name,groupdf in suspect_df.groupby('best_match'):
    time_f = open('lexicon_timespent_%s'%name,'a')
    filename = 'lexicon_extension_%s.csv'%name
    f = open(filename,'a')
    header = ['idx','word','include','top_occurence']
    all_oc = sum(groupdf.top_occurence)
    try:
        done_df = load_dataframe(filename)
        #done_df = pd.read_csv(filename,names=['word','idx','include','top_occurence'])
        done = set(done_df.idx) # 2 is always, 1 is sometimes, 0 is no, -1 is never in any dictionary
    except Exception as e:
        print(str(e))
        print(os.path.getsize(filename))
        done_df = pd.DataFrame([],columns = ['word','idx','include','top_occurence'])
        done = set()
    indices = lex2idx[groupdf.best_match_dk.values[0]]
    print(name,groupdf.best_match_dk.values[0],len(indices))
    best_indices = sorted(indices,key=lambda x: topwords[x],reverse=True)[0:5]
    # print best words describing the Class
    for idx in best_indices:
        print(Index[idx],topwords[idx])

    print('%d words seen, %d included already that amounts to %d topoccurences %r'%(len(done),len(done_df[done_df.include>2]),sum(done_df[done_df.include>2].top_occurence),sum(done_df.top_occurence)/all_oc))
    # filter insignifikant words for now
    groupdf = groupdf[groupdf['n_matches']>5]
    occurence_count = done_df.top_occurence.sum()
    input('Are you ready to start? press enter')
    count=0

    for w,idx,occurence in groupdf.sort_values('all_frac',ascending=False)[['w','idx','top_occurence']].values:
        if idx in done or idx in indices:# or idx in never:
            continue
        if idx in idx2lex:
            continue
        if str(idx) in idx2lex:
            continue
        print()
        print(w,occurence,idx)

        include = int(input('Do you want to include this:\n %s? 3 always,2 sometimes, 1 might make sense, 0 not this, -1 never, 9 skip category\n'%w))
        if include==9:
            break
        if include > 2:
            occurence_count+=occurence
            #['word','idx','include','top_occurence']
        row = [w,idx,include,occurence]
        f.write(','.join(map(str,row))+'\n')
        f.flush()
        t = time.time()
        time_spent =t - last_t
        last_t = t
        time_f.write('%r,%r\n'%(time_spent,t))
        done.add(idx)
        if include==-1:
            never.add(idx)
        count+=1
        if count%10==0:
            continue_ = input('You have now included %d new top occurences from %d terms to the lexicon, and gone through %d new terms.\nDo you want to continue to next? press y or Enter.'%(occurence_count,len(done),count))
            if continue_=='y':
                continue
            else:
                break
