#usr/bin/env python3

nsent = 25
ignorelen = 15
output_size = 50
import numpy as np

vec = {}
with open('./glove.txt') as f:
    for line in f:
        l = line.split()
        word = l[0]
        v = np.array(list(map(float, l[1:])))

        vec[word] = v

import os
from nltk.tokenize import sent_tokenize
from ast import literal_eval
import numpy as np

folder = [os.path.join(dp, f) for dp, dn, filenames in os.walk('./bbc') for f in filenames if os.path.splitext(f)[1] == '.txt']

def get_vector(word):
    return vec.get(word, None)

def make_vectors_from_news():
    counter = 1
    for news in folder:
        print(news)
        with open(news) as f:
            flag = True

            for line in f:
                if len(line) <2:
                    continue
                if flag:
                    with open('./sent_rep/'+str(counter)+'.headline','a') as headline:
                        print(list(filter(None, map(get_vector, line.split()))), file=headline)
                    flag = False
                else:
                    for sent in sent_tokenize(line):
                        with open('./sent_rep/'+str(counter)+'.news','a') as news:
                            print(list(filter(lambda x: (x is not None) and len(x)>0, map(get_vector, sent.split()))), file=news)
        counter +=1

def chunk(s, sz):
    seq=s.copy()
    final, inter = [], []
    while len(seq) > 0:
        inter.append(seq.pop(0))
        if len(inter) == sz:
            final.append(inter.copy())
            inter = []
    if len(inter) > 0:
        final.append(inter)
    return final

def find_nearest_word(vector):
    best_word=" "
    best_dist=100
    # for item in vector:
    #     best_dist+=item**2

    for key,val in vec.items():
        dist=abs(sum((val-vector)**2))
        # for i in range(len(vector)):
        #     dist += (val[i]-vector[i])**2

        if dist < best_dist:
            # print(dist)
            # print(key)
            best_dist = dist
            best_word = key

    return best_word

def find_vectors(text):
    vec_list = []

    for sent in sent_tokenize(text):
        vec_list.append( list( map(get_vector, sent.split())) )

    return vec_list

def read_vectors_for_model(i):
    with open('./sent_rep/'+str(i)+'.headline') as f:
        l=literal_eval(f.read())
        headline = np.array(l)
        print(headline.shape)
        
        pad_vectors = 10 - headline.shape[0]
        padding = np.zeros((pad_vectors, output_size))
        headline = np.append(headline, padding, axis=0)
        headline = headline.reshape((1,-1,output_size))

    newslist = []
    with open('./sent_rep/'+str(i)+'.news') as f:
        news = []
        for line in f:
            sent = np.array(literal_eval(line)).reshape((1,-1,output_size))
            news.append(sent)
        nlist = chunk(news, nsent)

        for news in nlist:
            if len(news) < ignorelen:
                continue
            if len(news) < nsent:
                padding = np.zeros((1,5, output_size))
                news.extend([padding]*(nsent-len(news)))
            newslist.append(news)
    return headline, newslist


if __name__ == '__main__':
    make_vectors_from_news()
    # a=set()

    # for key, val in vec.items():
        # a.add(len(val))
    # print(a)
