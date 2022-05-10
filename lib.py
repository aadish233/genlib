import gensim
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from random import sample
import scipy.cluster.hierarchy as sch
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import single, fcluster
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import single, cophenet
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestCentroid
from collections import Counter
import sys
import math
import random
random.seed(2000)
sys.setrecursionlimit(5000)

# main class to execute
# the pipeline
class GenLib:
    # constructor
    '''
    name : string (trained model will be saved as model_name)
    docs : A list of tuples
           where each tuple is of the form
           (text, list of tags)
    tags : List of all tags
    entities : A list of entities which will be ignored from text
    usePretrained : bool, if true will use google news vectors as
                    starting point during training
    '''
    def __init__(self, name, docs, tags, entities=[], usePretrained=False):
        self.name = name
        self.entities = entities
        self.tags = tags
        self.docs = docs
        self.pretrained = usePretrained

    # helper function to preprocess text
    # function to remove stopwords, perform stemming and entity blinding.
    def filter_text(self, text, entities):
        stop_words = set(stopwords.words('english'))
        entities = set(entities)
        ps = PorterStemmer()
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = re.sub("\S*\d\S*", "", text)
        text = re.sub('[^A-Za-z0-9]+', ' ', text)
        text = text.lower()
        text = word_tokenize(text)
        text = [w for w in text if not w in stop_words|entities]
        text = [ps.stem(w) for w in text]
        return text

    # function to train model
    def train_d2v(self):
        # taggeddocuments used for training
        self.tdocs = [TaggedDocument(self.filter_text(self.docs[i][0], self.entities),[i] + self.docs[i][1]) for i in range(len(self.docs))]
        print('Documents collected for training, number of documents:', len(self.tdocs))
        model = Doc2Vec(dm=0,vector_size=300,dbow_words=1,window=5,min_count=3,alpha=0.1,min_alpha=0.001)
        print('Model Initialized')
        model.build_vocab(self.tdocs)
        print('Vocabulary size: ', len(model.wv.vocab.keys()))
        # update vectors from pre-trained model
        if self.pretrained:
            # load
            print('Loading google news word vectors...')
            pre_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
            print('Pre-trained vectors loaded')
            print('Updating vectors...')
            for k in model.wv.vocab:
                if k in pre_model:
                    model.wv[k] = pre_model[k]
            print('Vectors updated')
            print('Model training...')
            model.train(self.tdocs,total_examples=len(self.tdocs),epochs=120)
        else:
            print('Model training...')
            model.train(self.tdocs,total_examples=len(self.tdocs),epochs=200)
        print('Model trained')
        self.d2vmodel = model
        print('Saving model...')
        file = open('model_'+self.name,'wb')
        model.save(file)
        file.close()
        print('Model Saved as model_' + self.name)

    def load_model(self):
        self.d2vmodel = Doc2Vec.load('model_'+self.name)

    #Downsampling
    def sampling(self,data):
        limit = 10000
        ind = []
        if len(data)>limit:
            ind = sample([i for i in range(len(data))],limit)
            data = [data[i] for i in ind]
        else:
            ind = [i for i in range(len(data))]
        return data,ind

    #Eliminate outlier clusters
    def eliminate_outlier(self,data,linkage_matrix,method,metric,thresh=50):
        labels = sch.fcluster(linkage_matrix, thresh, criterion='distance')
        d = Counter(labels)
        sorted_d = sorted(d.items(), key = lambda kv:(kv[1], kv[0])) 
        maxx = None
        art_thresh = None
        itm = list(d.values())
        for art in range(1,101):
            if art not in itm:
                continue
            clusters = []
            for i,j in sorted_d:
                if j<art:
                    clusters.append(i)
            indices = []
            for c in clusters:
                indices = indices+[i for i, x in enumerate(labels) if x == c]
            revised_data = [data[i] for i in range(len(data)) if i not in indices]
            #print(len(revised_data),len(indices),len(data1))
            linkage_matrix = sch.linkage(revised_data, method, metric=metric)
            c, coph_dists = cophenet(linkage_matrix, pdist(revised_data))
        
            if maxx is None or c>=maxx:
                maxx = c
                art_thresh = art
            # print(art_thresh,c,maxx)
        if art_thresh is None:
            art = itm[0]
            clusters = []
            for i,j in sorted_d:
                if j<art:
                    clusters.append(i)
            indices = []
            for c in clusters:
                indices = indices+[i for i, x in enumerate(labels) if x == c]
            revised_data = [data[i] for i in range(len(data)) if i not in indices]
            #print(len(revised_data),len(indices),len(data1))
            linkage_matrix = sch.linkage(revised_data, method, metric=metric)
            c, coph_dists = cophenet(linkage_matrix, pdist(revised_data))
        
            if maxx is None or c>=maxx:
                maxx = c
                art_thresh = art
            # print(art_thresh,c,maxx)
        return art_thresh

    #Compute rank
    def compute_scr(self,linkage_matrix,data,art_thresh,thresh=50):
        labels = sch.fcluster(linkage_matrix, thresh, criterion='distance')
        d = Counter(labels)
        sorted_d = sorted(d.items(), key = lambda kv:(kv[1], kv[0])) 
        clusters = []
        for i,j in sorted_d:
            if j<=art_thresh:
                clusters.append(i)
        indices = []
        for c in clusters:
            indices = indices+[i for i, x in enumerate(labels) if x == c]
        ind = [i for i in range(len(data)) if i not in indices]
        if ind==[]:
            return self.compute_scr(linkage_matrix,data,self.eliminate_outlier(data,linkage_matrix,'ward','euclidean',thresh//2),thresh//2)
        revised_data = [data[i] for i in ind]
        labels=list(labels)
        labels = [labels[i] for i in ind]
        #print(len(ind),len(revised_data),len(data))
        clf = NearestCentroid()
        clf.fit(revised_data, labels) 
        centroids = clf.centroids_
        num1 = [i for i in range(len(set(labels)))]
        num2 = sorted(set(labels))
        for i in range(len(num1)):
            labels=[num1[i] if x==num2[i] else x for x in labels]
        ranks = []
        for i in range(len(labels)):
            ranks.append((cosine_similarity(centroids[labels[i]].reshape(1,-1),revised_data[i].reshape(1,-1))[0][0]+1)/2) #cos_sim(row1, row2)- minx)/(maxx-minx)
        return revised_data,labels,centroids,ranks,ind

    #driver function
    def driver_func(self,data,thresh=100):
        data,indx = self.sampling(data)
        method = 'ward'
        metric = 'euclidean'
        linkage_matrix = sch.linkage(data, method, metric=metric)
        art_thresh = self.eliminate_outlier(data,linkage_matrix,method,metric,thresh)
        return indx,self.compute_scr(linkage_matrix,data,art_thresh,thresh)

    #global centroid
    def global_centroid(self,data):
        return np.median(np.array(data),axis=0)

    def save_cent(self,data, tag_name):
        indx,out= self.driver_func(data)
        (revised_data,labels,centroids,rankA,indA) = out
        gc = self.global_centroid(revised_data)
        # print(gc)
        np.save('centroids/'+tag_name+ '_' + str(self.name) + '.npy', gc)
        return gc
    
    # function to find centroids of each tag
    def clustering(self):
        import os
        if not os.path.isdir('centroids'):
            os.mkdir('centroids')
        # store indices of articles for each tag
        tag_arts = dict()
        tag_indices = dict()
        for tag in self.tags:
            tag_arts[tag] = []
            tag_indices[tag] = []
        
        for i in range(len(self.docs)):
            for tag in self.tags:
                if tag in self.docs[i][1]:
                    try:
                        tag_arts[tag].append(self.d2vmodel.docvecs[i])
                        tag_indices[tag].append(i)
                    except:
                        pass
        for tag in tag_arts:
            if tag_arts[tag]==[]:
                del tag_arts[tag]
                del tag_indices[tag]
                
        self.tag_s = tag_arts
        self.tag_inds = tag_indices
        centrs = dict()
        for tag in tag_arts:
            g_c = self.save_cent(tag_arts[tag], tag)
            centrs[tag] = g_c
        self.tag_centroids = centrs

    # function to load saved centroids
    def load_centroids(self):
        tag_arts = dict()
        tag_indices = dict()
        for tag in self.tags:
            tag_arts[tag] = []
            tag_indices[tag] = []
        
        for i in range(len(self.docs)):
            for tag in self.tags:
                if tag in self.docs[i][1]:
                    try:
                        tag_arts[tag].append(self.d2vmodel.docvecs[i])
                        tag_indices[tag].append(i)
                    except:
                        pass
        for tag in tag_arts:
            if tag_arts[tag]==[]:
                del tag_arts[tag]
                del tag_indices[tag]
                
        self.tag_s = tag_arts
        self.tag_inds = tag_indices
        centrs = dict()
        for tag in self.tag_s:
            centrs[tag] = np.load('centroids/'+tag+ '_' + str(self.name) + '.npy')
        self.tag_centroids = centrs
        

    def get_vector(self, doc_id):
        return self.d2vmodel.docvecs[doc_id]

    def get_centroid_for_tag(self, tag):
        return self.tag_centroids[tag]

    def comp_similarity(self, v1,v2):
        return (cosine_similarity(v1.reshape(1,-1),v2.reshape(1,-1))[0][0]+1)/2
        
    def compute_score(self, tag, vec):
        avg_cossim = 0
        gc = self.tag_centroids[tag]
        score = self.comp_similarity(gc, vec)
        n = 0
        for t in self.tag_centroids:
            if tag != t:
                avg_cossim += self.comp_similarity(self.tag_centroids[t],vec)
                n += 1
        avg_cossim /= n
        score *= (1-avg_cossim)
        return score
    
        
    # function to get top K tags corresponding to a
    # tag
    def get_top_k(self, tag, K):
        indices = self.tag_inds[tag]
        vecs = self.tag_s[tag]
        tmp = [(self.compute_score(tag,v),i) for (i,v) in zip(indices,vecs)]
        tmp.sort(reverse=True)
        return tmp[:K]

    # to predict tag for a new document
    def pred_tag(self, text):
        filtered_text = self.filter_text(text, self.entities)
        v_pred = self.d2vmodel.infer_vector(filtered_text)
        tmp = [(self.compute_score(tag,v_pred),tag) for tag in self.tag_centroids]
        x,y = max(tmp)
        return y,x
    
