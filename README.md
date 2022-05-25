# genlib
A generic tool to extract most relevant documents related to a subtopic in labelled documents using gensim doc2vec with tag based embeddings


The main purpose of the tool is to provide a generic document ranking framework that can be used to obtain most relevant documents pertaining to a tag and predict the closest tag for a new document. This can be used with any text based data and requires a list of document text along with tags associated with each document. To use the library, follow these steps : 

## Step 0 : Setup.

The library achieves slightly better accuracy with less number of training epochs if pre-trained word vectors are used. Using pre-trained word vectors is optional and depends on the user. However, if it is intended to use pre-trained word vectors, then download and unzip these vectors from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g). The file size is big (around 1.5 GB). The unzipped file and the library file must be present at the same directory level. The library file is available [here](https://drive.google.com/file/d/1Ffi17vd3t4r-nJgzCk5aUjthZsgi_xwh/view?usp=sharing). 

#### Compatibility : 
```
  Python versions : 3.5, 3.6, 3.7, 3.8
  Gensim version : 3.6.0
  OS (Tested on) : Windows, Linux (Linux Recommended for fast training due to easy availability of Cython extensions for gensim) 
```

#### Sample dataset for collection and formatting
```

from lib import GenLib
import zlib
import pickle

######## Collection of dataset
tags = set()
# load dataset and model
# the dataset is a collection of documents related to MNREGA scheme
# with different clusters
d = open('data_sample','rb')
dataset = pickle.loads(zlib.decompress(pickle.load(d)))
d.close()

print("Data Loaded")

#### Create dataset
# data_docs is a list of tuples
# where each tuple has form (text, list of tags)
# this list of documents is passed to the library's object 
data_docs = []
states = set()
for i in dataset:
    # text, tags : id, cluster
    data_docs.append((i[2], [str(i[0]), str(i[-1])]))
    states.add(i[5])
    tags.add(str(i[-1]))

states = list(states)
tags = list(tags)
```

A sample file on how to use the library is also available as [test.py](https://github.com/aadish233/genlib/blob/main/test.py)

## Step 1 : Initialize library 
To initialize, create an object of class “GenLib” from lib.py. The arguments passed to the constructor are : 
> 1. name : A string denoting the name related to the documents. This will be used to save models and centroids obtained later.
> 2. docs : A list of tuples  where each tuple is of the form (text, list of tags) for each document. The list of tags can also include a unique identifier of 		the document which can then be used to obtain the trained vector for a document. If no unique identifier is passed, the index of the document in the docs 	     list can be used to obtain the vector.
> 3. tags : List of all tags which are used to tag the documents (apart from the unique identifier). For instance, the nine subclasses in case of development 	       dataset, OR cluster number in case of nrega dataset etc.
> 4. entities :  List of string entities. These will be removed from the text of documents (entity blinding). If no entities must be ignored, pass an empty list
> 5. usePretrained : Boolean value. If True, the pretrained vectors will be used to initialize the model. Otherwise, the pretrained vectors will be ignored and 		 the model will be trained for a larger number of training epochs to ensure similar level of accuracy


#### Sample Code for initialization using data_sample
```
# Step 1 : Initialize model
# name = nrega, dataset = data_docs, tags = tags
# entities = states, usePreTrained - True
gl = GenLib('nrega', data_docs, tags, states, True)
```

## Step 2 : Model Training 
The model will be trained based on the empirical evaluation of hyperparameters for DocTag2Vec. This step is computationally expensive and would take several hours to execute depending on size of the dataset. 

Initially, each document’s text is pre-processed using a helper function which removes stopwords, performs stemming and entity blinding. Then the model is trained on these pre-processed documents with the following parameters which are empirically found to produce good embeddings : 
>1. Vector Size : Vectors of dimension 300 are used as the vectors used in the pre-trained dataset are also of dimensions 300
>2. Context Window Size : 5 
>3. MinCount :  3 (Minimum frequency of a word for it to be considered for training)
>4. Algorithm : Distributed Bag of Words (DBOW) with the option to train word vectors in skip gram fashion simultaneously with document training
>5. Number of Training Epochs : If pre-trained word vectors are used, less number of iterations of training are needed (120). Otherwise if they are not used, 200 training epochs are used.


To train the model, use the “train_d2v()” method of the GenLib class. The method will train and save the model as “model_name” in the same directory. This saved model can be reloaded to avoid training again and again using the “load_model()” method of the GenLib class after initializing as done in step 1. 

#### Methods : 
```
train_d2v() : 
	Arguments => None
	Use => Trains and saves the model
load_model() : 
	Arguments => None
	Use => To load an already trained and saved model. This avoids the need for re-training.
```

#### Sample code for model training and loading (after initialization)
```
# step 2 : Train model
# training of model can be skipped if model is already saved
# this method will save a trained gensim doc2vec model by the name 'model_nrega'
gl.train_d2v()

### load if model is trained and already saved
# Uncomment if needed
'''
gl.load_model()
'''
```

## Step 3 : Clustering 
This step performs hierarchical agglomerative clustering for document vectors of each tag. The clustering is performed based on the method described here. To perform clustering after training of the model, use the “clustering()” method of the GenLib class. The clustering uses the ward method with the euclidean distance metric. It also performs outlier elimination as given in the paper. The end result of this step is a representative vector for each tag which will be saved in the “centroids” folder. Once the representative vectors are saved, clustering need not be performed again. The centroids can be reloaded using the “load_centroids()” method of the GenLib class.

#### Methods : 
```
clustering() : 
	Arguments => None
	Use => Performs clustering and saves the representative vectors for each tag
load_centroids() : 
	Arguments => None
	Use => To load representative vectors. This avoids the need for re-clustering.
```

#### Sample code to perform clustering and save representative documents
```
# Step 3 : Perform Clustering
# will save representative vectors for each tag
gl.clustering()

### load if clustering is already performed and saved
# After steps 1 and 2
# Uncomment if needed
'''
gl.load_centroids()
'''
```

## Step 4 : Utilities 
- The trained vector for a document can be obtained using the “get_vector()” method of the GenLib class. The unique identifier of the document (or the index of document in docs list) is passed as argument. It will return a 300-element vector for a document.

  #### Method : 

  ```
  '''
  get_vector(doc_id) : 
    doc_id => Either the index of document in docs list or the unique identifier for the document
    Use => Returns the learned vector of the document obtained during training.
  '''
  
  # Sample use
  # Will print a 300-dimension vector for the first document in the dataset
  # learned during training
  print(gl.get_vector(0))
  ```

- The representative vector for a tag can be obtained using the “get_centroid_for_tag()” method of the GenLib class. The tag is passed as argument to the method and the representative vector of the tag is returned.

  #### Method : 
  ```
  '''
  get_centroid_for_tag(tag) : 
    tag => Tag name for which the centroid must be obtained.
    Use => Returns the representative vector of the tag.
  '''
  # Will print a 300-dimension vector representing the tag
  print((gl.get_centroid_for_tag('1.0')))
  ```

- For any tag, top K documents which are most similar to the centroid vector of the tag and dissimilar to the centroid vectors of other tags are returned. The metric used for this purpose is :
  ```
  M = C * D 
  where C = cosine similarity of document’s vector with the centroid vector of the tag 
        D = (1 - average cosine similarity of document’s vector with the centroid vector of other tags). 
  ```
  The method “get_top_k()” of the GenLib class can be used to obtain a list of top K documents’ list where tag and value of K are passed as arguments. Each element of the list consists of a tuple of two elements where first element is the metric value and second value is the index of the document in original docs list. For this purpose, the value of metric is computed for each tag’s documents. Then documents with highest values of the metric are returned.

  #### Method : 
  ```
  '''
  get_top_k(tag, K) : 
    tag => Name of the tag  
    K => Number of documents
    Use => Returns top K relevant documents for the tag
  '''
  
  # will print a list of 10 documents with highest M values
  # (most relevant) to tag 2.0
  print(gl.get_top_k('2.0',10))
  ```
- For a new document text, the method “pred_tag()” of the GenLib class can be used which takes the text as argument.This method preprocesses the text of the document and later infers the vector of the document. It returns a tuple which has the first element as the predicted tag and second element as the highest metric value(M) corresponding to the predicted tag for the document. 
  #### Method : 
  ```
  '''
  pred_tag(text) : 
    text => Content of the document
    Use => Returns the most suitable tag for the document.
  '''
  # will print a tuple with first value as the closest tag to which the text most likely belongs to
  # and second value is the degree of relevance (on scale of 0-1) Metric M value
  print(gl.pred_tag('State clears pending wages to nrega workers'))
  ```
