from lib import GenLib
import zlib
import pickle

######## Collection of dataset
tags = set()
# load dataset and model
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
print("Number of documents: ",len(data_docs))
print(data_docs[0])

# Step 1 : Initialize model
# name = nrega, dataset = data_docs, tags = tags
# entities = states, usePreTrained - True
gl = GenLib('nrega', data_docs, tags, states, True)

# step 2 : Train model
# training of model can be skipped if model is already saved
gl.train_d2v()

### load if model is trained and already saved
# Uncomment if needed
'''
gl.load_model()
'''

# Step 3 : Perform Clustering
gl.clustering()

### load if clustering is already performed and saved
# Uncomment if needed
'''
gl.load_centroids()
'''

# Step 4 : Utility methods
# get vector for a document (by index in data_docs
print(gl.get_vector(0))

# get representative vector for a tag
print((gl.get_centroid_for_tag('1.0')))

# predict subtopic (tag) for a given text
print(gl.pred_tag('State clears pending wages to nrega workers'))

# get top K documents related to a tag
print(gl.get_top_k('2.0',10))
