import pandas as pd
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.cluster import KMeans
glove_input_file = 'archive/glove.6B.100d.txt'

tags_df=pd.read_csv('archive/tags.csv')
tags=tags_df['tag'].str.lower().unique()
model = KeyedVectors.load_word2vec_format(glove_input_file, binary=False,no_header=True)
def get_vector(word):
    try:
        return model[word]
    except KeyError:
        return None
tag_vectors = [get_vector(tag) for tag in tags]
tag_vectors = [vec for vec in tag_vectors if vec is not None]  # 移除无法转换的标签

n_clusters = 100  # 选择适当的簇数
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(tag_vectors)
clusters = kmeans.labels_

tag_cluster_pairs = list(zip(tags, clusters))
tag_cluster_df = pd.DataFrame(tag_cluster_pairs, columns=['Tag', 'Cluster'])
tag_cluster_df.to_csv('archive/tag_cluster.csv', index=False)
print(tag_cluster_pairs[:10])

cluster_df = pd.read_csv('archive/tag_cluster.csv')

for i in range(100):
    cluster_tags = cluster_df[cluster_df['Cluster'] == i]['Tag'].tolist()
    print(cluster_tags)