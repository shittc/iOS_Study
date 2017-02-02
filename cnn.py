import tensorflow as tf
import numpy as np
import pandas
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

def conv2D(X, kernel,stride = 1):
    conv = tf.nn.conv2d(X, kernel, strides=[1,stride,stride,1] )
    return conv

def dropout(X, prob):
    dropout = tf.nn.dropout(X, prob )
    return dropout

def pooling(X, prob):
    dropout = tf.nn.dropout(X, prob )
    tf.nn.pool(X,)
    return dropout


def CreateNet(X):
    W1 = tf.Variable(tf.truncated_normal([5, 5, 3, 32]), name = "W1")
    b1 = tf.Variable(tf.zeros([32]), name = "b1")
    conv1 = conv2D(X, W1)
    activation1  = tf.nn.relu(tf.matmul(conv1, W1) +b1)
    prob = tf.Variable(0.8)
    cnn1 = dropout(activation1, prob)

label = [1,3,5]
#labels = tf.sparse_tensor_to_dense(label)
#one_hot_a = tf.nn.embedding_lookup(np.identity(3), label)

all_species = ["Setosa", "Versicolor", "Virginica"]
#onehot_labels = tf.one_hot(indices=tf.cast(all_species, tf.int32), depth=3)
#tf.one_hot(all_species, 3, 1, 0)
sess = tf.Session()
str = tf.convert_to_tensor(all_species)
one =tf.string_to_number(str, out_type=tf.float32)
print(sess.run(str))
#enc = OneHotEncoder()
#enc.fit(all_species)
labels_oneHot = pandas.get_dummies(all_species)
print(labels_oneHot.shape)
#print(labels_oneHot[0])

#print(labels_oneHot.head())
onehot = {}
# Target number of species types (target classes) is 3 ^
species_count = len(all_species)

# Print out each one-hot encoded string for 3 species.
for i, species in enumerate(all_species):
    # %0*d gives us the second parameter's number of spaces as padding.
    print("%s,%0*d" % (species, species_count, 10 ** i))
