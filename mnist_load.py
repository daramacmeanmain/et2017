"""
Created on Wed Feb 15 2017
@author: Itzik Ben Shabat www.itzikbs.com
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
# Get the Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

from PIL import Image
import PIL.ImageOps
im = Image.open("test_image2.png")
img = im.resize((28, 28))
grey = img.convert("L")
greyInvert = PIL.ImageOps.invert(grey)
arr = np.array(greyInvert)

#===============================================================================
# Restore trained Graph
#===============================================================================
SummariesDirectory = "C:/Users/Dara/Desktop/emergingTech/project/etProject2017/mnist_checkpoints/"
ModelName = "mnist-2900"
sess = tf.Session()
new_saver = tf.train.import_meta_graph(SummariesDirectory + ModelName + ".meta")
new_saver.restore(sess, SummariesDirectory + ModelName)

#all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
y_conv = tf.get_collection("y_conv")[0]
x = tf.get_collection("x")[0]
y_ = tf.get_collection("y_")[0]
keep_prob =  tf.get_collection("keep_prob")[0]

InputImage = arr.reshape(1,784)

logit = sess.run(y_conv,feed_dict={ x: InputImage, keep_prob: 1.0})
prediction = sess.run(tf.argmax(logit,1))
print("Prediction : %d"% (prediction))

plt.imshow(InputImage.reshape(28,28),cmap = 'gray')
plt.show()
#"predicted: %d, Actual:%d", prediction,