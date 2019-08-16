import tensorflow as tf
#from PIL import Image
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from IPython.display import Image
from matplotlib.image import imread
import matplotlib.pyplot as plt
from skimage import data, io, filters
from skimage.color import rgb2gray
import warnings
from PIL import Image
import sys

warnings.filterwarnings(action="ignore")
# img = imread("http://localhost:9090/mini/image/33.png")
# img1 = imread("C:/miniproject/.metadata/.plugins/org.eclipse.wst.server.core/tmp0/wtpwebapps/StudyProject_ver1/uploads/6.png")
img = Image.open(sys.argv[1])
# img = Image.open("C:/python_ML/data/number/11.png")
img = img.rotate(-90)


img_test = img.resize((28,28))
img = np.array(img_test)


# img = sess.run(tf.reshape(img,[28,28]))
# size = (28,28)
# img.thumbnail(size)  

# img = color.rgb2gray(img)
img= rgb2gray(img)
img = 1-img

# print("20")
# plt.imshow(img, cmap = "gray")

sess = tf.InteractiveSession()
img = sess.run(tf.reshape(img,[1,784]))

mnist = input_data.read_data_sets('./data/mnist', one_hot=True)

new_saver = tf.train.import_meta_graph('C:/python_ML/momodel/train_model.ckpt.meta')
new_saver.restore(sess,  'C:/python_ML/momodel/train_model.ckpt')

# print("21")
# tf.all_variables()
modelNum =2
result = np.zeros([1, 10])
# print("22")
for num in range(modelNum):
    modelName = "model"+str(num)
    X = sess.graph.get_tensor_by_name(modelName+"/Placeholder_1:0")
    logits = sess.graph.get_tensor_by_name(modelName+"/dense_1/BiasAdd:0")
    training = sess.graph.get_tensor_by_name(modelName+"/Placeholder:0")
    result += sess.run(logits, feed_dict={X:img, training:False})
    
print("MNIST predicted Number : ", np.argmax(result))
print(np.argmax(result))

