import os  
import tensorflow as tf
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt  

'''
IST 597: Foundations of Deep Learning
Problem 3: Multivariate Regression & Classification

Adapted from Alexander G. Ororbia II and Ankur Mali
@author - Nikhil Sunil Nandoskar

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

# meta-parameters for program
alpha = 0.02 # step size/ learning rate coefficient
eps = 0.00001 # controls convergence criterion
n_epoch = 5000 # number of epochs (full passes through the dataset)

# begin simulation
def regress(X, theta):
	return (theta[0] + tf.multiply(X,theta[1]))

def gaussian_log_likelihood(mu, y, theta):
	return (regress(mu, theta) - y)
	
def computeCost(X, y, theta): # loss is now Bernoulli cross-entropy/log likelihood
    m = tf.cast(tf.shape(X), tf.float64)
    return (tf.divide(tf.reduce_sum(tf.square(gaussian_log_likelihood(X, y, theta))), 2*m[0]))
	
def computeGrad(X, y, theta):
    m = tf.cast(tf.shape(X), tf.float64)
    dL_db = tf.divide(tf.reduce_sum(gaussian_log_likelihood(X, y, theta)), m[0]) # derivative w.r.t. model weights w
    dL_dw = tf.divide(tf.matmul(gaussian_log_likelihood(X, y, theta), X, transpose_a = True), m[0]) # derivative w.r.t model bias b
    nabla = (dL_db, dL_dw) # nabla represents the full gradient
    return nabla

path = os.getcwd() + '/data/prob1.dat'  
data = pd.read_csv(path, header=None, names=['X', 'Y']) 

# display some information about the dataset itself here
print(data.shape)

# WRITEME: write your code here to print out information/statistics about the data-set "data" using Pandas (consult the Pandas documentation to learn how)
print(data.describe())

# WRITEME: write your code here to create a simple scatterplot of the dataset itself and print/save to disk the result
plt.scatter(data.iloc[:,0], data.iloc[:,1], c= 'red', marker = "+")
plt.savefig(os.getcwd() + '/Pb_1_Scattered_plot')

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)

# convert np array to tensor objects
X_t = tf.convert_to_tensor(X, dtype = tf.float64)
y_t = tf.convert_to_tensor(y, dtype = tf.float64)

# create an placeholder variable for X(input) and Y(output)
X_p = tf.placeholder(tf.float64, shape = (X_t.shape[0], 1))
y_p = tf.placeholder(tf.float64, shape = (y_t.shape[0], 1))

# convert to numpy arrays and initalize the parameter array theta 
w = np.zeros((1,X.shape[1]))
b = np.array([0])
theta = (b, w)

#Converting w and b to tensors
w_t = tf.Variable(w, dtype = tf.float64, name = "w")
b_t = tf.Variable(b, dtype = tf.float64, name = "b")

init = tf.global_variables_initializer()

    
    
L = computeCost(X_t, y_t, theta)
L_best = L
cost = []
with tf.Session() as sess:
    sess.run(init)
    print("-1 L = {0}".format(L.eval()))
    cost.append(L.eval())

#saver = tf.train.Saver(sess)
i = 0
with tf.Session() as sess:
    sess.run(init)
    
    while(i < n_epoch):
        dL_db, dL_dw = computeGrad(X_t, y_t, theta)
        b_u,w_u = sess.run([dL_db, dL_dw], feed_dict = {X_p: X, y_p: y})
        
        b = theta[0]
        w = theta[1]
        
        b = b - alpha*b_u
        w = w - alpha*w_u
        theta = (b,w)        
        
        L = computeCost(X_t, y_t, theta)
        if (cost[-1] - L.eval()) < eps:
            break
        cost.append(L.eval())
        print('{0} L = {1}'.format(i,L.eval()))
        i += 1
    print('W:', w)
    print('b:', b)
    saver = tf.train.Saver([w_t, b_t])
    saver.save(sess, os.getcwd() + '/Pb_1_Saved/Model', global_step = i )
    
    
#Save everything into saver object in tensorflow 
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.getcwd() + '/Pb_1_2_Tensorboard', sess.graph)
# Plotting 
kludge = 0.25 # helps with printing the plots (you can tweak this value if you like)
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_test = np.expand_dims(X_test, axis=1)


with tf.Session() as sess:
    plt.figure(1)
    plt.plot(X_test, sess.run(regress(X_test, theta)), label="Model")
    plt.scatter(X[:,0], y, edgecolor='g', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
    plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
    plt.legend(loc="best")
    plt.savefig(os.getcwd() + '/Pb_1_2_Test_Fig.jpg')
    
# visualize the loss as a function of passes through the dataset
 
    plt.figure(2)    
    plt.plot(cost)
    plt.title('Loss vs Number of Epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Cost')
    plt.savefig(os.getcwd() + '/Pb_1_2_Loss_Fig.jpg')
    
    
        
        
    plt.show() # convenience command to force plots to pop up on desktop