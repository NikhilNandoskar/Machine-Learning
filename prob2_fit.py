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
trial_name = 'p1_fit' # will add a unique sub-string to output of this program
degree = 15 # p, order of model
beta = 0.01 # regularization coefficient
alpha = 1.5 # step size coefficient
eps = 0.00001 # controls convergence criterion
n_epoch = 50 # number of epochs (full passes through the dataset)

# begin simulation

def regress(X, theta):
	return (theta[0] + tf.matmul(X,theta[1], transpose_b = True))

def gaussian_log_likelihood(mu, y, theta):
	return (regress(mu, theta) - y)
	
def computeCost(X, y, theta, beta): # loss is now Bernoulli cross-entropy/log likelihood
    m = tf.cast(tf.shape(X), tf.float64)
    beta = tf.cast(beta, dtype = tf.float64)
    return (tf.divide(tf.add(tf.reduce_sum(tf.square(gaussian_log_likelihood(X, y, theta))),tf.multiply(beta,tf.reduce_sum(tf.square(theta[1])))), 2*m[0]))
    
	

    
def computeGrad(X, y, theta, beta):
    # derivative w.r.t. to model output units (fy)
    m = tf.cast(tf.shape(X), tf.float64)
    beta = tf.cast(beta, dtype = tf.float64)
    dL_db = tf.divide(tf.reduce_sum(gaussian_log_likelihood(X, y, theta)), m[0]) # derivative w.r.t. model weights w
    dL_dw = tf.divide(tf.add(tf.matmul(gaussian_log_likelihood(X, y, theta), X, transpose_a = True),tf.multiply(beta, theta[1])), m[0]) # derivative w.r.t. model weights b
    nabla = (dL_db, dL_dw) # nabla represents the full gradient
    return nabla

path = os.getcwd() + '/data/prob2.dat'  
data = pd.read_csv(path, header=None, names=['X', 'Y']) 

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

# create an placeholder variable for Y(output)
y_p = tf.placeholder(tf.float64, shape = (X_t.shape[0], 1))

# apply feature map to input features x1
X_feat = []
for i in range(1,degree+1):
    X_feat.append(np.power(X, i))
X_feat_new = np.concatenate((X_feat), axis = 1)
X_feat_new_t = tf.convert_to_tensor(X_feat_new, dtype = tf.float64)
X_feat_new_p = tf.placeholder(tf.float64, shape = (X_feat_new_t.shape[0], degree))

# convert to numpy arrays and initalize the parameter array theta
w = np.zeros((1,X_feat_new.shape[1]))
b = np.array([0])
theta = (b, w) 

#Converting w and b to tensors
w_t = tf.Variable(w, dtype = tf.float64, name = "w")
b_t = tf.Variable(b, dtype = tf.float64, name = "b")

init = tf.global_variables_initializer()

L = computeCost(X_feat_new_t, y_t, theta, beta)

cost = []
with tf.Session()as sess:
    sess.run(init)
    print("-1 L = {0}".format(L.eval()))
    cost.append(L.eval())
    
i = 0
with tf.Session() as sess:
    sess.run(init)
    while(i < n_epoch):
        dL_db, dL_dw = computeGrad(X_feat_new, y_t, theta, beta)
        o,k = sess.run([dL_db, dL_dw], feed_dict = {X_feat_new_p:X_feat_new, y_p:y})
        b = theta[0]
        w = theta[1]
        # update rules go here...
        b = b - alpha*o
        w = w - alpha*k
        theta = (b, w)
        
        L = computeCost(X_feat_new_t, y_t, theta, beta)
         
        if (cost[- 1] - L.eval()) < eps:
          break
        cost.append(L.eval())
        print(" {0} L = {1}".format(i,L.eval()))
        i += 1
	# print parameter values found after the search
    print("w = ",w)
    print("b = ",b)
    saver = tf.train.Saver([w_t, b_t])
    saver.save(sess, os.getcwd() + '/Pb_2_Saved/Model', global_step = i )
    
#Save everything into saver object in tensorflow
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.getcwd() + '/Pb_2_Tensorboard', sess.graph)
    
#Visualize using tensorboard
kludge = 0.25
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_feat_new = np.expand_dims(X_test, axis=1) # we need this otherwise, the dimension is missing (turns shape(value,) to shape(value,value))
X_feat = []
for i in range(1,degree+1):
    X_feat.append(np.power(X_feat_new, i))
X_feat = np.concatenate((X_feat), axis = 1)

# apply feature map to input features x1
with tf.Session() as sess:
    plt.figure(1)
    plt.plot(X_test, sess.run(regress(X_feat, theta)), label="Model")
    plt.scatter(X[:,0], y, edgecolor='g', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
    plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
    plt.legend(loc="best")
    plt.savefig(os.getcwd() + '/Pb_2_15_Test_Fig.jpg')
  
    plt.figure(2)    
    plt.plot(cost)
    plt.title('Loss vs Epoch')
    plt.xlabel('Number of epochs')
    plt.ylabel('Cost')
    plt.savefig(os.getcwd() + '/Pb_2_15_Loss_Fig.jpg')

plt.show()
