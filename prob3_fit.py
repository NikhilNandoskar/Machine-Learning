import os  
import tensorflow as tf 
import pandas as pd 
import numpy as np 
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
trial_name = 'p6_reg0' # will add a unique sub-string to output of this program
degree = 6 # p, degree of model
beta = 1 # regularization coefficient
alpha = 1.5 # step size coefficient
n_epoch = 50 # number of epochs (full passes through the dataset)
eps = 0.00001 # controls convergence criterion


# begin simulation
def sigmoid(z):
	return (tf.divide(1,tf.add(tf.cast(1, dtype = tf.float64), tf.exp(-z))))

def predict(X, theta):  
    n = sigmoid(theta[0] + tf.matmul(X,theta[1], transpose_b = True))
    u = tf.round(n)
    return (u)
	
def regress(X, theta):
	return (sigmoid(theta[0] + tf.matmul(X,theta[1], transpose_b = True)))

def bernoulli_log_likelihood(p, y, theta):
	return (tf.add(tf.multiply(y, tf.log(regress(p, theta))),tf.multiply(tf.subtract(tf.cast(1, dtype = tf.float64), y),tf.log(tf.subtract(tf.cast(1, dtype = tf.float64), regress(p, theta))))))
	
def computeCost(X, y, theta, beta): # loss is now Bernoulli cross-entropy/log likelihood
    m = tf.cast(tf.shape(X), tf.float64)
    beta = tf.cast(beta, dtype = tf.float64)
    return (tf.add(tf.divide(tf.reduce_sum(-bernoulli_log_likelihood(X, y, theta)), m[0]), tf.divide(tf.multiply(beta,tf.reduce_sum(tf.square(theta[1]))), 2*m[0]) ))
	
def computeGrad(X, y, theta, beta): 
	
    m = tf.cast(tf.shape(X), tf.float64)
    beta = tf.cast(beta, dtype = tf.float64)
    dL_dfy = None # derivative w.r.t. to model output units (fy)
    dL_db = tf.divide(tf.reduce_sum(tf.subtract(regress(X, theta), y)), m[0]) # derivative w.r.t. model weights w
    dL_dw = tf.divide(tf.add(tf.matmul(tf.subtract(regress(X, theta), y),X, transpose_a = True), tf.multiply(beta, theta[1])), m[0]) # derivative w.r.t model bias b
    nabla = (dL_db, dL_dw) # nabla represents the full gradient
    return nabla
	
path = os.getcwd() + '/data/prob3.dat'  
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]

#Convert positive and negative samples into tf.Variable 
positive_tf_var = tf.Variable(positive)
negative_tf_var = tf.Variable(negative)

x1 = data2['Test 1']  
x2 = data2['Test 2']

#Convert x1 and x2 to tensorflow variables
x1_tf_var = tf.Variable(x1)
x2_tf_var = tf.Variable(x2)

# apply feature map to input features x1 and x2
cnt = 0
for i in range(1, degree+1):  
	for j in range(0, i+1):
		data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
		cnt += 1

data2.drop('Test 1', axis=1, inplace=True)  
data2.drop('Test 2', axis=1, inplace=True)

# set X and y
cols = data2.shape[1]  
X2 = data2.iloc[:,1:cols]  
y2 = data2.iloc[:,0:1]

# make results reproducible
seed = 1491189
np.random.seed(seed)
tf.set_random_seed(seed)

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)  
y2 = np.array(y2.values)  
w = np.zeros((1,X2.shape[1]))
b = np.array([0])
theta = (b, w)

#Converting w and b to tensors
w_t = tf.Variable(w, dtype = tf.float64, name = "w")
b_t = tf.Variable(b, dtype = tf.float64, name = "b")

#Convert all numpy variables into tensorflow variables
X2_t = tf.convert_to_tensor(X2, dtype = tf.float64)
y2_t = tf.convert_to_tensor(y2, dtype = tf.float64)

#Creating placeholders for X2 and y2
X2_p = tf.placeholder(dtype = tf.float64, shape = (X2_t.shape[0], 27))
y2_p = tf.placeholder(dtype = tf.float64, shape = (y2_t.shape[0], 1))

init = tf.global_variables_initializer()

L = computeCost(X2_t, y2_t, theta, beta)
cost = []
with tf.Session()as sess:
    sess.run(init)
    print("-1 L = {0}".format(L.eval()))
    cost.append(L.eval())

i = 0
halt = 0
#Initialize graph and all variables
with tf.Session() as sess:
    sess.run(init)
    while(i < n_epoch and halt == 0):
            dL_db, dL_dw = computeGrad(X2_t, y2_t, theta, beta)
            o,k = sess.run([dL_db, dL_dw], feed_dict = {X2_p:X2, y2_p:y2})
            b = theta[0]
            w = theta[1]
		# update rules go here...
            b = b - alpha*o
            w = w - alpha*k
            theta = (b,w)
            L = computeCost(X2_t, y2_t, theta, beta)
            if (cost[- 1] - L.eval()) < eps:
                break
            cost.append(L.eval())
            print(" {0} L = {1}".format(i,L.eval()))
            i += 1
            
# print parameter values found after the search
    print("w = ",w)
    print("b = ",b)
    saver = tf.train.Saver([w_t, b_t])
    saver.save(sess, os.getcwd() + '/Pb_3_Saved/Model', global_step = i )
    
#Save everything into saver object in tensorflow
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.getcwd() + '/Pb_3_Tensorboard', sess.graph)
    
# Predictions
with tf.Session() as sess:
    sess.run(init)
    predictions = predict(X2, theta)
    m = tf.cast(tf.shape(X2), tf.float64)
    err = tf.subtract(tf.cast(1, dtype = tf.float64), tf.divide(tf.add(tf.matmul(tf.transpose(y2_t), predictions), tf.matmul(tf.transpose(tf.subtract(tf.cast(1, dtype = tf.float64), y2_t)), tf.subtract(tf.cast(1, dtype = tf.float64), predictions))), m[0]))
    print('Error = {0}%'.format(err.eval()*100.))


# make contour plot
xx, yy = np.mgrid[-1.2:1.2:.01, -1.2:1.2:.01]
xx1 = xx.ravel()
yy1 = yy.ravel()
grid = np.c_[xx1, yy1]
grid_nl = []
# re-apply feature map to inputs x1 & x2
#Convert the below feature map into tensorflow environment

for i in range(1, degree+1):  
    for j in range(0, i+1):
        feat = np.power(xx1, i-j) * np.power(yy1, j)
        if(len(grid_nl) > 0):
            grid_nl = np.c_[grid_nl, feat]
        else:
            grid_nl = feat

grid_nl_in = tf.convert_to_tensor(grid_nl, dtype = tf.float64)
prob1 = []
with tf.Session() as sess:
    probs = regress(grid_nl_in, theta)
    probs = tf.reshape(probs, shape = xx.shape)  
    prob1 = probs.eval()

f, ax = plt.subplots(figsize=(8, 6))
ax.contour(xx, yy, prob1, levels=[.5], cmap="Greys", vmin=0, vmax=.6)
x1 = np.expand_dims(x1, axis=1)
x2 = np.expand_dims(x2, axis=1)
ax.scatter(x1, x2, c=y2, s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)

ax.set(aspect="equal",
       xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
       xlabel="$X_1$", ylabel="$X_2$")
plt.savefig(os.getcwd() + '/Pb_3_Contour_Fig.jpg')
plt.show()
   
plt.plot(cost)
plt.title('Loss vs Epoch')
plt.xlabel('Number of epochs')
plt.ylabel('Cost')
plt.savefig(os.getcwd() + '/Pb_3_Loss_Fig.jpg')
plt.show()