##Plot dataset and design boundary
##Credit given https://towardsdatascience.com/decision-boundary-visualization-a-z-6a63ae9cca7d

import matplotlib.pyplot as plt
import numpy as np

# GRADED FUNCTION: sigmoid

def sigmoid(z):
    c = 1/(1+np.exp(-z))
    return c

def normalize(mtx):
    
    mean = np.mean(mtx, axis = 1, keepdims = True)
    std = np.std(mtx, axis = 1, keepdims = True)
    mtx = (mtx-mean) /std
    return mtx


def predict(W, w_0, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    W = W.reshape(X.shape[0], 1)
    
    ### START CODE HERE ### 
    z = np.dot(W.T, X) + w_0
    Y_prediction = sigmoid(z)
    ### END CODE HERE ###
    
    
    
    return Y_prediction

def map_features(x, degree):
    x_old = x.copy()
    
    column_index = 0
    for i in range(2, degree+1):
        for j in range(0, i+1):
            itm = np.multiply(x_old[:,0]**(i-j), x_old[:,1]**(j))
            itm = itm.reshape(itm.shape[0], 1)
            x = np.append(x, itm, axis = 1)
            column_index+=1
    return x
	
	
def plotData(X_train, y_train, xlabel = "X1", ylabel="X2", w=[], b = 0, plot_db = False, part_I = False):
	c0 = c1 = 0 # Counter of label 0 and label 1 instances
	X_train = X_train.T #Transpose input data
	X_train = X_train[:, [0,1]]
	y_train = y_train.T
	for i in range(0, X_train.shape[0]):
		if y_train[i] == 0:
			c0 = c0 + 1
		else:
			c1 = c1 + 1
			
	x0 = np.ones((c0,2)) # matrix label 0 instances
	x1 = np.ones((c1,2)) # matrix label 1 instances
	k0 = k1 = 0

	for i in range(0,y_train.shape[0]):
		if y_train[i] == 0:
			x0[k0] = X_train[i]
			k0 = k0 + 1
		else:
			x1[k1] = X_train[i]
			k1 = k1 + 1

	X_col = [x0, x1]
	colors = ["green", "blue"] # colours for Scatter Plot

		
	for x, c in zip(X_col, colors):
		if c == "green":
			plt.scatter(x[:,0], x[:,1], color = c, label = "Negative")
		else:
			plt.scatter(x[:,0], x[:,1], color = c, label = "Positive")
	
	#Plot decision boundary for Part I
	if  plot_db and part_I:
		w = w[[0,1], :]
		w = w.reshape(2)
		w = np.insert(w, 0, b)
		# getting the x co-ordinates
		plot_x = np.array([min(X_train[:,0]) - 2, max(X_train[:,0]) + 2])
		# getting corresponding y co-ordinates of the decision boundary
		plot_y = (-1/w[2]) * (w[1] * plot_x + w[0])
		# Plotting the Single Line Decision Boundary
		plt.plot(plot_x, plot_y, label = "Decision_Boundary")
		plt.legend()
	
	#Plot decision boundary for Part II
	if plot_db and not part_I:
		h = .02  # step size in the mesh
		x_min, x_max = min(X_train[:, 0]), max(X_train[:, 0])
		y_min, y_max = min(X_train[:, 1]), max(X_train[:, 1])
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
							 np.arange(y_min, y_max, h))

		nX = normalize(map_features(np.c_[xx.ravel(), yy.ravel()], 6).T)

		#Put the result into a color plot
		Z = predict(w, b, nX)
		Z = Z.reshape(xx.shape)
		plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
		
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)