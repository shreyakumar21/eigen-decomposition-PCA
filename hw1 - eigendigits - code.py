#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import operator
from keras.datasets import mnist
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


def hw1FindEigendigits(A):
    # A = x by k
    avg = np.mean(A,axis=1) # Mean of A
    avg = avg[:,np.newaxis]   
    A = A - avg 
    # Covariance of matrix A 
    covariance,var1 = la.eig( A.T @ A ) # finding the covariance matrix   
    indx = np.argsort(-covariance) #Eigenvectors ranking using indices
    covariance = covariance[indx]
    var1 = var1[:,indx]  
    var1 = A @ var1 # k x k dimension          
    V = normalize(var1.real, axis=0) #normalize the eigenvector
    return avg,V


# In[3]:


(X_train_data, y_train_data), (X_test_data, y_test_data) = mnist.load_data()
X_train_data = X_train_data.reshape(len(X_train_data),-1).T
X_test_data = X_test_data.reshape(len(X_test_data),-1).T


# In[4]:


def plot_images(images, col=5, row=1):
    k=images.shape[1]
    images=images.T.reshape(k,28,28)
    fig=plt.figure(figsize=(10,10)) 
    for i,img in enumerate(images,1):
        try:
            fig.add_subplot(row,col,i+1)
            plt.imshow(img)
        except ValueError:
            break
    plt.show()


# In[5]:


A = X_train_data[:,:500]
n_column = 6 #Number of columns for digits
n_row = 3 #Number of rows for digits
mean, eigendigits = hw1FindEigendigits(A)
SEED_value = 10
plot_images(A,n_column,n_row)


# In[6]:


plot_images(eigendigits,n_column,n_row)


# In[7]:


eigen_size = 500
images_sample = X_test_data[:,:10] 
images_sample = (images_sample - mean).T  
projection  = images_sample @ eigendigits[:,:eigen_size]  
reconstruct = (eigendigits[:,:eigen_size] @ projection.T) + mean  


# In[8]:


plot_images(X_test_data[:,:10],n_column,n_row)


# In[9]:


plot_images(reconstruct,n_column,n_row)


# In[10]:


train_size_expected_accuracy = []
train_size = np.array([50, 100, 200, 500, 1000, 2500, 5000, 7500])

for size in train_size:
    A = X_train_data[:,:100]
    mean, eigendigits = hw1FindEigendigits(A)
   
    X_train_projection = (X_train_data - mean).T @ eigendigits
    X_test_projection  = (X_test_data  - mean).T @ eigendigits
    
    # Split training and testing dataset
    _, X_train_sample,_, y_train_sample = train_test_split(X_train_projection, y_train_data, test_size=size, random_state=SEED_value)
    sample_X_test = X_test_projection[:7500,:] #try 10-15% of 60000
    sample_Y_test = y_test_data[:7500]
    
    kNN = KNeighborsClassifier(3, weights='distance')
    kNN.fit(X_train_sample,y_train_sample)
    accuracy = kNN.score( sample_X_test, sample_Y_test)
    train_size_expected_accuracy.append(accuracy*100)

    print("Accuracy percentage for eigen samples training size %s: %s" %(size, round(accuracy*100, 4)))


# In[11]:


def accuracy_plot(accuracy_vector):
   train_size = np.array([50, 100, 200, 500, 1000, 2500, 5000, 7500])
   fig, ax = plt.subplots()
   ax.plot(train_size,train_size_expected_accuracy, '#800000')
   ax.set_title('Accuracy % vs Number of training images')
   ax.set_facecolor('#C0C0C0')
   ax.set_xlabel('Number of training images')
   ax.set_ylabel('Accuracy %')
   plt.show()


# In[12]:


accuracy_plot(train_size_expected_accuracy)


# In[ ]:


eigen_accuracy = []
num_eig_samples = np.array([10,25,50,75,100,150,200,300,400,500])

# Find mean and eigendigits
A = X_train_data[:,:500]
mean, V = hw1FindEigendigits(A)

for n in num_eig_samples:
    eigendigits = V[:,:n]
    
    # Project data
    X_train_projection = (X_train_data - mean).T @ eigendigits
    X_test_projection  = (X_test_data  - mean).T @ eigendigits
    
    # Split training and testing dataset
    _, X_train_sample,_, y_train_sample = train_test_split(X_train_projection, y_train_data, test_size=size, random_state=SEED_value)
    X_test_sample = X_test_projection[:5000,:]
    y_test_sample = y_test_data[:5000]
    
    # kNN
    kNN = KNeighborsClassifier(3, weights='distance')
    kNN.fit(X_train_sample,y_train_sample)
    accuracy = kNN.score(X_test_sample, y_test_sample)
    eigen_accuracy.append(accuracy*100)

    print("accuracy for eigen values %s: %s" %(n, round(accuracy*100, 4)))


# In[ ]:


fig, ax = plt.subplots()
ax.plot(num_eig_samples,eigen_accuracy, '#800000')
ax.set_facecolor('#C0C0C0')
ax.set_title('Accuracy % vs Eigenvector dimension values')
ax.set_xlabel('Eigenvector Dimension values')
ax.set_ylabel('Accuracy %')
plt.show()
#change labels and colors


# In[ ]:


A = X_train_data[:,:700]
mean, eigendigits = hw1FindEigendigits(A)
eigendigits = eigendigits[:,:50]    # Top 50 eigenvectors

# Project data
X_train_proj = (X_train_data - mean).T @ eigendigits
X_test_proj  = (X_test_data  - mean).T @ eigendigits

# Split training and testing dataset
_, X_train_sample,_, y_train_sample = train_test_split(X_train_proj, y_train_data, test_size=0.33, random_state=SEED_value)
X_test_sample = X_test_proj[:7500,:]
y_test_sample = y_test_data[:7500]

kNN_accuracy = []
k_value = []
for k in range(1,30,2):
    kNN = KNeighborsClassifier(k, weights='distance')
    kNN.fit(X_train_sample,y_train_sample)
    accuracy = kNN.score(X_test_sample, y_test_sample)
    kNN_accuracy.append(accuracy*100)
    k_value.append(k)

index, value = max(enumerate(kNN_accuracy), key=operator.itemgetter(1))
print("k = %s with accuracy = %s and this is the best k-value" %(k_value[index], round(value,4)))


# In[ ]:


fig, ax = plt.subplots()
ax.plot(k_value, kNN_accuracy, '#800000')
ax.set_facecolor('#C0C0C0')
ax.set_xlabel('k Nearest Neighbors values')
ax.set_ylabel('Accuracy %')
ax.set_title('Accuracy % Vs Nearest Neighbors')
plt.show()

