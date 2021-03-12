#!/usr/bin/env python
# coding: utf-8
# Steps in Convolution Neural Networks
1.Convolution
2.Max pooling
3.Flattening
4.Full ConnectionStep 1:Convolution
    *convolution is done to an image using feature dectector
    *when input image is convolveed with the feature detector then we get a feature map
    *by applying convolution operation size of the image is reduced so we may lose some information. but features detector is one which stores the features and unwanted features are removed.
    * WE will apply no of feature detectors(filter) to a single image so we will be getting nof features detectors. SO using no of feature detectors we get max no of features in an image so we will be getting no of feature maps
    * Group of feature map is called CONVOLUTION layerStep 2:Max pooling
Types of pooling-Max pooling,Mean Pooling,Sum pooing
Max Pooling:By applyimg max pooling we ar neglecting 75% of unwanted features and we are reducing spacial invariance this will avoid over fitting of the data .
Step3:Flattening
flattening is converting n dimension to 1 dimension and applying ann to that 1 dimension array which just acts like inputs to the neurons.
When output is not correct then in the backward propogation along with the weights feature detector(filter) is also optimisedStep 4: Full Connection
Full connection is dense layershttp://scs.ryerson.ca/~aharley/vis/conv/
# In[1]:



#import keras libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


# In[2]:


model=Sequential()


# In[3]:


model.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu'))#1St parameter =no of features detectors 2nd& 3rd =Size of feature detector, 
#4th EXPECTED input image size,5 th parameter is channel for color=3 gray scale=1,6 th to avoid negative pixels we use activation function


# In[4]:


model.add(MaxPooling2D(pool_size=(2,2)))#1parmeter=size of pooling matrix


# In[5]:


model.add(Flatten())#CONVERTS ndiminestion to 1 Dx


# In[6]:


model.add(Dense(output_dim=128,activation='relu',init='random_uniform'))


# In[7]:


model.add(Dense(output_dim=1,activation='sigmoid',init='random_uniform'))


# In[8]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[9]:


# IMAGE argumentaion which done for pre processing images  to avoid overfitting of images
from keras.preprocessing.image import ImageDataGenerator


# In[10]:


train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[11]:


x_train = train_datagen.flow_from_directory(r'C:\Users\Akshay\Desktop\dataset\training_set',target_size=(64,64),batch_size=32,class_mode='binary')
x_test = train_datagen.flow_from_directory(r'C:\Users\Akshay\Desktop\dataset\test_set',target_size=(64,64),batch_size=32,class_mode='binary')


# In[13]:


print(x_train.class_indices)


# In[14]:


model.fit_generator(x_train,samples_per_epoch = 8000,epochs=25,validation_data=x_test,nb_val_samples=2000)#(samples_per_epoch= no of traininig or testing images/batch size)
                                                                        #                                                       =8000/32=250


# In[18]:


model.save('mymodel.h5')# this will save the weights,for keras h5 is extension


# In[ ]:




