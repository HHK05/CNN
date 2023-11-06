#!/usr/bin/env python
# coding: utf-8

# In[37]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[38]:


train_datagen=ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)
training_set = train_datagen.flow_from_directory("C:\\Users\\Harsh\\OneDrive\\Desktop\\hackathin\\neural network\\Part 2 - Convolutional Neural Networks\\dataset\\training_set",
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


# In[39]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory("C:\\Users\\Harsh\\OneDrive\\Desktop\\hackathin\\neural network\\Part 2 - Convolutional Neural Networks\\dataset\\test_set",
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# In[40]:


cnn=tf.keras.models.Sequential()


# In[41]:


#step-1:- convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=[64, 64, 3]))


# In[42]:


#adding pooling layer 
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# In[43]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# In[44]:


#flattening 
cnn.add(tf.keras.layers.Flatten())


# In[45]:


#completely connected layer 
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))


# In[46]:


#outputlayer 
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))


# In[47]:


cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[48]:


cnn.fit(x=training_set,validation_data=test_set,epochs=25)


# In[56]:


import numpy as np 
from keras.preprocessing.image import load_img 
test_image=image.load_img("C:\\Users\\Harsh\\OneDrive\\Desktop\\hackathin\\neural network\\Part 2 - Convolutional Neural Networks\\dataset\\single_prediction\\cat_or_dog_1.jpg",)
test_image=image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result=cnn.predict(test_image)
training_set.class_indices
if result[0][0]>0.5:
    prediction='dog'
else:
    prediction='cat'



# In[ ]:


print(prediction)


# In[ ]:





# In[ ]:




