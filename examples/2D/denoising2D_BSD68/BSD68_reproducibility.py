#!/usr/bin/env python
# coding: utf-8

# # Noise2Void - 2D Example for BSD68 Data
# 
# The data used in this notebook is the same as presented in the paper.

# In[1]:


# We import all our dependencies.
from n2v.models import N2VConfig, N2V
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt
import urllib
import os
import zipfile


# # Training Data Preparation

# In[2]:



# In[3]:


# create a folder for our data
if not os.path.isdir('./data'):
    os.mkdir('data')

# check if data has been downloaded already
zipPath="data/BSD68_reproducibility.zip"
if not os.path.exists(zipPath):
    #download and unzip data
    data = urllib.request.urlretrieve('https://cloud.mpi-cbg.de/index.php/s/pbj89sV6n6SyM29/download', zipPath)
    with zipfile.ZipFile(zipPath, 'r') as zip_ref:
        zip_ref.extractall("data")


# In[4]:


X = np.load('data/BSD68_reproducibility_data/train/DCNN400_train_gaussian25.npy')
X_val = np.load('data/BSD68_reproducibility_data/val/DCNN400_validation_gaussian25.npy')

# Adding channel dimension
X = X[..., np.newaxis]
print(X.shape)
X_val = X_val[..., np.newaxis]
print(X_val.shape)


# In[5]:


# IMPORTANT!! I add clip
X = np.clip(X, 0, 255.)
X_val = np.clip(X_val, 0, 255.)


# In[6]:


# Let's look at one of our training and validation patches.
plt.figure(figsize=(14,7))
plt.subplot(1,2,1)
plt.imshow(X[0,...,0], cmap='gray')
plt.title('Training Patch');
plt.subplot(1,2,2)
plt.imshow(X_val[0,...,0], cmap='gray')
plt.title('Validation Patch');


# # Configure

# In[7]:


config = N2VConfig(X, unet_kern_size=3, 
                   train_steps_per_epoch=400, train_epochs=200, train_loss='mse', batch_norm=True, 
                   train_batch_size=128, n2v_perc_pix=0.198, n2v_patch_shape=(64, 64), 
                   unet_n_first = 96,
                   unet_residual = True,
                   n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=2)

# Let's look at the parameters stored in the config-object.
vars(config)


# In[8]:


# a name used to identify the model
model_name = 'BSD68_test'
# the base directory in which our model will live
basedir = 'models'
# We are now creating our network model.
model = N2V(config, model_name, basedir=basedir)
model.prepare_for_training(metrics=())


# # Training
# 
# Training the model will likely take some time. We recommend to monitor the progress with TensorBoard, which allows you to inspect the losses during training. Furthermore, you can look at the predictions for some of the validation images, which can be helpful to recognize problems early on.
# 
# You can start TensorBoard in a terminal from the current working directory with tensorboard --logdir=. Then connect to http://localhost:6006/ with your browser.

# In[9]:


# We are ready to start training now.
history = model.train(X, X_val)


# ### After training, lets plot training and validation loss.

# In[10]:


print(sorted(list(history.history.keys())))
plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss']);


# # Compute PSNR to GT

# In[11]:


groundtruth_data = np.load('data/BSD68_reproducibility_data/test/bsd68_groundtruth.npy', allow_pickle=True)


# In[12]:


test_data = np.load('data/BSD68_reproducibility_data/test/bsd68_gaussian25.npy', allow_pickle=True)


# In[13]:


def PSNR(gt, img):
    mse = np.mean(np.square(gt - img))
    return 20 * np.log10(255) - 10 * np.log10(mse)


# In[14]:


# Weights corresponding to the smallest validation loss
# Smallest validation loss does not necessarily correspond to best performance, 
# because the loss is computed to noisy target pixels.
model.load_weights('weights_best.h5')


# In[15]:


pred = []
psnrs = []
for gt, img in zip(groundtruth_data, test_data):
    p_ = model.predict(img.astype(np.float32), 'YX');
    pred.append(p_)
    psnrs.append(PSNR(gt, p_))

psnrs = np.array(psnrs)


# In[16]:


print("PSNR:", np.round(np.mean(psnrs), 2))


# In[17]:


# The weights of the converged network. 
model.load_weights('weights_last.h5')


# In[18]:


pred = []
psnrs = []
for gt, img in zip(groundtruth_data, test_data):
    p_ = model.predict(img.astype(np.float32), 'YX')
    pred.append(p_)
    psnrs.append(PSNR(gt, p_))
    
    
psnrs = np.array(psnrs)


# In[19]:


print("PSNR:", np.round(np.mean(psnrs), 2))


# In[22]:


# % show images.
pred = []
psnrs = []
for gt, img in zip(groundtruth_data, test_data):
    img = np.clip(img, 0, 255.) # add. 
    
    p_ = model.predict(img.astype(np.float32), 'YX')
    
    p_ = np.clip(p_, 0, 255.) # add.
    
    pred.append(p_)
    psnrs.append(PSNR(gt, p_))
    
    plt.figure(figsize=(20,20))
    # We show the noisy input...
    plt.subplot(1,3,1)
    plt.imshow(img[:, :], cmap='gray')
    plt.title('Input');

    # and the result.
    plt.subplot(1,3,2)
    plt.imshow( p_[:, :], cmap='gray')
    plt.title('Prediction');
    
    plt.subplot(1,3,3)
    plt.imshow( gt[:, :], cmap='gray')
    plt.title('GT');
    
    plt.show()
    print('psnr:', psnrs[-1])
psnrs = np.array(psnrs)


# In[21]:


print("PSNR:", np.round(np.mean(psnrs), 2))


# In[ ]:




