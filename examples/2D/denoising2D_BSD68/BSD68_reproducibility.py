#!/usr/bin/env python
# coding: utf-8

# # Noise2Void - 2D Example for BSD68 Data
# 
# The data used in this notebook is the same as presented in the paper.

# In[1]:

import __init__
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

from config import Config
import logging
# V0: no cat in neighbor
# V1: concat + residual


args = Config()._get_args()

# In[2]:


# In[3]:


# create a folder for our data
if not os.path.isdir('./data'):
    os.mkdir('data')

# check if data has been downloaded already
zipPath = "data/BSD68_reproducibility.zip"
if not os.path.exists(zipPath):
    # download and unzip data
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
X = np.clip(X, 0, 255.).astype(np.uint8).astype(np.float32)
X_val = np.clip(X_val, 0, 255.)

config = N2VConfig(X, unet_kern_size=3,
                   train_steps_per_epoch=400, train_epochs=200, train_loss='mse', batch_norm=True,
                   train_batch_size=args.batch_size, n2v_perc_pix=args.perc_pix,
                   n2v_patch_shape=(args.patch_size, args.patch_size),
                   unet_n_first=96,
                   unet_residual=True,
                   n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=2)
vars(config)

model = N2V(config, args.exp_name, basedir=args.ckpt_dir)
model.prepare_for_training(metrics=())


# We are ready to start training now.
history = model.train(X, X_val)
#
#
# # ### After training, lets plot training and validation loss.
#
# # In[10]:
#
#
# print(sorted(list(history.history.keys())))
# plt.figure(figsize=(16,5))
# plot_history(history,['loss','val_loss']);


# # Compute PSNR to GT

# In[11]:


groundtruth_data = np.load('data/BSD68_reproducibility_data/test/bsd68_groundtruth.npy', allow_pickle=True)

# In[12]:


test_data = np.load('data/BSD68_reproducibility_data/test/bsd68_gaussian25.npy', allow_pickle=True)
test_data = np.clip(test_data, 0, 255.).astype(np.uint8).astype(np.float32)
# In[13]:

from skimage.measure import compare_psnr, compare_ssim
import cv2

# Weights corresponding to the smallest validation loss
# Smallest validation loss does not necessarily correspond to best performance, 
# because the loss is computed to noisy target pixels.
model.load_weights('weights_best.h5')
# % show images.
pred = []
psnrs = []
ssims = []
for i, (gt, img) in enumerate(zip(groundtruth_data, test_data)):
    p_ = model.predict(img.astype(np.float32), 'YX')

    p_ = np.clip(p_, 0, 255.)  # prediction should be clipped.

    pred.append(p_)
    psnrs.append(compare_psnr(gt, p_, data_range=255.))
    ssims.append(compare_ssim(gt, p_, data_range=255.))

    # plt.figure(figsize=(20,20))
    # # We show the noisy input...
    # plt.subplot(1,3,1)
    # plt.imshow(img[:, :], cmap='gray')
    # plt.title('Input');
    #
    # # and the result.
    # plt.subplot(1,3,2)
    # plt.imshow( p_[:, :], cmap='gray')
    # plt.title('Prediction');
    #
    # plt.subplot(1,3,3)
    # plt.imshow( gt[:, :], cmap='gray')
    # plt.title('GT');
    #
    # plt.show()
    # print('psnr:', psnrs[-1])
    # save
    filename = '{}-{}-{:.3f}-{:.3f}.png'.format('last', i, psnrs[-1], ssims[-1])
    cv2.imwrite(os.path.join(args.res_dir, filename), p_.astype(np.uint8))
    logging.info('Best ckpt. {} \t psnr:{:.2f} \t ssim: {:.2f}'.format(i, psnrs[-1], ssims[-1]))
psnrs = np.array(psnrs)
logging.info('Best psnr:{:.2f} \t ssim: {:.3f}'.format(np.round(np.mean(psnrs), 2), np.round(np.mean(ssims), 2)))

# In[17]:


# The weights of the converged network. 
model.load_weights('weights_last.h5')

# % show images.
pred = []
psnrs = []
ssims = []
for i, (gt, img) in enumerate(zip(groundtruth_data, test_data)):
    p_ = model.predict(img.astype(np.float32), 'YX')

    p_ = np.clip(p_, 0, 255.)  # prediction should be clipped.

    pred.append(p_)
    psnrs.append(compare_psnr(gt, p_, data_range=255.))
    ssims.append(compare_ssim(gt, p_, data_range=255.))

    # plt.figure(figsize=(20,20))
    # # We show the noisy input...
    # plt.subplot(1,3,1)
    # plt.imshow(img[:, :], cmap='gray')
    # plt.title('Input');
    #
    # # and the result.
    # plt.subplot(1,3,2)
    # plt.imshow( p_[:, :], cmap='gray')
    # plt.title('Prediction');
    #
    # plt.subplot(1,3,3)
    # plt.imshow( gt[:, :], cmap='gray')
    # plt.title('GT');
    #
    # plt.show()
    # print('psnr:', psnrs[-1])
    # save
    filename = '{}-{}-{:.3f}-{:.3f}.png'.format('last', i, psnrs[-1], ssims[-1])
    cv2.imwrite(os.path.join(args.res_dir, filename), p_.astype(np.uint8))
    logging.info('Last ckpt. {} \t psnr:{:.2f} \t ssim: {:.2f}'.format(i, psnrs[-1], ssims[-1]))
psnrs = np.array(psnrs)
logging.info('Last psnr:{:.2f} \t ssim: {:.3f}'.format(np.round(np.mean(psnrs), 2), np.round(np.mean(ssims), 2)))
