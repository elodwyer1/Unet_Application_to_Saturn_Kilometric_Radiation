# -*- coding: utf-8 -*-
"""
Created on Tue May 23 17:18:33 2023

@author: eliza
"""

import numpy as np
import pandas as pd
import configparser
from mpl_toolkits import axes_grid1
import os
import tensorflow as tf
from matplotlib import colors
from keras import layers
import matplotlib.pyplot as plt
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']
model_name = config['filepaths']['model_name']
figure_fp = f'{output_data_fp}/{model_name}/figures'
def load_image(id_name, path):
    
    #path = root + 'ML_Dataset/Main_Dataset/train'
    image_h = 384
    image_w = 128
    
    image_path = os.path.join(path, id_name, "images", id_name) + ".npy"
    
    ## Reading Image
    _image = np.load(image_path, allow_pickle=True)
    resize = tf.keras.Sequential([layers.Resizing(image_h, image_w)])
    image = resize(_image)
    
    return image
def load_mask(id_name, path):
    #path = root + 'ML_Dataset/Main_Dataset/train'
    image_h = 384
    image_w = 128
    
    #image_path = os.path.join(path, id_name, "images", id_name) + ".npy"
    mask_path = os.path.join(path, id_name, "masks", id_name) + ".npy"
    
    ## Reading Image
    #image = np.load(image_path, allow_pickle=True)
    #resize = tf.keras.Sequential([layers.Resizing(image_h, image_w)])
    #image = resize(image)
    
    # Reading mask
    _mask_image = np.load(mask_path, allow_pickle=True)
    _mask_image=np.reshape(_mask_image,(_mask_image.shape[0],_mask_image.shape[1],1))
    resize = tf.keras.Sequential([layers.Resizing(image_h, image_w),layers.Rescaling(1./255)])
    mask = resize(_mask_image)[:,:,0]
    mask=np.where(mask>0, 1, 0).reshape(image_h, image_w,1)
    return mask
def plot_spectrogram(ax, flux):
    frequency_tmp = np.load(output_data_fp + '/frequency.npy')
    step = (np.log10(max(frequency_tmp))-np.log10(min(frequency_tmp)))/383
    frequency = np.flip(10**(np.arange(np.log10(frequency_tmp[0]),
                    np.log10(frequency_tmp[-1])+step/2, step, dtype=float)))
    time = np.arange(flux.shape[1])
    fontsize=14
    im = ax.pcolormesh(time, frequency, flux, cmap='binary_r')
    ax.set_yscale('log')
    ax.tick_params(labelsize=fontsize)
    return im  
#Load Data
train_iou_df = pd.read_csv(output_data_fp + f'/{model_name}/train_iou_processed_withhigherthresh.csv')
test_iou_df = pd.read_csv(output_data_fp + f'/{model_name}/test_iou_processed_withhigherthresh.csv')


#Dataframes with metrics for each image (loss, accuracy, iou)
train_metric_df = pd.read_csv(output_data_fp + f'/{model_name}/train_acc_df.csv')
test_metric_df = pd.read_csv(output_data_fp + f'/{model_name}/test_acc_df.csv')

#Load Data
test_label = np.load(output_data_fp + "/test_label.npy", allow_pickle=True)
test_path = output_data_fp +  '/test'
test_results = np.load(output_data_fp + f'/{model_name}' + '/test_results.npy', \
                  allow_pickle=True)[:,0,:,:,0]
#Probability threshold with maximum average IoU.
thresh = np.load(output_data_fp + f'/{model_name}/best_thresh.npy')
#Mask results at this threshold.
test_results_masked = np.where(test_results.copy()>=thresh, 1, 0)
#Number of samples in test set.
test_num_ = np.arange(test_results.shape[0])
#image 'id' for loading test images.
ids = [str(i).zfill(3) for i in test_num_]
#processed masks
test_processed_masks = np.load(output_data_fp + f'/{model_name}' + '/test_processed_masks_withhigherthresh.npy')

good_lfe= test_iou_df.loc[(test_iou_df['label']!= 'NoLFE') &
                         (test_iou_df['iou']> 0.93), :]
good_nolfe = test_iou_df.loc[(test_iou_df['label']== 'NoLFE') &
                            (test_iou_df['iou']> 0.9), :]
bad_lfe= test_iou_df.loc[(test_iou_df['label']!= 'NoLFE') &
                         (test_iou_df['iou']<0.8), :]
lfe_ext = test_iou_df.loc[test_iou_df['label']=='LFE_ext',:]
bad_nolfe = test_iou_df.loc[(test_iou_df['label']== 'NoLFE') &
                            (test_iou_df['iou']<0.8), :]
all_bad_ids = np.concatenate([bad_lfe.index, bad_nolfe.index])
fontsize=5
class_labels = {'NoLFE':'NoLFE', 'LFE':'LFE', 'LFE_sp':'LFE$\mathrm{_{sp}}$', 
                'LFE_m':'LFE$\mathrm{_m}$', 'LFE_dg':'LFE$\mathrm{_{dg}}$',
                'LFE_ext':'LFE$\mathrm{_{ext}}$'}

plt.ioff()
good_lfe_inds = [8, 84, 200, 455, 445, 9]
count=0
labels = [['a', 'b', 'c', 'd'], ['e', 'f', 'g', 'h'], ['i', 'j', 'k', 'l'], ['m', 'n', 'o', 'p'], ['q', 'r', 's', 't'] ,['u','v', 'w','x']]
for num in range(len(good_lfe_inds)):
    i=good_lfe_inds[num]
    fig, [ax1, ax2, ax3, ax4] = plt.subplots(1, 4, figsize=(13,3))
    labs = labels[num]
    ax1.text(-0.075,1.06, labs[0], horizontalalignment='center',verticalalignment='center',transform = ax1.transAxes,fontsize=fontsize+10, weight='bold')
    ax2.text(-0.075,1.06, labs[1], horizontalalignment='center',verticalalignment='center',transform = ax2.transAxes,fontsize=fontsize+10, weight='bold')
    ax3.text(-0.075,1.06, labs[2], horizontalalignment='center',verticalalignment='center',transform = ax3.transAxes,fontsize=fontsize+10, weight='bold')
    ax4.text(-0.075,1.06, labs[3], horizontalalignment='center',verticalalignment='center',transform = ax4.transAxes,fontsize=fontsize+10, weight='bold')
         
    plt.subplots_adjust(wspace=0.1)
    plt.style.use('default')
    fig.supxlabel('Time',fontsize=fontsize+8,y=0.1)
    #fig.supylabel('Frequency (kHz)',fontsize=fontsize+8)
    class_ = test_iou_df.loc[i, 'label']
    fig.suptitle(f'Models prediction of {class_labels[class_]} Class with an IoU of {round(test_iou_df.iou[i],2)}',
                 fontsize=fontsize+8, y=0.92)
    #lo Data
    im = load_image(str(i).zfill(3), test_path)
    flux = im[:,:,0]
    pol = im[:,:,1]
    mask = load_mask(str(i).zfill(3), test_path)[:,:,0]
    pred= test_results_masked[i,:,:]
    im=plot_spectrogram(ax1, flux)
    ax1.set_ylabel('Frequency (kHz)',fontsize=fontsize+6)
    ax1.tick_params(labelsize=fontsize+6)
    divider = axes_grid1.make_axes_locatable(ax1)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    step=1/6
    tick = np.arange(0, 1.01, step)
    cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax1, ticks=tick)
    lab=[1e-25, 1e-24, 1e-23, 1e-22, 1e-21, 1e-20, 1e-19]
    cb.ax.set_yticklabels(lab, fontsize=fontsize+2)
    cb.ax.tick_params(labelsize=fontsize+2)
    cb.ax.set_title(r'Wm$^{-2}$H$z^{-1}$', fontsize=fontsize+3)
    #cb.set_label(r'Wm$^{-2}Hz^{-1}$', fontsize=fontsize+4)
    
    im2=plot_spectrogram(ax2, pol)
    ax2.tick_params(labelsize=fontsize+6)
    divider = axes_grid1.make_axes_locatable(ax2)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    step = 1/4
    tick = np.arange(0, 1+step, step)
    cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax2, ticks=tick)
    label = [-1, -0.5, 0, 0.5, 1]
    cb.ax.set_yticklabels(label, fontsize=fontsize+2)
    cb.ax.tick_params(labelsize=fontsize+2)
    cb.ax.set_title(r'V', fontsize=fontsize+3)
    
    im3=plot_spectrogram(ax3, mask)
    ax3.tick_params(labelsize=fontsize+6)
    ax3.set_title('True Mask', fontsize=fontsize+8)
    
    frequency = np.flip(10**(np.arange(np.log10(3.95), np.log10(1500), \
                (np.log10(1500)-np.log10(3.95))/384, dtype=float)))
    time = np.arange(flux.shape[1])
    scalez= colors.Normalize(0, 1)
    im = ax4.pcolormesh(time, frequency, pred, norm=scalez, cmap='binary_r')
    ax4.set_yscale('log')
    ax4.tick_params(labelsize=fontsize+6)
    ax4.set_title('Predicted Mask', fontsize=fontsize+8)
    #divider = axes_grid1.make_axes_locatable(ax4)
    #cax = divider.append_axes("right", size=0.15, pad=0.2)
    #cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax4)
    #cb.ax.tick_params(labelsize=fontsize-2)
    #cb.set_label('Probability', fontsize=fontsize)
    
    plt.tight_layout()
    plt.savefig(figure_fp + f'/good_{class_}_results_{i}.png',bbox_inches='tight')
    plt.close(fig)
    plt.clf()
    count+=1
count=0
bad_lfe_inds = [85,13, 331, 460, 407, 410]
labels = [['a', 'b', 'c', 'd'], ['e', 'f', 'g', 'h'], ['i', 'j', 'k', 'l'], ['m', 'n', 'o', 'p'], ['q', 'r', 's', 't'] ,['u','v', 'w','x']]
for num in range(len(bad_lfe_inds)):
    i=bad_lfe_inds[num]
    fig, [ax1, ax2, ax3, ax4] = plt.subplots(1, 4, figsize=(13,3))
    labs = labels[num]
    ax1.text(-0.075,1.06, labs[0], horizontalalignment='center',verticalalignment='center',transform = ax1.transAxes,fontsize=fontsize+10, weight='bold')
    ax2.text(-0.075,1.06, labs[1], horizontalalignment='center',verticalalignment='center',transform = ax2.transAxes,fontsize=fontsize+10, weight='bold')
    ax3.text(-0.075,1.06, labs[2], horizontalalignment='center',verticalalignment='center',transform = ax3.transAxes,fontsize=fontsize+10, weight='bold')
    ax4.text(-0.075,1.06, labs[3], horizontalalignment='center',verticalalignment='center',transform = ax4.transAxes,fontsize=fontsize+10, weight='bold')
         
    plt.subplots_adjust(wspace=0.1)
    plt.style.use('default')
    fig.supxlabel('Time',fontsize=fontsize+8,y=0.1)
    #fig.supylabel('Frequency (kHz)',fontsize=fontsize+8)
    class_ = test_iou_df.loc[i, 'label']
    fig.suptitle(f'Models prediction of {class_labels[class_]} Class with an IoU of {round(test_iou_df.iou[i],2)}',
                 fontsize=fontsize+8, y=0.92)
    #lo Data
    im = load_image(str(i).zfill(3), test_path)
    flux = im[:,:,0]
    pol = im[:,:,1]
    mask = load_mask(str(i).zfill(3), test_path)[:,:,0]
    pred= test_results_masked[i,:,:]
    im=plot_spectrogram(ax1, flux)
    ax1.set_ylabel('Frequency (kHz)',fontsize=fontsize+6)
    ax1.tick_params(labelsize=fontsize+6)
    divider = axes_grid1.make_axes_locatable(ax1)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    step=1/6
    tick = np.arange(0, 1.01, step)
    cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax1, ticks=tick)
    lab=[1e-25, 1e-24, 1e-23, 1e-22, 1e-21, 1e-20, 1e-19]
    cb.ax.set_yticklabels(lab, fontsize=fontsize+2)
    cb.ax.tick_params(labelsize=fontsize+2)
    cb.ax.set_title(r'Wm$^{-2}$H$z^{-1}$', fontsize=fontsize+3)
    #cb.set_label(r'Wm$^{-2}Hz^{-1}$', fontsize=fontsize+4)
    
    im2=plot_spectrogram(ax2, pol)
    ax2.tick_params(labelsize=fontsize+6)
    divider = axes_grid1.make_axes_locatable(ax2)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    step = 1/4
    tick = np.arange(0, 1+step, step)
    cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax2, ticks=tick)
    label = [-1, -0.5, 0, 0.5, 1]
    cb.ax.set_yticklabels(label, fontsize=fontsize+2)
    cb.ax.tick_params(labelsize=fontsize+2)
    cb.ax.set_title(r'V', fontsize=fontsize+3)
    
    im3=plot_spectrogram(ax3, mask)
    ax3.tick_params(labelsize=fontsize+6)
    ax3.set_title('True Mask', fontsize=fontsize+8)
    
    frequency = np.flip(10**(np.arange(np.log10(3.95), np.log10(1500), \
                (np.log10(1500)-np.log10(3.95))/384, dtype=float)))
    time = np.arange(flux.shape[1])
    scalez= colors.Normalize(0, 1)
    im = ax4.pcolormesh(time, frequency, pred, norm=scalez, cmap='binary_r')
    ax4.set_yscale('log')
    ax4.tick_params(labelsize=fontsize+6)
    ax4.set_title('Predicted Mask', fontsize=fontsize+8)
    #divider = axes_grid1.make_axes_locatable(ax4)
    #cax = divider.append_axes("right", size=0.15, pad=0.2)
    #cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax4)
    #cb.ax.tick_params(labelsize=fontsize-2)
    #cb.set_label('Probability', fontsize=fontsize)
    
    plt.tight_layout()
    plt.savefig(figure_fp + f'/bad_{class_}_results_{i}.png',bbox_inches='tight')
    plt.close(fig)
    plt.clf()
    count+=1
