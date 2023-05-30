# -*- coding: utf-8 -*-
"""
Created on Fri May 19 11:04:19 2023

@author: eliza
"""

import numpy as np
import pandas as pd
import configparser
import matplotlib.pyplot as plt
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']
model_name = config['filepaths']['model_name']
figure_fp = f'{output_data_fp}/{model_name}/figures'

def make_hist_norm(ax, df, label, bins,percentiles,fontsize):
    hist_, bins_ = np.histogram(df[label], bins)
    norm_hist_ = hist_/len(df)
    ax.tick_params(labelsize=12)
    ax.hist(bins_[:-1], bins_, weights=norm_hist_, color='steelblue')
    ax.set_ylabel('Normalised Counts', fontsize=fontsize)
    
    ax.set_ylim(0, max(norm_hist_)+0.1)

    if percentiles == True:
        median_ = round(np.percentile(df[label], 50), 2)
        p75 = round(np.percentile(df[label], 75), 2)
        p25 = round(np.percentile(df[label], 25), 2)
        ax.vlines(p75, 0, max(norm_hist_)+0.1, linestyle='dotted', label='75th percentile={}'.format(p75))
        ax.vlines(median_, 0, max(norm_hist_)+0.1, linestyle='dotted',label=f'50th percentile={median_}')
        ax.vlines(p25, 0, max(norm_hist_)+0.1, linestyle='dotted', label='25th percentile={}'.format(p25))
        ax.legend(fontsize=fontsize-4)
    return ax
    
#Load Data
train_iou_df = pd.read_csv(output_data_fp + f'/{model_name}/train_iou_processed_withhigherthresh.csv')
test_iou_df = pd.read_csv(output_data_fp + f'/{model_name}/test_iou_processed_withhigherthresh.csv')


#Dataframes with metrics for each image (loss, accuracy, iou)
train_metric_df = pd.read_csv(output_data_fp + f'/{model_name}/train_acc_df.csv')
test_metric_df = pd.read_csv(output_data_fp + f'/{model_name}/test_acc_df.csv')

#Make figure of loss accuracy and IoU 
plt.ioff()
plt.style.use('seaborn')
fig, [[ax3,ax4], [ax5, ax6]] = plt.subplots(2, 2, figsize=(16, 10))
plt.subplots_adjust(hspace=0.6)
bins=np.arange(0, 1.05, 0.05)
#training
fontsize=15
ax3 = make_hist_norm(ax3, train_metric_df, 'accuracy', bins, percentiles=True,fontsize=fontsize)
ax3.text(-0.075,1.05, 'a', horizontalalignment='center',verticalalignment='center',transform = ax3.transAxes,fontsize=fontsize, weight='bold')
ax3.set_xlabel('Accuracy', fontsize=fontsize)
ax3.set_title('Training', fontsize=fontsize+6, y=1.02, fontweight=548)
ax5 = make_hist_norm(ax5, train_iou_df, 'iou', bins, percentiles=True,fontsize=fontsize)
ax5.set_xlabel('IoU', fontsize=fontsize+2)
ax5.text(-0.075,1.05, 'b', horizontalalignment='center',verticalalignment='center',transform = ax5.transAxes,fontsize=fontsize, weight='bold')

#testing
ax4 = make_hist_norm(ax4, test_metric_df, 'accuracy', bins, percentiles=True,fontsize=fontsize)
ax4.text(-0.075,1.05, 'c', horizontalalignment='center',verticalalignment='center',transform = ax4.transAxes,fontsize=fontsize, weight='bold')
ax4.set_xlabel('Accuracy', fontsize=fontsize)
ax4.set_title('Testing', fontsize=fontsize+6, y=1.02, fontweight=548)
ax6 = make_hist_norm(ax6, test_iou_df, 'iou', bins, percentiles=True,fontsize=fontsize)
ax6.text(-0.075,1.05, 'd', horizontalalignment='center',verticalalignment='center',transform = ax6.transAxes,fontsize=fontsize, weight='bold')
ax6.set_xlabel('IoU', fontsize=fontsize)
plt.tight_layout()
plt.savefig(figure_fp + '/metrics_training_testing.png')
plt.clf()
#testing

#Make figure of IoU processed for train and test
plt.ioff()
fig, [[ax,ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(16, 10))
plt.subplots_adjust(hspace=0.5)
#Training
ax = make_hist_norm(ax, train_iou_df, 'iou', bins, percentiles=True,fontsize=fontsize)
ax.set_xlabel('IoU', fontsize=14)
ax.set_title('Training', fontsize=18, y=1.1,fontweight=450)
ax.text(0.5, 1.05, 'Before Processing', horizontalalignment='center',
        verticalalignment='center',transform = ax.transAxes,fontsize=15)
ax3 = make_hist_norm(ax3, train_iou_df, 'iou_processed', bins, percentiles=True,fontsize=fontsize)
ax3.set_title('After Processing', fontsize=15)
ax3.set_xlabel('IoU', fontsize=14)
#Testing
ax2 = make_hist_norm(ax2, test_iou_df, 'iou', bins, percentiles=True,fontsize=fontsize)
ax2.set_xlabel('IoU', fontsize=14)
ax2.set_title('Testing',fontsize=18, y=1.1,fontweight=450)
ax2.text(0.5, 1.05, 'Before Processing', horizontalalignment='center',
        verticalalignment='center',transform = ax2.transAxes,fontsize=15)
ax4 = make_hist_norm(ax4, test_iou_df, 'iou_processed', bins, percentiles=True,fontsize=fontsize)
ax4.set_title('After Processing', fontsize=15)
ax4.set_xlabel('IoU', fontsize=14)
#Save figure
plt.tight_layout()
plt.savefig(figure_fp + '/iou_processed_Withhigherthresh_trainandtest.png')
plt.clf()
