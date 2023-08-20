# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:34:23 2023

@author: eliza
"""

from shapely.validation import make_valid
import itertools
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras import layers
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, MultiPoint
import configparser
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']
model_name = config['filepaths']['model_name']
figure_fp = f'{output_data_fp}/{model_name}/figures'
if not os.path.exists(figure_fp+'/lfe_sm_results/'):
    os.makedirs(figure_fp+'/lfe_sm_results/')
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



def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
def get_contours(mask):
    binary=np.where(mask>0.5, 1, 0)
    cv2img = binary.astype(np.uint8)
    contours, hierarchy = cv2.findContours(cv2img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    reshape_contours = lambda x: np.reshape(x, (x.shape[0], x.shape[-1]))
    reshaped_contours = [reshape_contours(i) for i in contours]
    return reshaped_contours
def map_frequencies(contour_):
    frequency = np.flip(10**(np.arange(np.log10(3.95), np.log10(1500), \
                (np.log10(1500)-np.log10(3.95))/384, dtype=float)))
    y = np.array(contour_[:,1])
    y_mapped = frequency[y]
    x=contour_[:,0]
    mapped_coords = np.array([(i,j) for i,j in zip(x, y_mapped)])
    return mapped_coords
def total_contours(vals_, results):
    im_id = []
    contours_unmapped = []
    contours_total=[]
    for i in vals_:
        mask = results[i, :, :]
        contours = get_contours(mask)
        mapped_contours = [map_frequencies(i) for i in contours]
        contours_unmapped.append(contours)
        contours_total.append(mapped_contours)
        ids_ = np.repeat(i, len(mapped_contours))        
        im_id.append(ids_)
    im_id = list(itertools.chain.from_iterable(im_id))
    contours_unmapped = list(itertools.chain.from_iterable(contours_unmapped))
    contours_total_ = list(itertools.chain.from_iterable(contours_total))
    return im_id, contours_total_, contours_unmapped
def analyse_contours(vals_, results):
    im_id, contours_total_, contours_unmapped = total_contours(vals_, results)
    ### Delta F
    fmapped_min = [min(i[:,1]) for i in contours_total_]
    min_f = []
    max_f = []
    area = []
    min_t = []
    max_t = []
    for i in contours_unmapped:
        area.append(cv2.contourArea(i))
        min_f.append(min(i[:,1]))
        max_f.append(max(i[:,1]))
        min_t.append(min(i[:,0]))
        max_t.append(max(i[:,0]))
    del_f = [(j-i) for i, j in zip(min_f, max_f)]   
    del_t = [(j-i) for i, j in zip(min_t, max_t)]  
    
    df = pd.DataFrame({'area':area,'min_f':fmapped_min, 'delta_f':del_f, 'delta_t':del_t, 'id':im_id})
    return df
### remove misclassified polygons from masks.
def remove(id_, un_df, results, c_unmapped):
    mask = results[id_, :, :].copy()
    
    contour_inds = un_df.loc[un_df['id']==id_, :].reset_index(drop=True)
    data_len = len(mask.flatten())
    index = np.arange(data_len, dtype=int)
    times, freqs=np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
    coords = [(t, f) for t,f in zip(times.flatten(), freqs.flatten())]
    data_points = MultiPoint([Point(x, y, z) for (x, y), z in zip(coords, index)])
    #indices of each item in flattened 2D signal array.
    index = np.arange(data_len, dtype=int)
    if len(contour_inds)>0:
        inds = np.array(contour_inds.loc[:,'ind']).astype(int)
        for i in inds:
            contour = c_unmapped[i]
            if (contour.shape[0]==1):
                mask[contour[:,1], contour[:,0]] = 0
                
            elif (contour.shape[0]<3) and (contour.shape[0]>1):
                line_coords = np.array([(i, j) for i,j in zip(contour[:, 0], contour[:, 1])])
                line = LineString(line_coords)
                fund_points = line.intersection(data_points)
                fund_coords = np.array([i.coords for i in fund_points.geoms]).astype(int)[:,0,0:2]
                mask[fund_coords[:, 1], fund_coords[:, 0]] = 0
                
            else:
                fund_polygon = Polygon(contour)
                valid_shape = make_valid(fund_polygon)
                fund_points = valid_shape.intersection(data_points)
                fund_coords = np.array([i.coords for i in fund_points.geoms]).astype(int)[:,0,0:2]
                mask[fund_coords[:, 1], fund_coords[:, 0]] = 0
    
    return mask

def twodhist(df):
    scatter_x = np.array(df['delta_f'])
    scatter_y = np.array(df['delta_t'])
    xedges=np.arange(0, 390, 5)
    yedges = np.arange(0, 135, 5)

    H, x, y = np.histogram2d(scatter_x, scatter_y, bins=(xedges, yedges))
    H=(H).T
    return H
def make_hist_norm(ax, df, label, bins,percentiles):
    hist_, bins_ = np.histogram(df[label], bins)
    norm_hist_ = hist_/len(df)
    ax.tick_params(labelsize=12)
    ax.hist(bins_[:-1], bins_, weights=norm_hist_,color='steelblue')
    ax.set_ylabel('Normalised Counts', fontsize=14)
    
    ax.set_ylim(0, max(norm_hist_)+0.05)

    if percentiles == True:
        median_ = round(np.percentile(df[label], 50), 3)
        p75 = round(np.percentile(df[label], 75), 3)
        p25 = round(np.percentile(df[label], 25), 3)
        ax.vlines(p75, 0, max(norm_hist_)+0.1, linestyle='dotted', label='75th percentile={}'.format(p75))
        ax.vlines(median_, 0, max(norm_hist_)+0.1, linestyle='dotted',label=f'50th percentile={median_}')
        ax.vlines(p25, 0, max(norm_hist_)+0.1, linestyle='dotted', label='25th percentile={}'.format(p25))
        ax.legend(fontsize=10)
    return ax
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

thresh = np.load(output_data_fp + f'/{model_name}/best_thresh.npy')
lfe_sm_metrics = pd.read_csv(output_data_fp + f'/{model_name}/lfe_sm_metrics.csv')
###### Analyse Test Set ###############################
test_path = output_data_fp + '/lfe_sm/test_lfesm'
test_results = np.load(output_data_fp + f'/{model_name}/test_lfesm_results.npy', \
                  allow_pickle=True)[:,0,:,:,0]
test_results_masked=np.where(test_results.copy()>=thresh, 1, 0)
test_num_ = np.arange(test_results.shape[0])
test_im_ids = [str(i).zfill(4) for i in test_num_]

#Plot Accuracy Metrics 
plt.ioff()
plt.style.use('seaborn')
fig, [ax,ax2, ax3] = plt.subplots(1,3, figsize=(14, 5))
bins=np.arange(0, 1.05, 0.05)
ax = make_hist_norm(ax, lfe_sm_metrics, 'loss', bins, percentiles=True)
ax.set_xlabel('Loss', fontsize=14)
ax2 = make_hist_norm(ax2, lfe_sm_metrics, 'accuracy', bins, percentiles=True)
ax2.set_xlabel('Accuracy', fontsize=14)
ax3 = make_hist_norm(ax3, lfe_sm_metrics, 'iou', bins, percentiles=True)
ax3.set_xlabel('IoU', fontsize=14)
plt.tight_layout()

#Save figure
plt.savefig(figure_fp + '/lfe_sm_metrics_plot.png')
plt.clf()

#Contour Analysis
im_id, contours_total, contours_unmapped = total_contours(test_num_,test_results_masked)
test_df = analyse_contours(test_num_,test_results_masked)
# Separate polygons according to criterion
test_selected = test_df.loc[(test_df['delta_f']>=100) & (test_df['min_f'] < 100), :]
test_selected_inds = np.array(test_selected.index).astype(int)
selected_contours = [contours_total[i] for i in test_selected_inds]
test_selected.insert(4, 'ind', test_selected_inds, True)
test_selected=test_selected.reset_index(drop=True)
test_unselected_inds = [i for i in range(len(test_df)) if i not in test_selected_inds]
test_unselected = test_df.loc[test_unselected_inds, :].reset_index(drop=True)
test_unselected.insert(4, 'ind', test_unselected_inds, True)

############ Remove contours from mask that didn't fulfill criterion#########

lfe_sm_processed_masks=[]

for i in test_num_:
    print(f'{i}', end='\r')
    mask = remove(i, test_unselected, test_results_masked, contours_unmapped)   
    lfe_sm_processed_masks.append(mask)
lfe_sm_processed_masks=np.array(lfe_sm_processed_masks)
np.save(output_data_fp + f'/{model_name}' + '/train_processed_masks_withhigherthresh.npy', lfe_sm_processed_masks)

#Calculate IoU of processed mask
lfe_sm_iou_processed = []
count=0
#zip through each image and calculate given metric by camparing to true mask.
#lfe_sm_processed_masks = (lfe_sm_processed_masks == 0)
for i in test_num_:
    mask = load_mask(str(i).zfill(3), test_path)
    #proc_result = np.ma.masked_array(test_results[i, :, :], lfe_sm_processed_masks[i, :, :]).filled(0)
    iou = tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=thresh)
    
    #Processed
    iou.update_state(mask, lfe_sm_processed_masks[i, :, :])
    lfe_sm_iou_processed.append(iou.result().numpy())
    iou.reset_state()
    
    print(f'{i}', end='\r')
    count+=1
lfe_sm_iou_processed = np.array(lfe_sm_iou_processed)
np.save(output_data_fp + f'/{model_name}/lfe_sm_processed_iou.npy', lfe_sm_iou_processed)
#Make figure of IoU processed for train and test
plt.ioff()
fig, [ax, ax2] = plt.subplots(2, 1, figsize=(10, 10))
lfe_sm_iou_processed_df = pd.DataFrame({'iou_processed': lfe_sm_iou_processed})
ax = make_hist_norm(ax, lfe_sm_metrics, 'iou', bins, percentiles=True)
ax.set_xlabel('IoU', fontsize=14)
ax.set_title('Before Processing', fontsize=18)
ax2 = make_hist_norm(ax2, lfe_sm_iou_processed_df, 'iou_processed', bins, percentiles=True)
ax2.set_xlabel('IoU', fontsize=14)
ax2.set_title('After Processing', fontsize=18)
plt.tight_layout()
plt.savefig(figure_fp + '/iou_processed_Withhigherthresh_lfesm.png')
plt.clf()



