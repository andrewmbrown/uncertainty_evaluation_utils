import os
import math
import numpy as np
import itertools
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from enum import Enum
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import scipy.stats as scipy_stats
import pickle
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import halfnorm
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from fastkde import fastKDE
import pylab as PP

def ap(rec, prec):
    """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
    #if use_07_metric:
        # 11 point metric
    #    ap = 0.
    #    for t in np.arange(0., 1.1, 0.1):
    #        if np.sum(rec >= t) == 0:
    #            p = 0
    #        else:
    #            p = np.max(prec[rec >= t])
    #        ap = ap + p / 11.
    #else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope (going backwards, precision will always increase as sorted by -confidence)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def count_cadc_npos(gt_path,df):
    """
    Function to calculate total number of ground truths from kitti dataset
    Use the split val.txt to index label files. In each label file count instances of "car" detections
    args: val_split_file, gt_path, dataframe
    returns: npos (num ground truths)
    """
    #npos += cadc_label_npos(gt_path,stripped_line)
    det_idx = 0
    npos = np.zeros(3)
    labels = os.listdir(gt_path)
    frame_idx = np.asarray(df['frame_idx'])
    unique_idx = np.unique(np.sort(frame_idx), axis=0)

    for label in labels:
        label_int = int(label.replace('.txt',''))
        if det_idx == len(unique_idx):
            break
        if (unique_idx[det_idx] == label_int):
            npos += cadc_label_npos(gt_path,label.replace('.txt',''))
            det_idx += 1
    
    return npos

def cadc_label_npos(path,filename):
    """
    Function to count each car ground truth inside a label file. filepath comes from
    kitti split file.
    args: path, filename
    returns: count (npos instance inside 1 file)

    Difficulty calculation was taken from kitt_lidb.py
    trunc = float(label_arr[1])
    occ   = int(label_arr[2])
    y1 = float(label_arr[5])
    y2 = float(label_arr[7])
    BBGT_HEIGHT = y2 - y1
    """

    count = np.zeros(3)
    label_file = os.path.join(path,filename+'.txt')
    with open(label_file,'r') as file:
        for line in file:
            line_arr = line.split(' ')
            if (line[0:2] == 'Ca'):
                count[0:3] += 1
    return count

def count_kitti_npos(gt_path,df):
    """
    Function to calculate total number of ground truths from kitti dataset
    Use the split val.txt to index label files. In each label file count instances of "car" detections
    args: val_split_file, gt_path, dataframe
    returns: npos (num ground truths)
    """
    #npos += kitti_label_npos(gt_path,stripped_line)
    det_idx = 0
    npos = np.zeros(3)
    labels = []
    frame_idx = np.asarray(df['frame_idx'])
    idx = frame_idx  
    idx_sorted = np.sort(idx)
    unique_idx = np.unique(idx_sorted, axis=0)
    val_split_file = os.path.join(gt_path,'splits','val.txt')
    with open(val_split_file, "r") as file:
        for line in file:
            labels.append(line.strip())

    for label in labels:
        label_int = int(label)
        if det_idx == len(unique_idx):
            break
        if (unique_idx[det_idx] == label_int):
            npos += kitti_label_npos(gt_path,label)
            det_idx += 1
    
    return npos

def kitti_label_npos(path,filename):
    """
    Function to count each car ground truth inside a label file. filepath comes from
    kitti split file.
    args: path, filename
    returns: count (npos instance inside 1 file)

    Difficulty calculation was taken from kitt_lidb.py
    trunc = float(label_arr[1])
    occ   = int(label_arr[2])
    y1 = float(label_arr[5])
    y2 = float(label_arr[7])
    BBGT_HEIGHT = y2 - y1
    """

    count = np.zeros(3)
    label_file = os.path.join(path,filename+'.txt')
    with open(label_file,'r') as file:
        for line in file:
            line_arr = line.split(' ')
            if (line[0:2] == 'Ca'):

                trunc = float(line_arr[1])
                occ = float(line_arr[2])
                BBGT_HEIGHT = float(line_arr[7]) - float(line_arr[5])

                if(occ <= 0 and trunc <= 0.15 and (BBGT_HEIGHT) >= 40):  # difficulty 0
                    count[0:3] += 1
                elif(occ <= 1 and trunc <= 0.3 and (BBGT_HEIGHT) >= 25):  # difficulty 1
                    count[1:3] += 1
                elif(occ <= 2 and trunc <= 0.5 and (BBGT_HEIGHT) >= 25):  # difficulty 2
                    count[2] += 1
    return count

def unique(list):
    unique_list = []
    for x in list:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def count_npos(gt_file, df):
    """
    Function to calculate the total number of ground truths from a dataset. 
    npos = tp + fn (used in AP calculation)
    args: detection file, df 
    returns: npos 
    """
    with open(gt_file,'r') as labels_file:
        labels   = json.loads(labels_file.read())
    npos = np.zeros(3)
    gt_idx = 0
    det_idx = 0
    frame_idx = np.asarray(df['frame_idx'])
    scene_idx = np.asarray(df['scene_idx']) * 1000
    idx = (frame_idx + scene_idx)
    unique_idx = np.unique(idx, axis=0)
    idx_sorted = np.sort(unique_idx)
    # loop through each idx and find each gt in the gt file
    for label in labels:
        gt_idx = int(label['assoc_frame'])
        if det_idx == len(unique_idx):
            break
        if (unique_idx[det_idx] == gt_idx):
            for class_type, difficulty in zip(label['class'], label['difficulty']):
                if (class_type == 1):
                    if difficulty == 0:
                        npos[0:3] += 1
                    elif difficulty == 1:
                        npos[1:3] += 1
                    elif difficulty == 2:
                        npos[2] += 1
        det_idx += 1
    return npos

def calculate_ap(df,d_levels,gt_path=None,gt_file=None):
    """
    Function to calulate average precision by using tp and fp from dataframe. 
    tp when an associated ground truth exists. if no bbgt then fp. 
    args: df
    returns: mean_recall, mean_precision, mean_ap (average precision 0-1) for all difficulty levels
    """
    tp = np.zeros((len(df),d_levels))
    fp = np.zeros((len(df),d_levels))
    map = np.zeros((d_levels,))
    mrec = np.zeros((d_levels,))
    mprec = np.zeros((d_levels,))

    if 'cadc' in gt_path:
        npos = count_cadc_npos(gt_path,df)
    elif 'kitti' in gt_path:  # detection_file still in scope
        npos = count_kitti_npos(gt_path,df)
    else:
        npos = count_npos(gt_file,df)
    df_sorted = df.sort_values(by='confidence',ascending=False)

    for index, row in df_sorted.iterrows():
        # loop through each detection and mark as either tp or fp 
        if (row['difficulty'] == -1):
            fp[index,0] += 1
            fp[index,1] += 1
            fp[index,2] += 1
            continue
        if (row['difficulty'] <= 2):
            tp[index,2] += 1
        if (row['difficulty'] <= 1):
            tp[index,1] += 1
        if (row['difficulty'] <= 0):
            tp[index,0] += 1
            
    tp_sum = np.cumsum(tp,axis=0)
    fp_sum = np.cumsum(fp,axis=0)
    for i in range(0,d_levels):
        rec = tp_sum[:,i] / npos[i]
        prec = tp_sum[:,i] / np.maximum(tp_sum[:,i] + fp_sum[:,i], np.finfo(np.float64).eps) 

        rec, prec = zip(*sorted(zip(rec, prec)))
        mprec[i]  = np.average(prec)
        mrec[i] = np.average(rec)
        map[i] = ap(rec, prec)
        plt.plot(rec,prec)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.legend()
        plt.show()
    print(map)
    return mrec,mprec,map
