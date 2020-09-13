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

def count_cadc_npos(df,data_dir,sensor_type):
    """
    Function to calculate total number of ground truths from kitti dataset
    Use the split val.txt to index label files. In each label file count instances of "car" detections
    args: val_split_file, gt_path, dataframe
    returns: npos (num ground truths)
    """
    #npos += cadc_label_npos(gt_path,stripped_line)
    det_idx = 0
    npos = np.zeros(3)
    cadc_gt_path = os.path.join(data_dir,'val','annotation_00')
    labels = os.listdir(cadc_gt_path)
    frame_idx = np.asarray(df['frame_idx'])
    unique_idx = np.unique(np.sort(frame_idx), axis=0)
    label_idx = []
    for label in labels:
        label_idx.append(int(label.replace('.txt','')))
    label_idx = np.sort(np.asarray(label_idx))
    for i, label in enumerate(label_idx):
        if det_idx == len(unique_idx):
            break
        if (unique_idx[det_idx] == label):
            npos += cadc_label_npos(cadc_gt_path,labels[i])
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
    label_file = os.path.join(path,filename)
    with open(label_file,'r') as file:
        for line in file:
            line_arr = line.split(' ')
            if (line[0:2] == 'Ca'):
                count[0:3] += 1
    return count

def count_kitti_npos(df,data_dir,sensor_type):
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
    kitti_gt_path = os.path.join(data_dir,'training','label_2')
    val_split_file = os.path.join(data_dir,'splits','val.txt')
    with open(val_split_file, "r") as file:
        for line in file:
            labels.append(line.strip())

    for label in labels:
        label_int = int(label)
        if det_idx == len(unique_idx):
            break
        if (unique_idx[det_idx] == label_int):
            npos += kitti_label_npos(kitti_gt_path,label)
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

def count_npos(df,data_dir,sensor_type):
    """
    Function to calculate the total number of ground truths from a dataset. 
    npos = tp + fn (used in AP calculation)
    args: detection file, df 
    returns: npos 
    """
    gt_file = os.path.join(data_dir,'val','labels','{}_labels.json'.format(sensor_type))
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

def calculate_ap(df,d_levels,data_dir,sensor_type,plot=False):
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

    if 'cadc' in data_dir:
        npos = count_cadc_npos(df,data_dir,sensor_type)
    elif 'kitti' in data_dir:  # detection_file still in scope
        npos = count_kitti_npos(df,data_dir,sensor_type)
    else:
        npos = count_npos(df,data_dir,sensor_type)
    df_sorted = df.sort_values(by='confidence',ascending=False)
    for i in range(0,len(df_sorted.index)):
        row = df_sorted.iloc[i]
        # loop through each detection and mark as either tp or fp 
        if (row['fp'] == 1):
            fp[i,0] += 1
            fp[i,1] += 1
            fp[i,2] += 1
            continue
        if (row['difficulty'] <= 2):
            tp[i,2] += 1
        if (row['difficulty'] <= 1):
            tp[i,1] += 1
        if (row['difficulty'] <= 0):
            tp[i,0] += 1
            
    tp_sum = np.cumsum(tp,axis=0)
    fp_sum = np.cumsum(fp,axis=0)
    for i in range(0,d_levels):
        rec = tp_sum[:,i] / npos[i]
        prec = tp_sum[:,i] / np.maximum(tp_sum[:,i] + fp_sum[:,i], np.finfo(np.float64).eps) 

        rec, prec = zip(*sorted(zip(rec, prec)))
        mprec[i]  = np.average(prec)
        mrec[i] = np.average(rec)
        map[i] = ap(rec, prec)
        if(plot):
            plt.plot(rec,prec)
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.legend()
            plt.show()
    print(map)
    return mrec,mprec,map


if __name__ == '__main__':
    print('cannot run file stand-alone')