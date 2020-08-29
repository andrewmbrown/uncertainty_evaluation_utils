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

def parse_dets(det_file):
    if ("3d" in detection_file):  # is the file from lidar or img domain
        lidar_flag = True
    else:
        lidar_flag = False
    data_rows     = []
    column_names  = ['ID']
    int_en        = False
    skip_cols     = 0
    for i, line in enumerate(det_file):
        line = line.replace('bbdet', ' bbdet')
        line = line.replace('\n','').split(' ')
        row = []
        row.append(i)
        for j,elmnt in enumerate(line):
            if(skip_cols == 0):
                if(isfloat(elmnt)):
                    if(int_en):
                        row.append(int(elmnt))
                    else:
                        row.append(float(elmnt))
                else:
                    col = elmnt.replace(':','')
                    if (lidar_flag and col == 'bbgt3d'):  # eventually bbgt from lidar script becomes bbdgt3d. 
                                                          # Also, bbdet will become bbdet3d
                        col = 'bbgt'
                    #Override for track idx as it has characters, override to save as integer when needed
                    int_en   = False
                    if('track' in col):
                        row.append(line[j+1])
                        skip_cols = 1
                    elif('idx' in col or 'pts' in col or 'difficulty' in col):
                        int_en = True
                    elif(lidar_flag and 'bb' in col):
                        row.append([float(line[j+1]),float(line[j+2]),float(line[j+3]),float(line[j+4]),float(line[j+5]),float(line[j+6]),float(line[j+7])])
                        skip_cols = 7
                    elif('cls_var' in col):
                        row.append([float(line[j+1]),float(line[j+2])])
                        skip_cols = 2
                    elif('bb' in col and '3d' not in col):
                        row.append([float(line[j+1]),float(line[j+2]),float(line[j+3]),float(line[j+4])])
                        skip_cols = 4
                    elif('bb' in col and '3d' in col):
                        row.append([float(line[j+1]),float(line[j+2]),float(line[j+3]),float(line[j+4]),float(line[j+5]),float(line[j+6]),float(line[j+7])])
                        skip_cols = 7
                    if(col not in column_names and col != ''):
                        column_names.append(col)
            else:
                skip_cols = skip_cols - 1
        data_rows.append(row)
    df = pd.DataFrame(data_rows,columns=column_names)
    df.set_index('ID')
    return df

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False


def parse_labels(dets_df, gt_file):
    with open(gt_file,'r') as labels_file:
        labels   = json.loads(labels_file.read())
    scene_name_dict = {}
    weather_dict    = {}
    tod_dict        = {}
    calibration_dict = {}
    meta_dict = {}
    for label in labels:
        scene_type = label['scene_type'][0]
        tod = scene_type['tod']
        weather = scene_type['weather']
        scene_name = label['scene_name']
        assoc_frame  = label['assoc_frame']
        scene_idx  = int(int(label['assoc_frame'])/1000)
        calibration = label['calibration']  # for transforming 3d to 2d

        if(scene_idx not in scene_name_dict.keys()):
            scene_name_dict[scene_idx] = scene_name
            weather_dict[scene_idx]    = weather
            tod_dict[scene_idx]        = tod
            calibration_dict[scene_idx] = calibration
    full_df = dets_df
    full_df['scene_name'] = full_df['scene_idx'].map(scene_name_dict)
    full_df['weather'] = full_df['scene_idx'].map(weather_dict)
    full_df['tod'] = full_df['scene_idx'].map(tod_dict)
    full_df['calibration'] = full_df['scene_idx'].map(calibration_dict)
    return full_df

'''
monte carlo sampler 3D to 2D xyxy
takes a set of 3D bounding boxes and uncertainty input, samples the distribution 10000 times and then transforms into a new domain.
uncertainty is re-extracted after domain shift
'''

'''
plot scatter uncertainty 2D
Should be able to plot uncertainty (bbox or classification) defined in argument list as a function of a tuple pair of ordinal names.
'''

'''
plot scatter uncertainty
Should be able to plot uncertainty (bbox or classification) defined in argument list as a function of one ordinal, but multiple plots are allowed on one graph if multiple ordinal_names are specified
'''


'''
draw_combined_labels
Should read in an image based on the assoc_frame value of the labels, transform the lidar bbox to 8 pt image domain and draw, along with 2d image labels
'''

'''
associate_detections
combine both image and lidar result set to create one large panda db.
Detections without a corresponding ground truth are attempted to be matched with remaining bboxes via IoU
Detections that cannot be matched are removed
Additional columns to be added will be the transformed xyxy uncertainty values as well as the transformed 2D bbox.
'''

'''
plot scatter uncertainty per track
Same as above, but connect dots for individual track ID's. Ideally use a rainbow heatmap based on avg uncertainty
'''

#def plot_scatter_uc(dets,uc_type,scene_type,ordinal_names,max_val)

def extract_bad_predictions(df,confidence):
    """
    Filter low confidence predictions (filter out good predictions) and return 
    a new df. To be used with drawing and scatter plots.
    args: df, confidence(0-1,3.f)
    return: filtered df
    """
    #confidence_idx = df['confidence']>confidence  # everything below the confidence threshold
    #filtered_df = df[confidence_idx]
    filtered_df = df.loc[df['confidence']<confidence]
    return filtered_df

def extract_good_predictions(df,confidence):
    """
    Filter high confidence predictions (filter out bad predictions) and return 
    a new df. To be used with drawing and scatter plots.
    args: df, confidence(0-1,3.f)
    return: filtered df
    """
    #confidence_idx = df['confidence']<confidence  # everything above the confidence threshold
    #filtered_df = df[confidence_idx]
    filtered_df = df.loc[df['confidence']>confidence]
    return filtered_df
    
def draw_filtered_detections(df):
    """
    Draw filtered detections from dataframe
    open image, draw bbdets (for all dets in frame), repeat
    """
    frame_idx = np.asarray(df['frame_idx'])
    scene_idx = np.asarray(df['scene_idx'])
    pic_idx = (frame_idx + scene_idx) * 1000
    df.insert(len(df.columns),'pic_idx',pic_idx)  # insert picture values into df
    df = df.sort_values(by=['pic_idx'])  # sorting to group detections on SAME imgs (efficient iterating)
    
    # locate column indexs needed based on name 
    pic_column = df.columns.get_loc("pic_idx")
    bbdets_column = df.columns.get_loc("bbdet3d_2d")
    # convert dataframe to numpy arr for interating 
    data = df.values
    #data[:,pic_column] = np.char.zfill(data[:,pic_column],7)
    idx = data[0,pic_column]
    # open image
    img_data = Image.open(imgpath + (str(idx)).zfill(7) + '.png')
    draw = ImageDraw.Draw(img_data)
    for row in data:  # iterate through each detection
        current_idx = row[pic_column]  # either current pic or move to next pic
        if (current_idx == idx):  # string comparison, but is ok
            if (row[bbdets_column] == [-1,-1,-1,-1]):  # no detection
                continue
            draw.rectangle(row[bbdets_column])  # draw rectangle over jpeg
        else:
            out_file = os.path.join(savepath,(str(idx)).zfill(7))  # all detections drawn for pic, save and next
            img_data.save(out_file,'PNG')
            img_data = Image.open(imgpath + (str(current_idx)).zfill(7) + '.png')
            draw = ImageDraw.Draw(img_data)
            if (row[bbdets_column] == [-1,-1,-1,-1]):
                idx = current_idx  # update idx
                continue
            draw.rectangle(row[bbdets_column])  # must draw as for loop will iterate off detection
            idx = current_idx  # update idx

def get_df(cache_dir,det_path):
        """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
        #for line in traceback.format_stack():
        #    print(line.strip())
        det_file_name = os.path.basename(det_path).replace('.txt','')
        cache_file = os.path.join(cache_dir,det_file_name+'_df.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    df = pickle.load(fid)
                except:
                    df = pickle.load(fid, encoding='bytes')
            print('df loaded from {}'.format(cache_file))
            return df

        df = []
        with open(detection_file) as det_file:
            dets_df  = parse_dets(det_file.readlines())
        if ("kitti" not in det_path):
            df  = parse_labels(dets_df, gt_file)
        else:
            df = dets_df

        with open(cache_file, 'wb') as fid:
            pickle.dump(df, fid, pickle.HIGHEST_PROTOCOL)
        print('df wrote to {}'.format(cache_file))

        return df

mypath = '/home/andrew'
imgpath = '/home/andrew/data2/waymo/val/images/'  # can join these later for completion
savepath = '/home/andrew/2d_uncertainty_drawn/'
splitpath = '/home/andrew/faster_rcnn_pytorch_multimodal/tools/splits/'
#detection_file = os.path.join(mypath,'faster_rcnn_pytorch_multimodal','tools','results','vehicle.car_detection_results_simple.txt')
detection_file = os.path.join(mypath,'faster_rcnn_pytorch_multimodal','tools','results','image_pow2.txt')
#detection_file = os.path.join(mypath,'faster_rcnn_pytorch_multimodal','tools','results_kitti','image.txt')
#detection_file = os.path.join(mypath,'faster_rcnn_pytorch_multimodal','tools','results_kitti','kitti_200.txt')
gt_path = os.path.join(mypath,'data','datasets','kitti','training','label_2')
val_split_file = os.path.join(splitpath,'val.txt')  # from kitti validation

cache_dir = os.path.join(mypath,'cache_dir')
#detection_file = os.path.join(mypath,'faster_rcnn_pytorch_multimodal','tools','results','lidar_3d_iou_uncertainty_results.txt')
gt_file        = os.path.join(mypath,'labels_full_new','image_labels.json')
#column_names = ['assoc_frame','scene_idx','frame_idx','bbdet','a_cls_var','a_cls_entropy','a_cls_mutual_info','e_cls_entropy','e_cls_mutual_info','a_bbox_var','e_bbox_var','track_idx','difficulty','pts','cls_idx','bbgt']
num_scenes = 210
top_crop = 300
bot_crop = 30

if __name__ == '__main__': 
    df = get_df(cache_dir,detection_file)
    #(mrec,prec,map) = calculate_ap(df,3,gt_path,gt_file)  # 2nd value is # of difficulty types
    #df = bbdet3d_to_bbdet2d(df,top_crop)
    print(df)
    df = df.loc[df['difficulty'] != -1]
    #df   = df.loc[df['confidence'] > 0.9]
    night_dets = df.loc[df['tod'] == 'Night']
    day_dets = df.loc[df['tod'] == 'Day']
    rain_dets = df.loc[df['weather'] == 'rain']
    sun_dets = df.loc[df['weather'] == 'sunny']
    scene_dets = df.loc[df['scene_idx'] == 168]
    diff1_dets = df.loc[df['difficulty'] != 2]
    diff2_dets = df.loc[df['difficulty'] == 2]
    minm = 0
    maxm = .02
    
    """
    2d - x_c, y_c, l1, w1
    3d - x_c, y_c, z_c, l2, w2, h, r_y
    """
    #plot_histo_inverse_gaussian(df,'scene','a_bbox_var','x1')
    #plot_histo_KDE(df,'scene','a_bbox_var','x1', 0)
    vals = ['x_c','y_c']
    plot_histo_multivariate_KDE(df,'scene','a_bbox_var',vals)
    #dets,scene,col,val,min_val=None,max_val=None
    # scene_data = plot_histo_bbox_uc(scene_dets,'scene',minm,maxm)
    #night_data = plot_histo_bbox_uc(night_dets,'night',minm,maxm)
    # day_data   = plot_histo_bbox_uc(day_dets,'day',minm,maxm)
    #day_mean = np.mean(day_dets)
    # day_mean = np.mean(day_data)
    #something = plot_histo_poisson(night_dets,'night',minm,maxm)
    #something = plot_histo_chi_squared(night_dets,'night',minm,maxm)
    #something = plot_histo_lognorm(df,df,'all_detections',minm,maxm)  # 
    #something = plot_histo_half_gaussian(df,df,'valid_predictions',minm,maxm)
    #data, mu, sigma = plot_histo_gaussian(df,df,'valid_predictions',minm,maxm)
    #data, mu, sigma = plot_histo_lognorm(df,df,'valid_predictions',minm,maxm)  # 
    #something = plot_histo_log(df,'log_valid_predictions',minm,maxm)  # need mu and sigma
    #plot_scatter_var(df,'bbdet','a_bbox_var')
    #confidence = 0.9
    #filtered_df = extract_bad_predictions(df,confidence)
    #filtered_df = extract_good_predictions(df,0.7)
    #filtered_df = extract_twosigma(filtered_df,1)  # none is high var, 1 is low var
    #draw_filtered_detections(filtered_df)
    #print(len(night_data))
    # print(len(day_data))
    #r = scipy_stats.poisson.rvs(day_mean)
    # result = scipy_stats.ks_2samp(day_data,scene_data)
    # print(result)
    #plot_histo_bbox_uc(rain_dets,'rain',minm,maxm)
    #plot_histo_cls_uc(night_dets,'night',minm,maxm)
    #plot_histo_cls_uc(day_dets,'day',minm,maxm)
    #plot_histo_cls_uc(rain_dets,'rain',minm,maxm)
    #plot_histo_cls_uc(sun_dets,'sunny',minm,maxm)
    #plot_histo_bbox_uc(diff2_dets,'lvl2',minm,maxm)
    #plot_histo_bbox_uc(diff1_dets,'lvl1',minm,maxm)
    #print('mu = ',mu,'sigma = ',sigma)
    plt.legend()
    plt.show()
    #print(day_dets)
    #print(night_dets)
    #print(rain_dets)