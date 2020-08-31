import sys
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
import pylab as PP
import argparse
import csv
import ap_utils
import modelling_utils
import transform_utils

def parse_args(manual_mode):
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Evaluate uncertainty')
    parser.add_argument(
        '--sensor_type',
        dest='sensor_type',
        help='type of sensor being analyzed',
        default=None,
        type=str)
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='dataset chosen for analysis',
        default=None,
        type=str)
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='root location of all datasets',
        default=None,
        type=str)
    parser.add_argument(
        '--cache_dir',
        dest='cache_dir',
        help='location of cache directory',
        default=None,
        type=str)
    parser.add_argument(
        '--out_dir',
        dest='out_dir',
        help='location of output dir',
        default='/',
        type=str)
    parser.add_argument(
        '--det_file_1',
        dest='det_file_1',
        help='location of first detection file',
        default=None,
        type=str)
    parser.add_argument(
        '--det_file_2',
        dest='det_file_2',
        help='location of second detection file',
        default=None,
        type=str)

    if len(sys.argv) == 1 and manual_mode is False:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    print(args)
    return args

def parse_dets(det_file,sensor_type):
    if (sensor_type == 'lidar'):  # is the file from lidar or img domain
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
        scene_name   = label['scene_name']
        assoc_frame  = label['assoc_frame']
        scene_idx    = int(int(label['assoc_frame'])/1000)
        calibration  = label['calibration']  # for transforming 3d to 2d

        if(scene_idx not in scene_name_dict.keys()):
            scene_name_dict[scene_idx]  = scene_name
            weather_dict[scene_idx]     = weather
            tod_dict[scene_idx]         = tod
            calibration_dict[scene_idx] = calibration
    full_df                = dets_df
    full_df['scene_name']  = full_df['scene_idx'].map(scene_name_dict)
    full_df['weather']     = full_df['scene_idx'].map(weather_dict)
    full_df['tod']         = full_df['scene_idx'].map(tod_dict)
    full_df['calibration'] = full_df['scene_idx'].map(calibration_dict)
    return full_df

def cadc_parse_labels(dets_df, scene_file):
    drive_dir       = ['2018_03_06','2018_03_07','2019_02_27']
    #2018_03_06
    # Seq | Snow  | Road Cover | Lens Cover
    #   1 | None  |     N      |     N
    #   5 | Med   |     N      |     Y
    #   6 | Heavy |     N      |     Y
    #   9 | Light |     N      |     Y
    #  18 | Light |     N      |     N

    #2018_03_07
    # Seq | Snow  | Road Cover | Lens Cover
    #   1 | Heavy |     N      |     Y
    #   4 | Light |     N      |     N
    #   5 | Light |     N      |     Y

    #2019_02_27
    # Seq | Snow  | Road Cover | Lens Cover
    #   5 | Light |     Y      |     N
    #   6 | Heavy |     Y      |     N
    #  15 | Med   |     Y      |     N
    #  28 | Light |     Y      |     N
    #  37 | Extr  |     Y      |     N
    #  46 | Extr  |     Y      |     N
    #  59 | Med   |     Y      |     N
    #  73 | Light |     Y      |     N
    #  75 | Med   |     Y      |     N
    #  80 | Heavy |     Y      |     N

    with open(scene_file,'r') as csvfile:
        scene_meta = {}
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(reader):
            for j, elem in enumerate(row):
                if(i == 0):
                    elem_t = elem.replace(' ','_').lower()
                    scene_meta[elem_t] = []
                else:
                    for k, (key,_) in enumerate(scene_meta.items()):
                        if j == k:
                            scene_meta[key].append(elem)
    scene_column = np.asarray(dets_df['frame_idx'],dtype=np.int32)/100
    scene_column = scene_column.astype(dtype=np.int32)
    dets_df['scene_idx'] = scene_column
    snow_dict = {}
    dates = scene_meta['date']
    scenes = scene_meta['number']
    lens_snow_cover = scene_meta['cam_00_lens_snow_cover']
    road_snow_cover = scene_meta['road_snow_cover']
    snow_pts_rem    = scene_meta['snow_points_removed']
    snow_level = []
    scene_idx  = []
    for i, snow_tmp in enumerate(snow_pts_rem):
        date = dates[i]
        snow_tmp = int(snow_tmp)
        if(snow_tmp < 25):
            scene_desc = 'none'
        elif(snow_tmp < 250):
            scene_desc = 'light'
        elif(snow_tmp < 500):
            scene_desc = 'medium'
        elif(snow_tmp < 750):
            scene_desc = 'heavy'
        else:
            scene_desc = 'extreme'
        scene_idx.append(drive_dir.index(date)*100 + int(scenes[i]))
        snow_level.append(scene_desc)
    snow_dict = dict(zip(scene_idx,snow_level))
    dets_df['lens_snow_cover'] = dets_df['scene_idx'].map(dict(zip(scene_idx,lens_snow_cover)))
    dets_df['road_snow_cover'] = dets_df['scene_idx'].map(dict(zip(scene_idx,road_snow_cover)))
    dets_df['snow_level'] = dets_df['scene_idx'].map(snow_dict)
    return dets_df
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
    
def draw_filtered_detections(df,out_dir,data_dir):
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
    img_data = Image.open(data_dir + (str(idx)).zfill(7) + '.png')
    draw = ImageDraw.Draw(img_data)
    for row in data:  # iterate through each detection
        current_idx = row[pic_column]  # either current pic or move to next pic
        if (current_idx == idx):  # string comparison, but is ok
            if (row[bbdets_column] == [-1,-1,-1,-1]):  # no detection
                continue
            draw.rectangle(row[bbdets_column])  # draw rectangle over jpeg
        else:
            out_file = os.path.join(out_dir,(str(idx)).zfill(7))  # all detections drawn for pic, save and next
            img_data.save(out_file,'PNG')
            img_data = Image.open(data_dir + (str(current_idx)).zfill(7) + '.png')
            draw = ImageDraw.Draw(img_data)
            if (row[bbdets_column] == [-1,-1,-1,-1]):
                idx = current_idx  # update idx
                continue
            draw.rectangle(row[bbdets_column])  # must draw as for loop will iterate off detection
            idx = current_idx  # update idx

def get_df(dataset,cache_dir,dets_file,data_dir,sensor_type,limiter=0):
        """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
        #for line in traceback.format_stack():
        #    print(line.strip())
        det_file_name = os.path.basename(dets_file).replace('.txt','')
        cache_filename = '{}_{}_{}_df.pkl'.format(sensor_type,dataset,det_file_name)
        cache_file = os.path.join(cache_dir,cache_filename)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    df = pickle.load(fid)
                except:
                    df = pickle.load(fid, encoding='bytes')
            print('df loaded from {}'.format(cache_file))
        else:
            df = []
            with open(dets_file) as det_file:
                dets_df  = parse_dets(det_file.readlines(),sensor_type)
            if (dataset == 'kitti'):
                df = dets_df
            elif(dataset == 'cadc'):
                scene_file = os.path.join(data_dir,'cadc_scene_description.csv')
                df = cadc_parse_labels(dets_df,scene_file)
            else:
                df = parse_labels(dets_df, gt_file)

            with open(cache_file, 'wb') as fid:
                pickle.dump(df, fid, pickle.HIGHEST_PROTOCOL)
            print('df wrote to {}'.format(cache_file))
        if(limiter != 0):
            print(len(df.index))
            frac = (limiter+0.1)/(len(df.index)+0.1)
            df = df.sample(frac=frac)
            print(len(df.index))
        return df

"""
2d - x_c, y_c, l1, w1
3d - x_c, y_c, z_c, l2, w2, h, r_y
"""
if __name__ == '__main__': 
    manual_mode = True
    args = parse_args(manual_mode)
    #-------------------------
    # Manual args
    #-------------------------
    if(manual_mode):
        args.root_dir    = os.path.join('/home','mat','thesis')
        args.sensor_type = 'image'
        args.dataset     = 'waymo'
        #KITTI LIDAR
        #args.det_file_1  = os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+aug_a_e_uc_4','test_results','results.txt')
        #KITTI IMAGE
        #args.det_file_1  = os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+aug_a_e_uc','test_results','results.txt')
        #WAYMO IMAGE
        args.det_file_1  = os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+aug_a_e_uc','test_results_2','results.txt')
        args.out_dir     = os.path.join(args.root_dir,'eval_out')
        args.cache_dir   = os.path.join(args.root_dir,'eval_cache')
        args.data_dir    = os.path.join(args.root_dir,'data2')
    num_scenes = 210
    top_crop = 300
    bot_crop = 30
    data_dir = os.path.join(args.data_dir,args.dataset)

    #-------------------------
    # Init
    #-------------------------
    if(args.dataset == 'waymo'):
        gt_file = os.path.join(data_dir,'val','labels','{}_labels.json'.format(args.sensor_type))
    else:
        gt_file = ''
    df = get_df(args.dataset,args.cache_dir,args.det_file_1,data_dir,args.sensor_type,limiter=0)

    #-------------------------
    # Example filters for data
    #-------------------------
    #df    = df.loc[df['confidence'] > 0.5]
    #df   = df.loc[df['confidence'] > 0.9]
    #night_dets = df.loc[df['tod'] == 'Night']
    day_dets = df.loc[df['tod'] == 'Day']
    #rain_dets = df.loc[df['weather'] == 'rain']
    #sun_dets = df.loc[df['weather'] == 'sunny']
    #scene_dets = df.loc[df['scene_idx'] == 168]
    #diff1_dets = df.loc[df['difficulty'] != 2]
    #diff2_dets = df.loc[df['difficulty'] == 2]
    df_tp = df.loc[df['difficulty'] != -1]
    df_fp = df.loc[df['difficulty'] == -1]
    df_n  = df.loc[df['tod'] == 'Night']
    #-------------------------
    # Compute AP
    #-------------------------
    #(mrec,prec,map) = ap_utils.calculate_ap(day_dets,3,data_dir,args.sensor_type,plot=False)  # 2nd value is # of difficulty types
    #print(map)
    #df = bbdet3d_to_bbdet2d(df,top_crop)

    #------------------------
    # Multivariate KDE generation and ROC curve generation
    #------------------------
    #plot_histo_multivariate(df_tp,'TP','a_bbox_var',vals,plot=False)
    #plot_histo_multivariate(df_fp,'FP','a_bbox_var',vals,plot=False)

    #------------------------
    # Multivariate statistical testing
    #------------------------
    #vals = ['x_c','y_c','z_c','l2','w2','h','r_y']
    #vals  = ['x_c']
    #kstest   = modelling_utils.run_kstest(df_tp,df_fp,'a_bbox_var',vals,sum_vals=True)
    #kldiv = modelling_utils.run_kldiv(df_tp,df_fp,'a_bbox_var',vals,sum_vals=True)
    #jsdiv = modelling_utils.run_jsdiv(df_tp,df_fp,'a_bbox_var',vals,sum_vals=True)
    #print('kstest: {} kldiv: {} jsdiv {}'.format(kstest,kldiv,jsdiv))

    #------------------------
    # Multivariate KDE generation and ROC curve generation
    #------------------------
    """
    vals = ['x_c','y_c','z_c','l2','w2','h','r_y']
    m_kde_tp = modelling_utils.plot_histo_multivariate_KDE(df_tp,'TP','a_bbox_var',vals,plot=False)
    m_kde_fp = modelling_utils.plot_histo_multivariate_KDE(df_fp,'FP','a_bbox_var',vals,plot=False)
    modelling_utils.plot_roc_curves(df,'a_bbox_var',vals,m_kde_tp,m_kde_fp)
    #https://stats.stackexchange.com/questions/57885/how-to-interpret-p-value-of-kolmogorov-smirnov-test-python
    #Also should run KL and Wilcoxon
    vals = ['x_c','y_c','z_c','l2','w2','h','r_y']
    m_kde_tp = modelling_utils.plot_histo_multivariate_KDE(df_tp,'TP','e_bbox_var',vals,plot=False)
    m_kde_fp = modelling_utils.plot_histo_multivariate_KDE(df_fp,'FP','e_bbox_var',vals,plot=False)
    modelling_utils.plot_roc_curves(df,'e_bbox_var',vals,m_kde_tp,m_kde_fp)
    vals = ['car','bg']
    m_kde_tp = modelling_utils.plot_histo_multivariate_KDE(df_tp,'TP','e_cls_var',vals,plot=True)
    m_kde_fp = modelling_utils.plot_histo_multivariate_KDE(df_fp,'FP','e_cls_var',vals,plot=True)
    modelling_utils.plot_roc_curves(df,'e_cls_var',vals,m_kde_tp,m_kde_fp)
    #ratios = modelling_utils.find_ratios(df,'a_bbox_var',vals,m_kde_tp,m_kde_fp,min_thresh=1.0)
    #print('hit: {} miss: {} false_alarm: {} correct_rejection: {}'.format(ratios[0],ratios[1],ratios[2],ratios[3]))
    """

    #------------------------
    # Scatter plot generation
    #------------------------
    #vals = None
    #vals = ['l2','w2','h']
    #modelling_utils.plot_scatter_var(df,'distance','e_bbox_var',vals,swap=True)
    
    #------------------------
    # Box plot generation
    #------------------------
    #vals = None
    #modelling_utils.plot_box_plot(df,'a_bbox_var',vals)


    #------------------------
    # Mat's Custom Script
    #------------------------
    #param  = 'all_var'
    #param = 'e_bbox_var,a_bbox_var'
    #param  = 'e_cls_var,a_cls_var'
    param = 'a_bbox_var'
    #vals = ['w1']
    #vals  = ['l1','w1']
    #vals = ['fg','bg']
    vals = ['x_c','y_c']
    #vals = ['w1']
    #vals = ['x_c','y_c','z_c','l2','w2','h','r_y']
    #vals  = ['bg']
    #vals = ['l2','w2','h']
    #Multi-UC val lists
    #vals = ['e_x_c','e_y_c','e_z_c','e_l2','e_w2','e_h','e_r_y','e_fg','e_bg','a_x_c','a_y_c','a_z_c','a_l2','a_w2','a_h','a_r_y','a_fg','a_bg']
    #vals = ['e_fg','e_bg','a_fg','a_bg']
    #vals = ['e_x_c','e_y_c','e_z_c','e_l2','e_w2','e_h','e_r_y','a_x_c','a_y_c','a_z_c','a_l2','a_w2','a_h','a_r_y']
    #vals = ['e_x_c','e_y_c','e_l','e_w','a_x_c','a_y_c','a_l','a_w','e_fg','e_bg','a_fg','a_bg']
    #kstest   = modelling_utils.run_kstest(df_tp,df_fp,param,vals,sum_vals=True)
    is_summed = False
    kldiv = modelling_utils.run_kldiv(df_tp,df_fp,param,vals,sum_vals=is_summed)
    print('kldiv: {} summed: {}'.format(kldiv,is_summed))
    #jsdiv = modelling_utils.run_jsdiv(df_tp,df_fp,param,vals,sum_vals=False)
    #print('kstest: {} kldiv: {} jsdiv {}'.format(kstest,kldiv,np.sum(jsdiv)))
    #box_stats = modelling_utils.plot_box_plot(df,param,vals,plot=False)
    #print('box_plot: mean: {:.3f} median: {:.3f} [Q1,Q3]: [{:.3f},{:.3f}] [min,max]: [{:.3f},{:.3f}]'.format(box_stats[0],box_stats[1],box_stats[2],box_stats[3],box_stats[4],box_stats[5]))
    #plt.rcParams.update({'font.size': 16})
    m_kde_fp = modelling_utils.plot_histo_multivariate_KDE(df_fp,'FP',param,vals,min_val=0, plot=True)
    m_kde_tp = modelling_utils.plot_histo_multivariate_KDE(df_tp,'TP',param,vals,min_val=0, plot=True)

    #m_kde = modelling_utils.plot_histo_multivariate_KDE(df,'All',param,vals,min_val=0, plot=False)
    #modelling_utils.plot_roc_curves(df,param,vals,m_kde_tp,m_kde_fp,limiter=10000)
    #plt.legend()
    #plt.show()

    #-------------------------
    # Misc
    #-------------------------
    #plot_histo_multivariate_KDE(df,'scene','a_bbox_var',vals)
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
    #plt.legend()
    #plt.show()
