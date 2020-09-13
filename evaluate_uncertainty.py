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
        df = modelling_utils._apply_limiter(df, limiter)
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
        args.sensor_type = 'lidar'
        args.dataset     = 'waymo'
        #KITTI LIDAR
        #args.det_file_1  = os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+aug_a_e_uc_4','test_results','results.txt')
        #KITTI IMAGE
        #args.det_file_1  = os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+aug_a_e_uc','test_results','results.txt')
        #WAYMO IMAGE
        #args.det_file_1  = os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+aug_a_e_uc','test_results_2','results.txt')
        #WAYMO IMAGE DAY
        #args.det_file_1  = os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'day+aug_a_e_uc','results_day.txt')
        #WAYMO IMAGE DROPOUT
        #args.det_file_2  = os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+aug_a_e_uc','test_results_2','dropout_02','d_0_2_results.txt')
        #args.det_file_2  = os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+aug_a_e_uc','test_results_2','blur_4_0','blur_4_0_results.txt')

        #WAYMO LIDAR
        args.det_file_1  = os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+a_e_uc_3c','test_results','results.txt')
        #CADC IMAGE
        #args.det_file_1  = os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+aug_a_e_uc','test_results','results.txt')
        #args.det_file_1  = os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+aug_a_e_uc','test_results_2','results.txt')
        #args.det_file_1  = os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+aug_a_e_uc','test_results_3','results_3.txt')
        #CADC LIDAR
        #args.det_file_1  = os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+aug_a_e_uc_2','test_results','results.txt')
        args.out_dir     = os.path.join(args.root_dir,'eval_out')
        args.cache_dir   = os.path.join(args.root_dir,'eval_cache')
        args.data_dir    = os.path.join(args.root_dir,'data2')
    test_file_list = []
    #WAYMO IMAGE TEST FILES
    #test_file_list.append(os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+aug_a_e_uc','test_results_2','blur_4_0','blur_4_0_results.txt'))
    #test_file_list.append(os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+aug_a_e_uc','test_results_2','dropout_02','d_0_2_results.txt'))
    #test_file_list.append(os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+aug_a_e_uc','test_results_2','dropout_04','d_0_4_results.txt'))
    #test_file_list.append(os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+aug_a_e_uc','test_results_2','fog','fog_results.txt'))
    #test_file_list.append(os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+aug_a_e_uc','test_results_2','spatter','spatter_results.txt'))
    #WAYMO_LIDAR_TEST_FILES
    #test_file_list.append(os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+a_e_uc_3c','test_results','rain_1mm','rain_1mm_results.txt'))
    #test_file_list.append(os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+a_e_uc_3c','test_results','rain_3mm','rain_3mm_results.txt'))
    #test_file_list.append(os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+a_e_uc_3c','test_results','rain_5mm','rain_5mm_results.txt'))
    #test_file_list.append(os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+a_e_uc_3c','test_results','d_0_2_results','d_0_2_results.txt'))
    #test_file_list.append(os.path.join(args.root_dir,'faster_rcnn_pytorch_multimodal','final_releases',args.sensor_type,args.dataset,'base+a_e_uc_3c','test_results','d_0_4_results','d_0_4_results.txt'))
    num_scenes = 210
    top_crop = 300
    bot_crop = 30
    data_dir = os.path.join(args.data_dir,args.dataset)
    conf_thresh = 0.5
    #-------------------------
    # Init
    #-------------------------
    if(args.dataset == 'waymo'):
        gt_file = os.path.join(data_dir,'val','labels','{}_labels.json'.format(args.sensor_type))
    else:
        gt_file = ''
    df = get_df(args.dataset,args.cache_dir,args.det_file_1,data_dir,args.sensor_type,limiter=0)
    #if(args.det_file_2 is not None):
    #    df_t = get_df(args.dataset,args.cache_dir,args.det_file_2,data_dir,args.sensor_type,limiter=50000)
    #    df_t = df_t.loc[df_t['confidence'] > conf_thresh]
    #-------------------------
    # Example filters for data
    #-------------------------
    limiter = 100000
    df    = df.loc[df['confidence'] > conf_thresh]
    df_orig = df
    #heavy_extreme_snow = df.loc[((df['snow_level'] == 'extreme') | (df['snow_level'] == 'heavy') | (df['snow_level'] == 'medium'))]
    #none_light_snow = df.loc[((df['snow_level'] == 'none') | (df['snow_level'] == 'light'))]
    #df = df.loc[((df['snow_level'] == 'none') | (df['snow_level'] == 'light') | (df['snow_level'] == 'medium'))]
    #no_cover_df   = df.loc[(df['lens_snow_cover'] == 'None')]
    #snow_cover_df = df.loc[(df['lens_snow_cover'] == 'Partial')]
    #df = df.loc[(df['lens_snow_cover'] == 'None')]


    day_dets = df.loc[df['tod'] == 'Day']
    sun_dets = df.loc[df['weather'] == 'sunny']
    rain_dets = df.loc[df['weather'] == 'rain']
    night_dets  = df.loc[df['tod'] == 'Night']
    #scene_idx = np.unique(np.asarray(rain_dets['scene_idx'].to_list()))
    #scene_dets = df.loc[df['scene_idx'] == 168]
    #diff1_dets = df.loc[df['difficulty'] != 2]
    #diff2_dets = df.loc[df['difficulty'] == 2]
    df = modelling_utils._apply_limiter(df,limiter)
    df_tp = df.loc[df['difficulty'] != -1]
    df_fp = df.loc[df['difficulty'] == -1]
    #df_dd = df.loc[df['tod'] == 'Dawn/Dusk']

    #-------------------------
    # Compute AP
    #-------------------------
    
    #(mrec,prec,map) = ap_utils.calculate_ap(snow_cover_df,3,data_dir,args.sensor_type,plot=False)  # 2nd value is # of difficulty types
    #print(map)

    #df = bbdet3d_to_bbdet2d(df,top_crop)

    #-------------------------
    # Run Ratios/AP testing
    #-------------------------
    ap_testing_en = False
    if(ap_testing_en):
        param_list  = []
        vals_list   = []
        thresh_list = []
        param_list.append('a_bbox_var')
        param_list.append('e_bbox_var')
        param_list.append('a_cls_var')
        param_list.append('e_cls_var')
        #param_list.append('e_bbox_var,a_bbox_var')
        #param_list.append('e_cls_var,a_cls_var')
        param_list.append('all_var')

        #UC Metrics
        #vals_list.append(['x_c','y_c','l1','w1'])
        #vals_list.append(['x_c','y_c','l1','w1'])
        vals_list.append(['x_c','y_c','z_c','l2','w2','h','r_y'])
        vals_list.append(['x_c','y_c','z_c','l2','w2','h','r_y'])
        vals_list.append(['fg','bg'])
        vals_list.append(['fg','bg'])
        #vals_list.append(['e_x_c','e_y_c','e_l','e_w','a_x_c','a_y_c','a_l','a_w'])
        #vals_list.append(['e_x_c','e_y_c','e_z_c','e_l2','e_w2','e_h','e_r_y','a_x_c','a_y_c','a_z_c','a_l2','a_w2','a_h','a_r_y'])
        vals_list.append(['e_fg','e_bg','a_fg','a_bg'])
        #vals_list.append(['e_x_c','e_y_c','e_l','e_w','e_fg','e_bg','a_x_c','a_y_c','a_l','a_w','a_fg','a_bg'])
        vals_list.append(['e_x_c','e_y_c','e_z_c','e_l2','e_w2','e_h','e_r_y','e_fg','e_bg','a_x_c','a_y_c','a_z_c','a_l2','a_w2','a_h','a_r_y','a_fg','a_bg'])
        #Waymo IMG thresh list 0.7
        #thresh_list.append(-0.0055)
        #thresh_list.append(-0.0137)
        #thresh_list.append(-0.2842)
        #thresh_list.append(-0.0055)
        #thresh_list.append(-0.0004)
        #thresh_list.append(-0.0522)
        #thresh_list.append(0.00019)
        #Waymo IMG thresh list 0.5
        #thresh_list.append(-0.0066)
        #thresh_list.append(-0.0133)
        #thresh_list.append(-0.1481)
        #thresh_list.append(-0.0196)
        #thresh_list.append(-0.0014)
        #thresh_list.append(-0.0223)
        #thresh_list.append(0.00000)
        #Waymo IMG day thresh list
        
        #thresh_list.append(-0.0049)
        #thresh_list.append(-0.0177)
        #thresh_list.append(-0.2593)
        #thresh_list.append(0.05010)
        #thresh_list.append(-0.0004)
        #thresh_list.append(-0.0141)
        #thresh_list.append(0.00089)
        #Waymo LiDAR all thresh list
        thresh_list.append(0.00163)
        thresh_list.append(0.00018)
        thresh_list.append(-0.1012)
        thresh_list.append(0.09006)
        thresh_list.append(0.00000)
        thresh_list.append(0.00670)
        thresh_list.append(-0.0002)
        #(mrec,prec,map) = ap_utils.calculate_ap(df_test,3,data_dir,args.sensor_type,plot=False)  # 2nd value is # of difficulty types
        #print(map)
        df_test_list = []
        df_test_list.append(df)
        df_test_list.append(sun_dets)
        df_test_list.append(rain_dets)
        df_test_list.append(day_dets)
        df_test_list.append(night_dets)
        for test_file in test_file_list:
            df_t = get_df(args.dataset,args.cache_dir,test_file,data_dir,args.sensor_type,limiter=50000)
            df_t = df_t.loc[df_t['confidence'] > conf_thresh]
            df_test_list.append(df_t)
        min_anom_thresh = 0.01
        test_limiter = 20000
        kde_list = modelling_utils.cache_kde_models(df,df_tp,df_fp,param_list,vals_list,bins=200)
        for df_test in df_test_list:
            df_test = modelling_utils._apply_limiter(df_test,limiter=test_limiter)
            print('{}'.format('New test stimulus'))
            for i, param in enumerate(param_list):
                vals   = vals_list[i]
                thresh = thresh_list[i]
                m_kde_tp = kde_list[i][0]
                m_kde_fp = kde_list[i][1]
                m_kde    = kde_list[i][2]
                print('param: {} vals: {} thresh: {}'.format(param,vals,thresh))
                #m_kde_fp = modelling_utils.plot_histo_multivariate_KDE(df_fp,'FP',param,vals, plot=False, bins=200)
                #m_kde_tp = modelling_utils.plot_histo_multivariate_KDE(df_tp,'TP',param,vals, plot=False, bins=200)
                #m_kde    = modelling_utils.plot_histo_multivariate_KDE(df,'ALL',param,vals, plot=False, bins=200)
                signal_tp = np.where(np.asarray(df_test['difficulty'].to_list()) != -1, True, False)
                test_var = modelling_utils._col_data_extract(df_test,param,vals)
                tp_densities = m_kde_tp.pdf(test_var)
                fp_densities = m_kde_fp.pdf(test_var)
                all_densities = m_kde.pdf(test_var)
                response_tp = np.where(tp_densities > fp_densities + thresh,True,False)
                response_anomaly = np.where(all_densities < min_anom_thresh*np.max(all_densities),True,False)
                #ratios = modelling_utils.find_ratios(df_test,param,vals,signal_tp,tp_densities,fp_densities,min_thresh=thresh)
                total_resp = np.bitwise_and(np.bitwise_not(response_tp),response_anomaly)
                print('TP,FP: \n{}\n{}'.format(np.sum(response_tp),np.sum(np.bitwise_not(response_tp))))
                #print('Anom: \n{}\n{}'.format(np.sum(response_anomaly),np.sum(np.bitwise_not(response_anomaly))))
                #print('FP&Anom:\n{}\n{}'.format(np.sum(total_resp),np.sum(np.bitwise_not(total_resp))))
                #print('ratios: {}'.format(ratios))
                #print('ratio: {} det-TP: {} det-FP: {}'.format(np.sum(np.bitwise_not(response_tp))/np.sum(response_tp),np.sum(response_tp),np.sum(np.bitwise_not(response_tp))))
                #print('ratio: {} anomaly: {} norm: {}'.format(np.sum(response_anomaly)/np.sum(np.bitwise_not(response_anomaly)),np.sum(response_anomaly),np.sum(np.bitwise_not(response_anomaly))))
    
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
    '''
    #param  = 'all_var'
    #param = 'e_bbox_var,a_bbox_var'
    #param  = 'e_cls_var,a_cls_var'
    param = 'a_cls_var'
    #vals  = ['h']
    #vals = ['bg']
    #vals = ['w1']
    #vals = ['w1']
    #vals = ['x_c','y_c','z_c','l2','w2','h','r_y']
    vals  = ['fg','bg']
    #vals = ['l2','w2','h']
    #Multi-UC val lists
    #vals = ['e_x_c','e_y_c','e_z_c','e_l2','e_w2','e_h','e_r_y','e_fg','e_bg','a_x_c','a_y_c','a_z_c','a_l2','a_w2','a_h','a_r_y','a_fg','a_bg']
    #vals = ['e_fg','e_bg','a_fg','a_bg']
    #vals = ['e_x_c','e_y_c','e_z_c','e_l2','e_w2','e_h','e_r_y','a_x_c','a_y_c','a_z_c','a_l2','a_w2','a_h','a_r_y']
    #vals = ['e_x_c','e_y_c','e_l','e_w','e_fg','e_bg','a_x_c','a_y_c','a_l','a_w','a_fg','a_bg']
    #kstest   = modelling_utils.run_kstest(df_tp,df_fp,param,vals,sum_vals=True)
    is_summed = False
    kldiv = modelling_utils.run_kldiv(df_tp,df_fp,param,vals,sum_vals=is_summed)
    print('kldiv: {} summed: {}'.format(kldiv,is_summed))
    #jsdiv = modelling_utils.run_jsdiv(df_tp,df_fp,param,vals,sum_vals=False)
    #print('kstest: {} kldiv: {} jsdiv {}'.format(kstest,kldiv,np.sum(jsdiv)))
    #box_stats = modelling_utils.plot_box_plot(df,param,vals,plot=False)
    #print('box_plot: mean: {:.3f} median: {:.3f} [Q1,Q3]: [{:.3f},{:.3f}] [min,max]: [{:.3f},{:.3f}]'.format(box_stats[0],box_stats[1],box_stats[2],box_stats[3],box_stats[4],box_stats[5]))
    #plt.rcParams.update({'font.size': 16})
    #m_kde_fp = modelling_utils.plot_histo_multivariate_KDE(df_fp,'FP',param,vals,min_val=0, plot=True, bins=200)
    #m_kde_tp = modelling_utils.plot_histo_multivariate_KDE(df_tp,'TP',param,vals,min_val=0, plot=True, bins=200)

    #m_kde = modelling_utils.plot_histo_multivariate_KDE(df,'All',param,vals,min_val=0, plot=False)
    #modelling_utils.plot_roc_curves(df,param,vals,m_kde_tp,m_kde_fp,limiter=0,num_pts=300)
    #plt.legend()
    #plt.show()
    '''

    #-------------------------
    # Mat's custom KDE script
    #-------------------------
    '''
    param_list = []
    for i in range(0,8):
        param_list.append('e_bbox_var')
    #param_list.append('e_bbox_var')
    #param_list.append('a_cls_var')
    #param_list.append('e_cls_var')
    #param_list.append('e_bbox_var,a_bbox_var')
    #param_list.append('e_cls_var,a_cls_var')
    #param_list.append('all_var')
    vals_list = []

    #vals_list.append(['x_c','y_c','l1','w1'])
    #vals_list.append(['x_c','y_c','l1','w1'])
    #LiDAR contour plots
    #vals_list.append(['x_c','y_c'])
    #vals_list.append(['l2','w2'])
    #vals_list.append(['z_c','h'])
    #vals_list.append(['x_c','y_c','z_c'])
    #vals_list.append(['l2','w2','h'])
    #LiDAR Individual vars
    vals_list.append(['x_c'])
    vals_list.append(['y_c'])
    vals_list.append(['z_c'])
    vals_list.append(['l2'])
    vals_list.append(['w2'])
    vals_list.append(['h'])
    vals_list.append(['r_y'])
    vals_list.append(['x_c','y_c','z_c','l2','w2','h','r_y'])
    #Classification Vars
    #vals_list.append(['fg'])
    #vals_list.append(['bg'])
    #vals_list.append(['fg','bg'])
    #LiDAR total vars
    #vals_list.append(['x_c','y_c','z_c','l2','w2','h','r_y'])
    x = np.linspace(0,1,10)
    plt.rcParams.update({'font.size': 14})
    for param, vals in zip(param_list,vals_list):
        #print('param: {} vals: {}'.format(param,vals))
        is_summed = True
        kldiv = modelling_utils.run_kldiv(df_tp,df_fp,param,vals,sum_vals=is_summed)
        print('kldiv: {} summed: {}'.format(kldiv,is_summed))
        #m_kde_fp = modelling_utils.plot_histo_multivariate_KDE(df_fp,'FP',param,vals,min_val=0, plot=True, bins=200)
        #m_kde_tp = modelling_utils.plot_histo_multivariate_KDE(df_tp,'TP',param,vals,min_val=0, plot=True, bins=200)
        box_stats = modelling_utils.plot_box_plot(df,param,vals,plot=False)
        print('box_plot: mean: {:.3f} median: {:.3f} [Q1,Q3]: [{:.3f},{:.3f}] [min,max]: [{:.3f},{:.3f}]'.format(box_stats[0],box_stats[1],box_stats[2],box_stats[3],box_stats[4],box_stats[5]))

        #m_kde = modelling_utils.plot_histo_multivariate_KDE(df,'All',param,vals,min_val=0, plot=False)
        #modelling_utils.plot_roc_curves(df,param,vals,m_kde_tp,m_kde_fp,limiter=10000,num_pts=2000)
        #plt.legend()
        #plt.show()
    '''
    #-------------------------
    # Mat's custom KDE/histo plotter
    #-------------------------
    custom_plotter = False
    if(custom_plotter):
        opt_point_list = []
        param_list = []
        param_list.append('a_bbox_var')
        #param_list.append('e_bbox_var')
        #param_list.append('a_cls_var')
        #param_list.append('e_cls_var')
        #param_list.append('e_bbox_var,a_bbox_var')
        #param_list.append('e_cls_var,a_cls_var')
        param_list.append('all_var')
        vals_list = []
        vals_list.append(['z_c'])
        #vals_list.append(['x_c','y_c','l1','w1'])
        #vals_list.append(['x_c','y_c','z_c','l2','w2','h','r_y'])
        #vals_list.append(['x_c','y_c','z_c','l2','w2','h','r_y'])
        #vals_list.append(['fg','bg'])
        #vals_list.append(['fg','bg'])
        #vals_list.append(['e_x_c','e_y_c','e_l1','e_w1','a_x_c','a_y_c','a_l1','a_w1'])
        #vals_list.append(['e_x_c','e_y_c','e_z_c','e_l2','e_w2','e_h','e_r_y','a_x_c','a_y_c','a_z_c','a_l2','a_w2','a_h','a_r_y'])
        #vals_list.append(['e_fg','e_bg','a_fg','a_bg'])
        #vals_list.append(['e_x_c','e_y_c','e_l1','e_w1','e_fg','e_bg','a_x_c','a_y_c','a_l1','a_w1','a_fg','a_bg'])
        #vals_list.append(['e_x_c','e_y_c','e_z_c','e_l2','e_w2','e_h','e_r_y','e_fg','e_bg','a_x_c','a_y_c','a_z_c','a_l2','a_w2','a_h','a_r_y','a_fg','a_bg'])
        plt.rcParams.update({'font.size': 16})
        #plt.ylabel('Percentage Object Detected')
        #plt.xlabel('Percentage False Alarm')
        #plt.title('Uncertainty Receiver Operating Characteristics')
        kde_list = []
        for param, vals in zip(param_list,vals_list):
            print('param: {} vals: {}'.format(param,vals))
            m_kde_tp = modelling_utils.plot_histo_multivariate_KDE(df_tp,'TP',param,vals,min_val=0, plot=True, bins=200)
        plt.legend()
        plt.show()
    
    #-------------------------
    # Mat's custom ROC script
    #-------------------------
    custom_roc_en = True
    if(custom_roc_en):
        opt_point_list = []
        param_list = []
        param_list.append('a_bbox_var')
        param_list.append('e_bbox_var')
        param_list.append('a_cls_var')
        param_list.append('e_cls_var')
        #param_list.append('e_bbox_var,a_bbox_var')
        #param_list.append('e_cls_var,a_cls_var')
        param_list.append('all_var')
        vals_list = []
        #vals_list.append(['x_c','y_c','l1','w1'])
        #vals_list.append(['x_c','y_c','l1','w1'])
        vals_list.append(['x_c','y_c','z_c','l2','w2','h','r_y'])
        vals_list.append(['x_c','y_c','z_c','l2','w2','h','r_y'])
        vals_list.append(['fg','bg'])
        vals_list.append(['fg','bg'])
        #vals_list.append(['e_x_c','e_y_c','e_l1','e_w1','a_x_c','a_y_c','a_l1','a_w1'])
        #vals_list.append(['e_x_c','e_y_c','e_z_c','e_l2','e_w2','e_h','e_r_y','a_x_c','a_y_c','a_z_c','a_l2','a_w2','a_h','a_r_y'])
        #vals_list.append(['e_fg','e_bg','a_fg','a_bg'])
        #vals_list.append(['e_x_c','e_y_c','e_l1','e_w1','e_fg','e_bg','a_x_c','a_y_c','a_l1','a_w1','a_fg','a_bg'])
        vals_list.append(['e_x_c','e_y_c','e_z_c','e_l2','e_w2','e_h','e_r_y','e_fg','e_bg','a_x_c','a_y_c','a_z_c','a_l2','a_w2','a_h','a_r_y','a_fg','a_bg'])
        x = np.linspace(0,1,10)
        plt.rcParams.update({'font.size': 16})
        plt.plot(x,x,'--',color='black')
        plt.ylabel('Percentage Object Detected')
        plt.xlabel('Percentage False Alarm')
        plt.title('Uncertainty Receiver Operating Characteristics')
        kde_list = []
        for param, vals in zip(param_list,vals_list):
            print('param: {} vals: {}'.format(param,vals))
            is_summed = False
            #kldiv = modelling_utils.run_kldiv(df_tp,df_fp,param,vals,sum_vals=is_summed)
            #print('kldiv: {} summed: {}'.format(kldiv,is_summed))
            m_kde_fp = modelling_utils.plot_histo_multivariate_KDE(df_fp,'FP',param,vals,min_val=0, plot=False, bins=200)
            m_kde_tp = modelling_utils.plot_histo_multivariate_KDE(df_tp,'TP',param,vals,min_val=0, plot=False, bins=200)
            m_kde = modelling_utils.plot_histo_multivariate_KDE(df,'All',param,vals,min_val=0, plot=False, bins=200)
            modelling_utils.plot_roc_curves(df,param,vals,m_kde_tp,m_kde_fp,limiter=15000,num_pts=1000)
        plt.legend()
        plt.show()
    
    #-------------------------
    # Mat's custom ROC TEST script
    #-------------------------
    custom_ROC_TEST_EN = False
    if(custom_ROC_TEST_EN):
        opt_point_list = []
        param_list = []
        param_list.append('a_bbox_var')
        param_list.append('e_bbox_var')
        param_list.append('a_cls_var')
        param_list.append('e_cls_var')
        param_list.append('e_bbox_var,a_bbox_var')
        param_list.append('e_cls_var,a_cls_var')
        param_list.append('all_var')
        vals_list = []
        vals_list.append(['x_c','y_c','l1','w1'])
        vals_list.append(['x_c','y_c','l1','w1'])
        #vals_list.append(['x_c','y_c','z_c','l2','w2','h','r_y'])
        #vals_list.append(['x_c','y_c','z_c','l2','w2','h','r_y'])
        vals_list.append(['fg','bg'])
        vals_list.append(['fg','bg'])
        vals_list.append(['e_x_c','e_y_c','e_l1','e_w1','a_x_c','a_y_c','a_l1','a_w1'])
        #vals_list.append(['e_x_c','e_y_c','e_z_c','e_l2','e_w2','e_h','e_r_y','a_x_c','a_y_c','a_z_c','a_l2','a_w2','a_h','a_r_y'])
        vals_list.append(['e_fg','e_bg','a_fg','a_bg'])
        vals_list.append(['e_x_c','e_y_c','e_l1','e_w1','e_fg','e_bg','a_x_c','a_y_c','a_l1','a_w1','a_fg','a_bg'])
        #vals_list.append(['e_x_c','e_y_c','e_z_c','e_l2','e_w2','e_h','e_r_y','e_fg','e_bg','a_x_c','a_y_c','a_z_c','a_l2','a_w2','a_h','a_r_y','a_fg','a_bg'])
        x = np.linspace(0,1,10)
        plt.rcParams.update({'font.size': 14})
        plt.plot(x,x,'--',color='black')
        plt.ylabel('Percentage Object Detected')
        plt.xlabel('Percentage False Alarm')
        plt.title('Uncertainty Receiver Operating Characteristics')
        kde_list = []
        for param, vals in zip(param_list,vals_list):
            print('param: {} vals: {}'.format(param,vals))
            is_summed = False
            #kldiv = modelling_utils.run_kldiv(df_tp,df_fp,param,vals,sum_vals=is_summed)
            #print('kldiv: {} summed: {}'.format(kldiv,is_summed))
            m_kde_fp = modelling_utils.plot_histo_multivariate_KDE(df_fp,'FP',param,vals,min_val=0, plot=False, bins=200)
            m_kde_tp = modelling_utils.plot_histo_multivariate_KDE(df_tp,'TP',param,vals,min_val=0, plot=False, bins=200)
            m_kde = modelling_utils.plot_histo_multivariate_KDE(df,'All',param,vals,min_val=0, plot=False, bins=200)
            kde_list.append([m_kde_tp,m_kde_fp,m_kde])
            max_pfa = 0.1
            opt_point, opt_ratio = modelling_utils.get_roc_opt_point(df,param,vals,m_kde_tp,m_kde_fp,limiter=15000,num_pts=1000, max_pfa=max_pfa)
            opt_point_list.append(opt_point)
            print('optimal ROC point for max pfa {}: {}, Ratio: {}'.format(max_pfa,opt_point,opt_ratio))
        
        #Optimal point operation of nominal KDE distributions
        #Testing on df_test (can be any subset of data, or even from a different detector!)
        df_test_list = []
        df_test_list.append(df_orig)
        #df_test_list.append(no_cover_df)
        #df_test_list.append(snow_cover_df)
        df_test_list.append(none_light_snow)
        df_test_list.append(heavy_extreme_snow)
        #df_test_list.append(sun_dets)
        #df_test_list.append(rain_dets)
        #df_test_list.append(day_dets)
        #df_test_list.append(night_dets)
        #df_test_list.append(medium_snow)
        #for test_file in test_file_list:
        #    df_t = get_df(args.dataset,args.cache_dir,test_file,data_dir,args.sensor_type,limiter=50000)
        #    df_t = df_t.loc[df_t['confidence'] > conf_thresh]
        #    df_test_list.append(df_t)
        #df_test = night_dets
        min_anom_thresh = 0.01
        test_limiter = 10000
        for df_test in df_test_list:
            df_test = modelling_utils._apply_limiter(df_test,test_limiter)
            print('{}'.format('New test stimulus'))
            for i, param in enumerate(param_list):
                vals = vals_list[i]
                opt_point = opt_point_list[i]
                m_kde_tp = kde_list[i][0]
                m_kde_fp = kde_list[i][1]
                m_kde    = kde_list[i][2]
                print('param: {}'.format(param))
                #print('param: {} vals: {} thresh: {}'.format(param,vals,opt_point))
                #m_kde_fp = modelling_utils.plot_histo_multivariate_KDE(df_fp,'FP',param,vals, plot=False, bins=200)
                #m_kde_tp = modelling_utils.plot_histo_multivariate_KDE(df_tp,'TP',param,vals, plot=False, bins=200)
                #m_kde    = modelling_utils.plot_histo_multivariate_KDE(df,'ALL',param,vals, plot=False, bins=200)
                signal_tp = np.where(np.asarray(df_test['difficulty'].to_list()) != -1, True, False)
                test_var = modelling_utils._col_data_extract(df_test,param,vals)
                tp_densities = m_kde_tp.pdf(test_var)
                fp_densities = m_kde_fp.pdf(test_var)
                all_densities = m_kde.pdf(test_var)
                response_tp = np.where(tp_densities > fp_densities + opt_point,True,False)
                response_anomaly = np.where(all_densities < min_anom_thresh*np.max(all_densities),True,False)
                ratios = modelling_utils.find_ratios(df_test,param,vals,signal_tp,tp_densities,fp_densities,min_thresh=opt_point)
                #print('ratios: {}'.format(ratios))
                total_resp = np.bitwise_and(np.bitwise_not(response_tp),response_anomaly)
                print('TP,FP:\n{}\n{}'.format(np.sum(response_tp),np.sum(np.bitwise_not(response_tp))))
                #print('Anom:\n{}\n{}'.format(np.sum(response_anomaly),np.sum(np.bitwise_not(response_anomaly))))
                #print('FP&Anom:\n{}\n{}'.format(np.sum(total_resp),np.sum(np.bitwise_not(total_resp))))
                #print('ratio: {:.3f} det-TP: {} det-FP: {}'.format(np.sum(np.bitwise_not(response_tp))/np.sum(response_tp),np.sum(response_tp),np.sum(np.bitwise_not(response_tp))))
                #print('ratio: {:.3f} anomaly: {} norm: {}'.format(np.sum(response_anomaly)/np.sum(np.bitwise_not(response_anomaly)),np.sum(response_anomaly),np.sum(np.bitwise_not(response_anomaly))))
            
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
