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

def parse_dets(det_file):
    if ("3d" in detection_file):  # is the file from lidar or img domain
        lidar_flag = True
    else:
        lidar_flag = False
    data_rows     = []
    column_names  = ['ID']
    track_en      = False
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


def plot_histo_cls_uc(dets,scene,min_val,max_val):
    #ax = dets.plot.hist(column='a_bbox_var',bins=12,alpha=0.5)
    bboxes = dets.columns
    for column in bboxes:
        if('a_mutual_info' in column):
            labelname = scene + '_' + column
            data = dets[column].to_list()
            conf_list = dets['confidence'].to_list()
            hist_data = []
            for i,var_line in enumerate(data):
                variance = var_line
                #conf = dets['confidence'][i]
                conf = conf_list[i]
                #variance = variance/con'
                hist_data.append(variance)
            #max_val = max(hist_data)
            #min_val = min(hist_data)
            #mean    = np.mean(hist_data)
            #hist_data = (hist_data-min_val)/(max_val-min_val)
            plt.hist(hist_data,bins=50,range=[min_val,max_val],alpha=0.3,label=labelname,density=True,stacked=True)
    #bboxes = bboxes.to_dict(orient="list")
    return None

def plot_histo_gaussian(df,dets,scene,min_val,max_val):
    bboxes = dets.filter(like='bb').columns
    a_bbox_var = np.asarray(df['a_bbox_var'].to_list())
    a_bbox_var = np.sort(np.sum(a_bbox_var,axis=1))  # sum and sort
    a_bbox_var_std_dev = np.std(a_bbox_var)

    e_bbox_var = np.asarray(df['e_bbox_var'].to_list())
    e_bbox_var = np.sort(np.sum(e_bbox_var,axis=1))  # sum and sort
    e_bbox_var_std_dev = np.std(e_bbox_var)

    for column in bboxes:
        if('a_bbox_var' in column):
            labelname = scene + '_' + column
            data = dets[column].to_list()  # uncertainty
            #bbgt = dets['bbgt'].to_list()
            #bbdet = dets['bbdet'].to_list()
            #hist_data = []
            data = np.asarray(data)
            data = np.sum(data,axis=1)  # sum and sort
            range = np.max(data) - np.min(data)
            data_truncated = data[data < range*0.20]
            data_truncated = (data_truncated)/a_bbox_var_std_dev  # convert to z scores            
            data_truncated = np.log(data_truncated)  # to fit normal distribution, log data 
            (mu, sigma) = norm.fit(data_truncated) # fit data to normal distribution
            #l = plt.plot(200, y, 'r--', linewidth=2)

            x_range = np.arange(np.min(data_truncated),np.max(data_truncated),0.001)
            pdf = norm.pdf(x_range, mu, sigma)
            estimated_mu = float("{:.3f}".format(mu))
            estimated_sigma = float("{:.3f}".format(sigma))

            plt.plot(x_range,pdf,label='fitted_norm_pdf')
            plt.hist(data_truncated,bins=200,range=[np.min(data_truncated),np.max(data_truncated)+1],alpha=0.5,label=labelname,density=True,stacked=True)
            plt.text(1.5,1.2,r'$\mu=$''%s\n'r'$\sigma=$''%s'%(np.mean(data_truncated),estimated_sigma)) 
    return data, estimated_mu, estimated_sigma


def plot_histo_chi_squared(dets,scene,min_val,max_val):
    bboxes = dets.filter(like='bb').columns
    for column in bboxes:
        if('a_bbox_var' in column):
            labelname = scene + '_' + column
            data = dets[column].to_list()  # uncertainty
            bbgt = dets['bbgt'].to_list()
            bbdet = dets['bbdet'].to_list()
            hist_data = []
            data = np.asarray(data)
            data = np.sort(np.sum(data,axis=1))  # sum and sort
            range = np.max(data) - np.min(data)
            data_truncated = data[data < range*0.20]
            data_std = np.std(data_truncated)
            data_truncated = (data_truncated)/data_std  # convert to z scores
            df = 4  # degrees of freedom
            x_range = np.arange(np.min(data_truncated),np.max(data_truncated),0.001)

            s = .2

            # chi_square = np.random.chisquare(1000,len(data)*10000)
            # chi_norm = np.linalg.norm(chi_square)
            # chi_square = chi_square/chi_norm
            # bbdet = np.asarray(bbdet)
            # bbox_area = (bbdet[:,2]-bbdet[:,0])*(bbdet[:,3]-bbdet[:,1]) + 1
            #print(len(bbox_area))
            #data = data/bbox_area
            #data = data/bbox_area
            #for i, bbox_var in enumerate(data):
            #
            #    bbox_area = (bbdet[i][2]-bbdet[i][0])*(bbdet[i][3]-bbdet[i][1]) + 1
            #    variance = sum(bbox_var)
            #    #variance = sum(bbox_var)/4
            #    hist_data.append(variance)
            #max_val = max(hist_data)
            #min_vdraw   = np.mean(hist_data)
            #hist_data = (hist_data-min_val)/(max_val-min_val)
    
            plt.hist(data_truncated,bins=200,range=[np.min(data_truncated),np.max(data_truncated)],alpha=0.5,label=labelname,density=True,stacked=True)
            plt.plot(x_range, lognorm.pdf(x_range, s,-0.8,1), alpha=0.6, label='lognorm pdf')
            #plt.plot(x_range, chi2.pdf(x_range, df), alpha=0.6, label='chi2 pdf')
            #plt.text(1.50,1.0,'mean= %s \n std_dev=%s '%(data_mean,data_std))
            #sns.distplot(data,label='fitting curve')
            #plt.hist(r,bins=200,range=[min_val,max_val],alpha=0.5,label='r',density=True,stacked=True)
    #bboxes = bboxes.to_dict(orient="list")
    return data

def plot_histo_lognorm(df,dets,scene,min_val,max_val):
    bboxes = dets.filter(like='bb').columns
    a_bbox_var = np.asarray(df['a_bbox_var'].to_list())
    a_bbox_var = np.sort(np.sum(a_bbox_var,axis=1))  # sum and sort
    a_bbox_var_std_dev = np.std(a_bbox_var)

    e_bbox_var = np.asarray(df['e_bbox_var'].to_list())
    e_bbox_var = np.sort(np.sum(e_bbox_var,axis=1))  # sum and sort
    e_bbox_var_std_dev = np.std(e_bbox_var)

    for column in bboxes:
        if('a_bbox_var' in column):
            labelname = scene + '_' + column
            data = dets[column].to_list()  # uncertainty
            bbgt = dets['bbgt'].to_list()
            bbdet = dets['bbdet'].to_list()
            hist_data = []
            data = np.asarray(data)
            data = np.sort(np.sum(data,axis=1))  # sum and sort
            range = np.max(data) - np.min(data)
            data_truncated = data[data < range*0.20]
            data_truncated = (data_truncated)/e_bbox_var_std_dev  # convert to z scores            
            df = 4  # degrees of freedom
            x_range = np.arange(np.min(data_truncated),np.max(data_truncated),0.001)
            data_mean = np.mean(data_truncated)
            s,loc,scale = lognorm.fit(data_truncated,floc=0)
            estimated_mu = np.log(scale)
            estimated_sigma = s  # shape is std dev
            estimated_mu = float("{:.3f}".format(estimated_mu))
            estimated_sigma = float("{:.3f}".format(estimated_sigma))
            pdf = lognorm.pdf(x_range,s,scale=scale)
            plt.plot(x_range,pdf,label='fitted_lognorm_pdf')
            plt.hist(data_truncated,bins=200,range=[np.min(data_truncated),np.max(data_truncated)+1],alpha=0.5,label=labelname,density=True,stacked=True)
            plt.text(1.5,1.2,r'$\mu=$''%s\n'r'$\sigma=$''%s'%(np.mean(data_truncated),estimated_sigma)) 
    return data, estimated_mu, estimated_sigma


def plot_histo_half_gaussian(df,dets,scene,min_val,max_val):
    bboxes = dets.filter(like='bb').columns
    a_bbox_var = np.asarray(df['a_bbox_var'].to_list())
    a_bbox_var = np.sort(np.sum(a_bbox_var,axis=1))  # sum and sort
    a_bbox_var_std_dev = np.std(a_bbox_var)
    for column in bboxes:
        if('a_bbox_var' in column):
            labelname = scene + '_' + column
            data = dets[column].to_list()  # uncertainty
            #bbgt = dets['bbgt'].to_list()
            #bbdet = dets['bbdet'].to_list()
            #hist_data = []
            data = np.asarray(data)
            data = np.sort(np.sum(data,axis=1))  # sum and sort
            range = np.max(data) - np.min(data)
            data_truncated = data[data < range*0.20]
            data_truncated = (data_truncated)/a_bbox_var_std_dev  # convert to z scores            
            x_range = np.arange(np.min(data_truncated),np.max(data_truncated),0.001)
            loc,scale = halfnorm.fit(data_truncated,floc=0)
            estimated_mu = float("{:.3f}".format(np.mean(data_truncated)))
            estimated_sigma = float("{:.3f}".format(np.std(data_truncated)))
            pdf = halfnorm.pdf(x_range,scale=scale)

            plt.plot(x_range,pdf,label='fitted_halfnorm_pdf')
            #plt.hist(data_truncated,bins=200,range=[np.min(data_truncated),np.max(data_truncated)+1],alpha=0.5,label=labelname,density=True,stacked=True)
            #plt.text(1.5,1.2,r'$\mu=$''%s\n'r'$\sigma=$''%s'%(estimated_mu,estimated_sigma)) 
    return data


def plot_histo_bbox_uc(dets,scene,min_val,max_val):
    #ax = dets.plot.hist(column='a_bbox_var',bins=12,alpha=0.5)
    bboxes = dets.filter(like='bb').columns
    for column in bboxes:
        if('a_bbox_var' in column):
            labelname = scene + '_' + column
            data = dets[column].to_list()  # uncertainty
            bbgt = dets['bbgt'].to_list()
            bbdet = dets['bbdet'].to_list()
            hist_data = []
            data = np.asarray(data)
            data = np.sum(data,axis=1)
            bbdet = np.asarray(bbdet)
            bbox_area = (bbdet[:,2]-bbdet[:,0])*(bbdet[:,3]-bbdet[:,1]) + 1
            #print(len(bbox_area))
            #data = data/bbox_area
            #data = data/bbox_area
            #for i, bbox_var in enumerate(data):
            #
            #    bbox_area = (bbdet[i][2]-bbdet[i][0])*(bbdet[i][3]-bbdet[i][1]) + 1
            #    variance = sum(bbox_var)
            #    #variance = sum(bbox_var)/4
            #    hist_data.append(variance)
            #max_val = max(hist_data)
            #min_vdraw   = np.mean(hist_data)
            #hist_data = (hist_data-min_val)/(max_val-min_val)
    return data

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

def extract_twosigma(df,mode=None):
    """
    Filter low confidence predictions beyond 2sigma log(var) in positive and negative direction
    return the filtered df, to be used with drawing and scatter plots.
    args: df, mode(0:high var, 1:low var)
    return: filtered twosigma df
    """
    a_bbox_var = np.asarray(df['a_bbox_var'].to_list())
    a_bbox_var = np.sum(a_bbox_var,axis=1)  # sum 
    a_bbox_var_log = np.log(a_bbox_var)
    df.insert(len(df.columns),'a_bbox_var_sum',a_bbox_var)
    df.insert(len(df.columns),'a_bbox_var_log',a_bbox_var_log)
    std_dev = np.std(a_bbox_var_log)
    mean = np.mean(a_bbox_var_log)
    twosigma = std_dev*2
    threshold = mean + twosigma
    if (not mode):
        filtered_df = df.loc[df['a_bbox_var_log'] >= threshold]  # high variance
    else:
        filtered_df = df.loc[df['a_bbox_var_log'] <= threshold]  # low variance
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

def plot_scatter_var(df,x,y):
    '''
    Scatter plot function x against y, y is usually variance (uncertainty)
    args: df, x, y (x and y must be same length)
    returns: None
    '''
    data_x = df[x].to_list()
    data_x = np.asarray(data_x)
    data_y = df[y].to_list()
    data_y = np.asarray(data_y)
    if  x == 'bbdet':  # x0,y0,x1,y1
        width = data_x[:,2]-data_x[:,0]
        length = data_x[:,3]-data_x[:,1]
        data_x = width * length 
        x = 'bbox_area'
    elif data_x[1].shape:  # for both data, check if they must be summed
        data_x = np.sum(data_x,axis=1) 
    if y == 'bbdet':  # x0,y0,x1,y1
        width = data_y[:,2]-data_y[:,0]
        length = data_y[:,3]-data_y[:,1]
        data_y = width * length 
        y = 'bbox_area'
    elif data_y[1].shape:
        data_y = np.sum(data_y,axis=1) 
        
    label = x + ' vs ' + y
    covariance = np.cov(data_x,data_y)
    plt.scatter(data_x,data_y,label=label,color='r',marker='*',s=1)
    #plt.text(.7,1,'cov = %s %s  %s %s' %(covariance[0,0],covariance[0,1],covariance[1,0],covariance[1,1])) 
    #plt.text(1.2,0,'cov = %s %s  %s %s' %(covariance[0,0],covariance[0,1],covariance[1,0],covariance[1,1]))
    print('covariance=\n', covariance[0,0],covariance[0,1],'\n',covariance[1,0],covariance[1,1]) 
    plt.xlabel(x)
    plt.ylabel(y)

def label_3D_to_image(json_calib, metadata, bbox):
    bbox_transform_matrix = get_box_transformation_matrix(bbox)  
    instrinsic = json_calib[0]['cam_intrinsic']
    extrinsic = np.array(json_calib[0]['cam_extrinsic_transform']).reshape(4,4)
    vehicle_to_image = get_image_transform(instrinsic, extrinsic)  # magic array 4,4 to multiply and get image domain
    box_to_image = np.matmul(vehicle_to_image, bbox_transform_matrix)


    # Loop through the 8 corners constituting the 3D box
    # and project them onto the image
    vertices = np.empty([2,2,2,2])
    # 1: 000, 2: 001, 3: 010:, 4: 100
    for k in [0, 1]:
        for l in [0, 1]:
            for m in [0, 1]:
                # 3D point in the box space
                v = np.array([(k-0.5), (l-0.5), (m-0.5), 1.])

                # Project the point onto the image
                v = np.matmul(box_to_image, v)

                # If any of the corner is behind the camera, ignore this object.
                if v[2] < 0:
                    return None

                vertices[k,l,m,:] = [v[0]/v[2], v[1]/v[2]]

    vertices = vertices.astype(np.int32)

    return vertices

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

def get_box_transformation_matrix(box):
    """Create a transformation matrix for a given label box pose."""

    #tx,ty,tz = box.center_x,box.center_y,box.center_z
    tx = box[0]
    ty = box[1]
    tz = box[2]
    c = math.cos(box[3])
    s = math.sin(box[3])

    #sl, sh, sw = box.length, box.height, box.width
    sl = box[4]
    sh = box[5]
    sw = box[6]

    return np.array([
        [ sl*c,-sw*s,  0,tx],
        [ sl*s, sw*c,  0,ty],
        [    0,    0, sh,tz],
        [    0,    0,  0, 1]])

def get_image_transform(intrinsic, extrinsic):
    """ For a given camera calibration, compute the transformation matrix
        from the vehicle reference frame to the image space.
    """
    # Camera model:
    # | fx  0 cx 0 |
    # |  0 fy cy 0 |
    # |  0 0  1 0 |
    camera_model = np.array([
        [intrinsic[0], 0, intrinsic[2], 0],
        [0, intrinsic[1], intrinsic[3], 0],
        [0, 0,                       1, 0]])

    # Swap the axes around
    axes_transformation = np.array([
        [0,-1,0,0],
        [0,0,-1,0],
        [1,0,0,0],
        [0,0,0,1]])

    # Compute the projection matrix from the vehicle space to image space.
    vehicle_to_image = np.matmul(camera_model, np.matmul(axes_transformation, np.linalg.inv(extrinsic)))
    return vehicle_to_image

def compute_2d_bounding_box(points):
    """Compute the 2D bounding box for a set of 2D points.
    
    img_or_shape: Either an image or the shape of an image.
                  img_or_shape is used to clamp the bounding box coordinates.
    
    points: The set of 2D points to use
    """

    # Compute the 2D bounding box and draw a rectangle
    x1 = np.amin(points[...,0])
    x2 = np.amax(points[...,0])
    y1 = np.amin(points[...,1])
    y2 = np.amax(points[...,1])

    return (x1,y1,x2,y2)
    
def clip_2d_bounding_box(shape, points):
    x1 = min(max(0,points[0]),shape[1])
    y1 = min(max(0,points[1]),shape[0])
    x2 = min(max(0,points[2]),shape[1])
    y2 = min(max(0,points[3]),shape[0])
    return (x1,y1,x2,y2)

def compute_truncation(points, clipped_points):
    clipped_area = (clipped_points[2] - clipped_points[0])*(clipped_points[3] - clipped_points[1])
    orig_area    = (points[2] - points[0])*(points[3] - points[1])
    if(clipped_area <= 0):
        return 1.0  # nothing has been clipped
    else:
        return 1-(clipped_area/orig_area)

def bbdet3d_to_bbdet2d(df):
    """
    Function to convert 3d bounding boxes to 2d. These transformed coordinates
    are appended to the df to be drawn. 
    args: df 
    returns: modified df
    """
    bbox2D_col = []
    im_shape = (950,1920,3)
    # for each bbdet entry, transform it to image domain and append
    for index, row in df.iterrows():
        bbox2D = label_3D_to_image(row['calibration'],0,row['bbdet3d'])  # investigate meta 
        if (bbox2D is None):  # check for a return
            bbox2D = [-1,-1,-1,-1]  # no return
            bbox2D_col.append(bbox2D)
            continue
        bbox2D = compute_2d_bounding_box(bbox2D)
        if min(bbox2D)<0:  # check if any inside image frame
            bbox2D = [-1,-1,-1,-1] # not inside image frame
            bbox2D_col.append(bbox2D)
            continue
        bbox2D_clipped = clip_2d_bounding_box(im_shape, bbox2D)
        truncation     = compute_truncation(bbox2D,bbox2D_clipped)
        if truncation > .90:  # last check if box is too far clipped
            bbox2D = [-1,-1,-1,-1] # not inside image frame
            bbox2D_col.append(bbox2D)
            continue
        bbox2D = [bbox2D[0], bbox2D[1]-top_crop, bbox2D[2], bbox2D[3]-top_crop]
        bbox2D_col.append(bbox2D)
    df['bbdet3d_2d'] = bbox2D_col
    return df

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

def count_kitti_npos(val_split_file,gt_path,df):
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
                elif(occ <= 1 and trunc <= 0.3 and (BBGT_HEIGHT) >= 25): # difficulty 1
                    count[1:3] += 1
                elif(occ <= 2 and trunc <= 0.5 and (BBGT_HEIGHT) >= 25): # difficulty 2
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

def calculate_ap(df,d_levels):
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

    if 'kitti' in detection_file:  # detection_file still in scope
        npos = count_kitti_npos(val_split_file,gt_path,df)
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

def column_value_to_index(val):
    """
    Function to take string from either 3d or 2d column value and convert to 
    the proper index. Used with histrogram plotting
    args: val (column value string: 
                2d - x1, y1, x2, y2
                3d - x_c, y_c, z_c, l, w, h)
    returns: index value
    """
    if (val == 'x_c'):
        return 0
    elif (val == 'y_c'):
        return 1
    elif (val == 'l1' or val == 'z_c'):
        return 2
    elif (val == 'w1' or val == 'l2'):
        return 3
    elif (val == 'w2'):
        return 4
    elif (val == 'h'):
        return 5
    elif (val == 'r_y'):
        return 6
    else:
        print("unable to index value - string not recognized")
        return

def plot_histo_inverse_gaussian(dets,scene,col,val,min_val=None,max_val=None):
    """
    Function to plot inverse gamma using uncertainty parameters
    args: dataframe, dets, scenetype, minval, maxval, column(to be plotted), value(1 or all)

    column can be a_bbox_var, e_bbox_var, a_cls_var, e_cls_var (4 or 7 values - 2d or 3d)

    2d - x_c, y_c, l, w (l1, w1)
    3d - x_c, y_c, z_c, l, w, h, r_y (l2, w2)
    """
    fit = 1 
    bboxes = dets.columns

    for column in bboxes:
        if (col in column):
            data = dets[column].to_list()
            data = np.asarray(data)
            if (val == 'all'):
                data_arr = np.sum(data,axis=1)  
            else:
                idx = column_value_to_index(val)
                data_arr = data[:,idx]

    if(fit):
        if (min_val == None and max_val == None):
            hist_range = (np.min(data_arr),np.max(data_arr))
            x = np.arange(np.min(data_arr),np.max(data_arr),.0001)
        else:
            hist_range = (min_val,max_val)
            x = np.arange(min_val,max_val,.0001)
        shape,loc,scale = scipy_stats.invgamma.fit(data_arr)
        g1 = scipy_stats.invgamma.pdf(x=x, a=shape, loc=loc, scale=scale)
        plt.plot(x,g1,label='{} fitted_gamma: {:.3f} {:.3f} {:.3f}'.format(scene,shape,loc,scale))
        plt.hist(data_arr,bins=200,range=hist_range,alpha=0.5,label=col + ': ' + val,density=True,stacked=True)
        #plt.hist(data_arr,bins=200,range=[min_val,max_val],alpha=0.5,label=cil+': '+val,density=True,stacked=True)
    return 

def plot_histo_KDE(dets,scene,col,val,min_val=None,max_val=None):
    """
    Function to plot KDE using uncertainty parameters. Can specify or not specify range (min/max)
    args: dataframe, dets, scenetype, column(to be plotted), value(1 or all), minval, maxval

    column can be a_bbox_var, e_bbox_var, a_cls_var, e_cls_var (4 or 7 values - 2d or 3d)

    2d - x_c, y_c, l, w
    3d - x_c, y_c, z_c, l, w, h, r_y
    """
    fit = 1 
    bboxes = dets.columns

    for column in bboxes:
        if (col in column):
            data = dets[column].to_list()
            data = np.asarray(data)
            data = np.sort(data,axis=1) # for filtering outliers
            if (val == 'all'):
                data_arr = np.sum(data,axis=1)  
            else:
                idx = column_value_to_index(val)
                data_arr = data[:,idx]
            range = np.max(data) - np.min(data)  
            data_arr = data_arr[data_arr < range*0.80]  # filter above 80% values
    if(fit):
        if (min_val == None and max_val == None):
            min_val = np.min(data_arr)
            max_val = np.max(data_arr)
        elif (min_val == None and max_val != None):
            min_val = 0
        elif (max_val == None and min_val != None):
            max_val = np.max(data_arr)
        hist_range = (min_val,max_val)
        x = np.arange(min_val,max_val,.0001)

        scipy_kernel = gaussian_kde(data_arr,bw_method=.035)  # kernel function centered on each datapoint
        pdf = scipy_kernel.evaluate(x)  # sum all functions together and normalize to obtain pdf
        h = scipy_kernel.factor * np.std(data_arr)  # smoothing parameter h also known as bandwidth
        plt.plot(x,pdf,'k')
        plt.hist(data_arr,bins=200,range=hist_range,alpha=0.5,label=col + ': ' + val,density=True,stacked=True)
        print("h/bandwidth value = ", h)
    return 

def plot_histo_multivariate_KDE(dets,scene,col,vals,min_val=None,max_val=None):
    """
    Function to multivariate KDE. vals is a list of strings to obtain multiple entries
    args: dataframe, dets, scenetype, column(to be plotted), value(1 or all), minval, maxval

    column can be a_bbox_var, e_bbox_var, a_cls_var, e_cls_var (4 or 7 values - 2d or 3d)

    """
    fit = 1 
    bboxes = dets.columns
    data_list = []
    x_list = []

    for column in bboxes:
        if (col in column):
            data = dets[column].to_list()
            data = np.asarray(data)
            data = np.sort(data,axis=1) # for filtering outliers
            for val in vals:
                idx = column_value_to_index(val)
                data_list.append(data[:,idx])
            data_arr = np.asarray(data_list)
            #range = np.max(data) - np.min(data)  
            #data_arr = data_arr[data_arr < range*0.80]  # filter above 80% values
    if(fit):
        for row in data_arr:
            min_val = np.min(row)
            max_val = np.max(row)
            x_list.append(np.linspace(min_val,max_val,1000))
        data_arr = np.swapaxes(data_arr,1,0)
        x_list = np.asarray(x_list)
        x_list = np.swapaxes(x_list,1,0)
        #hist_range = (min_val,max_val) 

        """
        myPDF,axes = fastKDE.pdf(data_arr[0,:],data_arr[1,:])
        #Extract the axes from the axis list
        v1,v2 = axes

        #Plot contours of the PDF should be a set of concentric ellipsoids centered on
        #(0.1, -300) Comparitively, the y axis range should be tiny and the x axis range
        #should be large
        PP.contour(v1,v2,myPDF)
        PP.show()

        df = pd.DataFrame({'x_c': data_arr[0, :], 'y_c': data_arr[1, :]})
        sns.jointplot(x='x_c',y='y_c', data=df, kind='kde')
        scipy_kernel = gaussian_kde(data_arr,bw_method=.035)  # kernel function centered on each datapoint
        pdf = scipy_kernel.evaluate(x_list)  # sum all functions together and normalize to obtain pdf
        plt.plot(x_list[0,:],pdf)
        plt.show()
        """

        multivariate_kernel = sm.nonparametric.KDEMultivariate(data_arr, var_type='cc', bw='normal_reference')
        pdf = multivariate_kernel.pdf(x_list)
        h = multivariate_kernel.bw
        plt.contour(x_list[:,0],x_list[:,1],pdf)
        plt.show()
        #x = np.linspace(np.min(data_arr[0,:]),np.max(data_arr[0,:]),len(pdf))
        plt.plot(x,pdf,'k')
        for count,row in enumerate(data_arr):
            plt.hist(row,bins=200,range=hist_range,alpha=0.5,label=col + ': ' + str(vals[count]),density=True,stacked=True)
        print("h/bandwidth value = ", h)
    return

if __name__ == '__main__': 
    df = get_df(cache_dir,detection_file)
    #(mrec,prec,map) = calculate_ap(df,3)  # 2nd value is # of difficulty types
    #df = bbdet3d_to_bbdet2d(df)
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