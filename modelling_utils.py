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

def plot_histo_inverse_gamma(dets,scene,col,val,min_val=None,max_val=None):
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

        hist_range = (min_val,max_val)
        
        #myPDF,axes = fastKDE.pdf(data_arr[0,:],data_arr[1,:])
        #Extract the axes from the axis list
        #v1,v2 = axes

        #Plot contours of the PDF should be a set of concentric ellipsoids centered on
        #(0.1, -300) Comparitively, the y axis range should be tiny and the x axis range
        #should be large
        #PP.contour(v1,v2,myPDF)
        #PP.show()

        df = pd.DataFrame({'x_c': data_arr[0, :], 'y_c': data_arr[1, :]})
        sns.jointplot(x='x_c',y='y_c', data=df, kind='kde')
        scipy_kernel = gaussian_kde(data_arr,bw_method=.035)  # kernel function centered on each datapoint
        pdf = scipy_kernel.evaluate(x_list)  # sum all functions together and normalize to obtain pdf
        plt.plot(x_list[0,:],pdf)
        #plt.show()
        

        x_mesh = np.meshgrid(x_list[:,0],x_list[:,1])
        x_mesh = x_mesh.swapaxes(0,2)
        multivariate_kernel = sm.nonparametric.KDEMultivariate(data_arr, var_type='cc', bw='normal_reference')
        pdf = multivariate_kernel.pdf(x_mesh)
        h = multivariate_kernel.bw
        plt.contour(x_list[:,0],x_list[:,1],pdf)
        plt.show()
        #x = np.linspace(np.min(data_arr[0,:]),np.max(data_arr[0,:]),len(pdf))
        plt.plot(x,pdf,'k')
        for count,row in enumerate(data_arr):
            plt.hist(row,bins=200,range=hist_range,alpha=0.5,label=col + ': ' + str(vals[count]),density=True,stacked=True)
        print("h/bandwidth value = ", h)
    return

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
        return None

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


if __name__ == '__main__':
    print('cannot run file stand-alone')