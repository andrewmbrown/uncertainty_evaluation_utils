import os
import sys
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
from sklearn.metrics import auc
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import halfnorm
from scipy.stats import gaussian_kde
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import ks_2samp
import scipy.optimize as opt
from scipy.stats import entropy
#from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import pylab as PP

def run_kstest(df1,df2,col,vals,sum_vals=False):
    data1 = extract_columns(df1,col,vals)
    data2 = extract_columns(df2,col,vals)
    if(data1.ndim > 1):
        if(sum_vals):
            data1      = np.sum(data1,axis=1)
            data2      = np.sum(data2,axis=1)
            stat, pval = ks_2samp(data1,data2)
        else:    
            stat = np.zeros((data1.shape[1]))
            pval = np.zeros((data1.shape[1]))
            for i in range(0,data1.shape[1]):
                stat[i], pval[i] = ks_2samp(data1[:,i],data2[:,i])
    else:
        stat, pval = ks_2samp(data1,data2)
    return [stat, pval]

def run_kldiv(df1,df2,col,vals,sum_vals=False,bins=100):
    data1 = extract_columns(df1,col,vals)
    data2 = extract_columns(df2,col,vals)
    if(data1.ndim > 1):
        if(sum_vals or len(vals) == 1):
            data1      = np.sum(data1,axis=1)
            data2      = np.sum(data2,axis=1)
            stat = compute_kl_divergence(data1,data2,bins)
        else:
            stat = m_KLdivergence(data1,data2)
            #stat = np.zeros((data1.shape[1]))
            #for i in range(0,data1.shape[1]):
            #    stat[i] = compute_kl_divergence(data1[:,i],data2[:,i],bins)
    else:
        stat = compute_kl_divergence(data1,data2)
    return stat

def run_jsdiv(df1,df2,col,vals,sum_vals=False,bins=100):
    data1 = extract_columns(df1,col,vals)
    data2 = extract_columns(df2,col,vals)
    if(data1.ndim > 1):
        if(sum_vals):
            data1      = np.sum(data1,axis=1)
            data2      = np.sum(data2,axis=1)
            stat = compute_js_divergence(data1,data2,bins)
        else:    
            stat = np.zeros((data1.shape[1]))
            for i in range(0,data1.shape[1]):
                stat[i] = compute_js_divergence(data1[:,i],data2[:,i],bins)
    else:
        stat = compute_js_divergence(data1,data2)
    return stat
    
def m_KLdivergence(x, y):
  """Compute the Kullback-Leibler divergence between two multivariate samples.
  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).
  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
  """
  from scipy.spatial import cKDTree as KDTree

  # Check the dimensions are consistent
  x = np.atleast_2d(x)
  y = np.atleast_2d(y)

  n,d = x.shape
  m,dy = y.shape

  assert(d == dy)


  # Build a KD tree representation of the samples and find the nearest neighbour
  # of each point in x.
  xtree = KDTree(x)
  ytree = KDTree(y)

  # Get the first two nearest neighbours for x, since the closest one is the
  # sample itself.
  r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
  s = ytree.query(x, k=1, eps=.01, p=2)[0]

  # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
  # on the first term of the right hand side.
  return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))

"""
Stolen from: https://medium.com/datalab-log/measuring-the-statistical-similarity-between-two-samples-using-jensen-shannon-and-kullback-leibler-8d05af514b15
"""
def compute_probs(data, n=10):
    h, e = np.histogram(data, n)
    p = h/data.shape[0]
    return e, p

def support_intersection(p, q):
    return list(filter(lambda x: (x[0]!=0) & (x[1]!=0), list(zip(p, q))))

def get_probs(list_of_tuples):
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q

def kl_divergence(p, q):
    return np.sum(p*np.log(p/q))

def js_divergence(p, q):
    m = (1./2.)*(p + q)
    return (1./2.)*kl_divergence(p, m) + (1./2.)*kl_divergence(q, m)

def compute_kl_divergence(train_sample, test_sample, n_bins=10): 
    """
    Computes the KL Divergence using the support intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)
    
    return kl_divergence(p, q)

def compute_js_divergence(train_sample, test_sample, n_bins=10): 
    """
    Computes the JS Divergence using the support intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)
    
    list_of_tuples = support_intersection(p,q)
    p, q = get_probs(list_of_tuples)
    
    return js_divergence(p, q)

def plot_scatter_var(df,x,y,y_cols=None,swap=False):
    '''
    Scatter plot function x against y, y is usually variance (uncertainty)
    args: df, x, y (x and y must be same length)
    returns: None
    '''
    #Attribute extract
    if x == 'bbox_area':  # x0,y0,x1,y1
        attr_data = np.asarray(df['bbdet'].to_list())
        width  = attr_data[:,2]-attr_data[:,0]
        length = attr_data[:,3]-attr_data[:,1]
        attr_data = width * length
    elif x == 'bbox_volume':
        attr_data = np.asarray(df['bbdet3d'].to_list())
        width  = attr_data[:,3]
        length = attr_data[:,4]
        height = attr_data[:,5]
        attr_data = width * length * height
    elif x == 'distance':
        attr_data = np.asarray(df['bbdet3d'].to_list())
        x_c = attr_data[:,0]
        y_c = attr_data[:,1]
        z_c = attr_data[:,2]
        attr_data = np.sqrt(np.power(x_c,2)+np.power(y_c,2)+np.power(z_c,2))
    elif x == 'rotation':
        attr_data = np.asarray(df['bbdet3d'].to_list())
        ry = attr_data[:,6]
        attr_data = np.sin(ry)
    else:
        attr_data = np.asarray(df[x])
        if(attr_data.ndim > 1):
            attr_data = np.sum(attr_data,axis=1) 

    #Uncertainty extract
    uc_data = df[y].to_list()
    uc_data = np.asarray(uc_data)
    if y_cols is not None:
        uc_data = extract_columns(df,y,y_cols)
    if uc_data.ndim > 1:
        uc_data = np.sum(uc_data,axis=1)
        
    label = x + ' vs ' + y
    covariance = np.cov(attr_data,uc_data)
    if(swap):
        plt.scatter(uc_data,attr_data,label=label,color='r',marker='*',s=1)
        plt.xlabel(y)
        plt.ylabel(x)
    else:
        plt.scatter(attr_data,uc_data,label=label,color='r',marker='*',s=1)
        plt.xlabel(x)
        plt.ylabel(y)
    #https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/#:~:text=The%20Pearson%20correlation%20coefficient%20(named,deviation%20of%20each%20data%20sample.
    corr, _ = pearsonr(uc_data,attr_data)
    scorr, _ = spearmanr(uc_data,attr_data)
    #plt.text(.7,1,'cov = %s %s  %s %s' %(covariance[0,0],covariance[0,1],covariance[1,0],covariance[1,1])) 
    #plt.text(1.2,0,'cov = %s %s  %s %s' %(covariance[0,0],covariance[0,1],covariance[1,0],covariance[1,1]))
    print('covariance=\n', covariance[0,0],covariance[0,1],'\n',covariance[1,0],covariance[1,1]) 
    print('correlation: {} spearman correlation: {}'.format(corr,scorr))

def plot_histo_inverse_gamma(df,scene,col,val,min_val=None,max_val=None,thresh=None):
    """
    Function to plot inverse gamma using uncertainty parameters
    args: dataframe, scenetype, minval, maxval, column(to be plotted), value(1 or all), thresh (float 0-1)
    if threshold value is given, filter all variance values above it  (elimate outliers)

    column can be a_bbox_var, e_bbox_var, a_cls_var, e_cls_var (4 or 7 values - 2d or 3d)

    2d - x_c, y_c, l, w (l1, w1)
    3d - x_c, y_c, z_c, l, w, h, r_y (l2, w2)
    """
    fit = 1 
    bboxes = df.columns

    data_arr = extract_columns(df,col,val)

    if(fit):
        if (min_val == None and max_val == None):
            hist_range = (np.min(data_arr),np.max(data_arr))
            x = np.arange(np.min(data_arr),np.max(data_arr),.0001)
        else:
            hist_range = (min_val,max_val)
            x = np.arange(min_val,max_val,.0001)
        if thresh != None:
            range = np.max(data_arr) - np.min(data_arr)
            data_arr = data_arr[data_arr < range*thresh]
        shape,loc,scale = scipy_stats.invgamma.fit(data_arr)
        g1 = scipy_stats.invgamma.pdf(x=x, a=shape, loc=loc, scale=scale)
        plt.plot(x,g1,label='{} fitted_gamma: {:.3f} {:.3f} {:.3f}'.format(scene,shape,loc,scale))
        if(val is None):
            val_str = ''
        else:
            val_str = ' : {}'.format(val)
        plt.hist(data_arr,bins=200,range=hist_range,alpha=0.5,label=col + val_str,density=True,stacked=True)
        #plt.hist(data_arr,bins=200,range=[min_val,max_val],alpha=0.5,label=cil+': '+val,density=True,stacked=True)
    return 

def plot_histo_KDE(df,scene,col,val,min_val=None,max_val=None,thresh=None):
    """
    Function to plot KDE using uncertainty parameters. Can specify or not specify range (min/max)
    args: dataframe, dets, scenetype, column(to be plotted), value(1 or all), minval, maxval, thresh (float 0-1)
    if threshold value is given, filter all variance values above it  (elimate outliers)

    column can be a_bbox_var, e_bbox_var, a_cls_var, e_cls_var (4 or 7 values - 2d or 3d)

    2d - x_c, y_c, l, w
    3d - x_c, y_c, z_c, l, w, h, r_y
    """
    fit = 1 

    data_arr = extract_columns(df,col,val)
    if(fit):
        if (min_val is None):
            min_val = np.min(data_arr)
        if (max_val is None):
            max_val = np.max(data_arr)
        hist_range = (min_val,max_val)
        x = np.arange(min_val,max_val,.0001)
        if thresh != None:
            range = np.max(data_arr) - np.min(data_arr)
            data_arr = data_arr[data_arr < range*thresh]

        scipy_kernel = gaussian_kde(data_arr,bw_method=.035)  # kernel function centered on each datapoint
        pdf = scipy_kernel.evaluate(x)  # sum all functions together and normalize to obtain pdf
        h = scipy_kernel.factor * np.std(data_arr)  # smoothing parameter h also known as bandwidth
        plt.plot(x,pdf,'k')
        plt.hist(data_arr,bins=200,range=hist_range,alpha=0.5,label=col + ': ' + val,density=True,stacked=True)
        print("h/bandwidth value = ", h)
    return 

def plot_histo_multivariate(df,plotname,col,vals,min_val=None,max_val=None,plot=False):
    """
    Function to multivariate KDE. vals is a list of strings to obtain multiple entries
    args: dataframe, df, scenetype, column(to be plotted), value(1 or all), minval, maxval

    column can be a_bbox_var, e_bbox_var, a_cls_var, e_cls_var (4 or 7 values - 2d or 3d)

    """
    data = extract_columns(df,col,vals)
    if(data is not None):
        if (min_val is None):
            min_val = np.min(data)
        if (max_val is None):
            max_val = np.max(data)
        hist_range = (min_val,max_val)
        for i in range(0,data.shape[1]):
            labelname = plotname + ': ' + vals[i]
            plt.hist(data[:,i],bins=200,range=(hist_range),alpha=0.5,label=labelname,density=True,stacked=True)
        if(plot):
            plt.legend()
            plt.show()
    else:
        print('specified column does not exist')
    return

def _col_data_extract(df,c,vals):
    data = None
    if(c == 'e_bbox_var+a_bbox_var'):
        a_bbox_var = np.asarray(df['a_bbox_var'].to_list())
        e_bbox_var = np.asarray(df['e_bbox_var'].to_list())
        data = a_bbox_var + e_bbox_var
    elif(c == 'e_bbox_var,a_bbox_var'):
        a_bbox_var = np.asarray(df['a_bbox_var'].to_list())
        e_bbox_var = np.asarray(df['e_bbox_var'].to_list())
        data = np.concatenate((e_bbox_var,a_bbox_var),axis=1)
    elif(c == 'e_cls_var+a_cls_var'):
        a_cls_var = np.power(np.asarray(df['a_cls_var'].to_list()),2)
        e_cls_var = np.asarray(df['e_cls_var'].to_list())
        data = a_cls_var + e_cls_var
    elif(c == 'e_cls_var,a_cls_var'):
        a_cls_var = np.power(np.asarray(df['a_cls_var'].to_list()),2)
        e_cls_var = np.asarray(df['e_cls_var'].to_list())
        data = np.concatenate((e_cls_var,a_cls_var),axis=1)
    elif(c == 'all_var'):
        a_bbox_var = np.asarray(df['a_bbox_var'].to_list())
        e_bbox_var = np.asarray(df['e_bbox_var'].to_list())
        a_cls_var = np.power(np.asarray(df['a_cls_var'].to_list()),2)
        e_cls_var = np.asarray(df['e_cls_var'].to_list())
        data = np.concatenate((e_bbox_var,e_cls_var,a_bbox_var,a_cls_var),axis=1)
    else:
        if(c in df.columns):
            col_vals = vals_postprocess(vals)
            data = np.asarray(df[c].to_list())
            data = data[:,col_vals]
            if(c == 'e_bbox_var'):
                data = data*1000
            if(c == 'a_cls_var'):
                data = np.power(data,2)*1000
        else:
            data = None
            print('column {} undefined'.format(c))
    return data

def vals_postprocess(vals):
    col_vals = []
    if(vals is None):
        col_vals = None
    else:
        for val in vals:
            col_vals.append(column_value_to_index(val))
    return col_vals

def extract_columns(df,col,vals):
    data = _col_data_extract(df,col,vals)
    return data

#Plot IQR and box plots
def plot_box_plot(df,col,vals,plot=False):
    data = extract_columns(df,col,vals)
    if(data.ndim > 1):
        data = np.sum(data,axis=1)
    if(plot):
        plt.boxplot(data,labels=[col],vert=False)
        plt.scatter(data,np.ones((data.shape[0])))
    median = np.median(data)
    mean   = np.mean(data)
    upper_quartile = np.percentile(data, 75)
    lower_quartile = np.percentile(data, 25)

    iqr = upper_quartile - lower_quartile
    upper_whisker = data[data<=upper_quartile+1.5*iqr].max()
    lower_whisker = data[data>=lower_quartile-1.5*iqr].min()
    return [mean,median,lower_quartile,upper_quartile,lower_whisker,upper_whisker]

#Plot ROC by sweeping density_thresh
def plot_roc_curves(df,col,vals,m_kde_tp,m_kde_fp,min_val=None,max_val=None,limiter=0):
    data = extract_columns(df,col,vals)
    if(limiter != 0):
        print(len(df.index))
        frac = (limiter+0.1)/(len(df.index)+0.1)
        df = df.sample(frac=1)
        df = df.sample(frac=frac)
        print(len(df.index))
    f_a  = []
    hits = []
    signal_tp = np.asarray(df['difficulty'].to_list(),dtype=np.int32)
    signal_tp = np.where(signal_tp != -1, True, False)
    if(data is not None):
        np.random.seed(int.from_bytes(os.urandom(4), sys.byteorder))
        np.random.shuffle(data)
        if(data.shape[0] > limiter and limiter != 0):
            data = data[:limiter,:]
        tp_densities = m_kde_tp.pdf(data)
        fp_densities = m_kde_fp.pdf(data)
        if (min_val is None):
            min_val = np.min(tp_densities-fp_densities)
        if (max_val is None):
            max_val = np.max(tp_densities-fp_densities)
        thresh_list = np.linspace(min_val,max_val,25)
        for thresh in thresh_list:
            ratios = find_ratios(df,col,vals,signal_tp,tp_densities,fp_densities,min_thresh=thresh)
            hits.append(ratios[0])
            f_a.append(ratios[2])
        plt.plot(f_a,hits,label='col {} vals {}'.format(col,vals))
        print('auc: {}'.format(auc(f_a,hits)))
    else:
        return None

#Find where function falls below a threshold
def find_kde_roots(m_kde,min_val,max_val, density_thresh):
    roots = opt.brentq(lambda x: m_kde(x) - density_thresh,min_val,max_val)
    print(roots)
    return None

def classify_dets(df,col,vals,tp_kde,fp_kde,min_thresh=0.0):
    data = extract_columns(df,col,vals)
    tp_densities = tp_kde.pdf(df)
    fp_densities = fp_kde.pdf(df)
    response_tp = np.where(tp_densities > fp_densities + min_thresh,True,False)
    response_fp = np.bitwise_not(response_tp)
    return [np.sum(response_tp),np.sum(response_fp)]

#From SDT, find hit ratio, miss ratio, etc. etc. based on TP/FP
def find_ratios(df,col,vals,signal_tp,tp_densities,fp_densities,min_thresh=0.0):
    ratios    = np.zeros((4),)
    signal_fp = np.bitwise_not(signal_tp)
    response_tp = np.where(tp_densities > fp_densities + min_thresh,True,False)
    response_fp = np.bitwise_not(response_tp)
    #response_tp = np.where(tp_densities > min_thresh,True,False)
    #fp_densities = None
    hit               = np.bitwise_and(response_fp,signal_fp)
    miss              = np.bitwise_and(response_tp,signal_fp)
    false_alarm       = np.bitwise_and(response_fp,signal_tp)
    correct_rejection = np.bitwise_and(response_tp,signal_tp)

    #hit_density  = fp_densities[hit]
    #miss_density = fp_densities[miss]
    #f_a_density  = tp_densities[false_alarm]
    #c_r_density  = tp_densities[correct_rejection]

    ratios[0]  = np.sum(hit)/np.sum(signal_fp)
    ratios[1]  = np.sum(miss)/np.sum(signal_fp)
    ratios[2]  = np.sum(false_alarm)/np.sum(signal_tp)
    ratios[3]  = np.sum(correct_rejection)/np.sum(signal_tp)
    return ratios

def plot_histo_multivariate_KDE(df,plotname,col,vals,min_val=None,max_val=None,plot=False,bins=200):
    """
    Function to multivariate KDE. vals is a list of strings to obtain multiple entries
    args: dataframe, plotname, column(to be plotted), value(1 or all), minval, maxval

    column can be a_bbox_var, e_bbox_var, a_cls_var, e_cls_var (4 or 7 values - 2d or 3d)

    """
    x_list = []
    col_vals = []
    if(vals is not None):
        num_col = len(vals)
    else:
        num_col = 1
        col_vals = None
    data_arr = extract_columns(df,col,vals)
    if(data_arr is None):
        return None
    if(min_val is None):
        min_val = np.min(data_arr,axis=0)
    else:
        min_val = np.ones((data_arr.shape[1]))*min_val
    if(max_val is None):
        max_val = np.ones((data_arr.shape[1]))*np.max(data_arr)
    else:
        max_val = np.ones((data_arr.shape[1]))*max_val
    ranges = np.linspace(min_val,max_val,bins)
    hist_range = (min_val,max_val)
    
    #myPDF,axes = fastKDE.pdf(data_arr[0,:],data_arr[1,:])
    #Extract the axes from the axis list
    #v1,v2 = axes

    #Plot contours of the PDF should be a set of concentric ellipsoids centered on
    #(0.1, -300) Comparitively, the y axis range should be tiny and the x axis range
    #should be large
    #PP.contour(v1,v2,myPDF)
    #PP.show()

    #df = pd.DataFrame({'x_c': data_arr[:,0], 'y_c': data_arr[:, 0]})
    #sns.jointplot(x='x_c',y='y_c', data=df, kind='kde')
    #scipy_kernel = gaussian_kde(data_arr,bw_method=.035)  # kernel function centered on each datapoint
    #pdf = scipy_kernel.evaluate(x_list)  # sum all functions together and normalize to obtain pdf
    #plt.plot(x_list[0,:],pdf)
    #plt.show()
    

    #x_mesh = np.meshgrid(x_list[:,0],x_list[:,1])
    #x_mesh = x_mesh.swapaxes(0,2)
    v_type = ''
    for i in range(0,data_arr.shape[1]):
        v_type = v_type + 'c'
    multivariate_kernel = sm.nonparametric.KDEMultivariate(data_arr, var_type=v_type, bw='normal_reference')
    h = multivariate_kernel.bw
    #Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, vlag, vlag_r, winter, winter_r
    if(plot):
        label_vals = []
        for i in range(0,len(vals)):
            if(vals[i] == 'l1' or vals[i] == 'l2'):
                label_vals.append('l')
            elif(vals[i] == 'w1' or vals[i] == 'w2'):
                label_vals.append('w')
            else:
                label_vals.append(vals[i])

        #if(num_col == 1):
        #    pdf_eval = multivariate_kernel.pdf(ranges)
        #    if(vals is None):
        #        val_str = ''
        #    else:
        #        val_str = ' : {}'.format(vals[0])
        #    labelname = plotname + val_str
        #    #plt.plot(ranges,pdf_eval)
        #    plt.hist(data_arr[:,0],bins=bins,range=(min_val[0],max_val[0]),alpha=0.5,label=labelname,density=True,stacked=True)
        if(num_col == 2):
            x_list  = np.swapaxes(np.asarray(np.meshgrid(ranges[:,0],ranges[:,1])),0,2)
            pdf_eval = multivariate_kernel.pdf(x_list.reshape(-1,num_col))
            pdf_eval = pdf_eval.reshape(bins,bins)
            if(plotname == 'TP'):
                contour_name = 'True Positives'
                c_map = 'Blues'
            elif(plotname == 'FP'):
                contour_name = 'False Positives'
                c_map = 'Reds'
            else:
                c_map = 'Accent'
                contour_name = '{}, {}'.format(col,label_vals)
            contour = plt.contourf(x_list[:,:,0],x_list[:,:,1],pdf_eval, cmap=c_map, label=contour_name, alpha=0.5)
            #plt.clabel(contour, inline=True, fontsize=8)#
            #plt.imshow(pdf_eval, extent=[0, np.max(x_list[:,:,0])*0.80, 0, np.max(x_list[:,:,1])*0.80], origin='lower',
            #cmap=c_map, alpha=0.5)
            #Gross, eh? :)
            plt.xlabel('\u03C3[{}]^2'.format(label_vals[0]))
            plt.ylabel('\u03C3[{}]^2'.format(label_vals[1]))
            plt.title(col)
            clb = plt.colorbar()
            clb.ax.set_title(contour_name,pad=20)
        else:
            if(plotname == 'TP'):
                plotname = 'True Positives'
            elif(plotname == 'FP'):
                plotname = 'False Positives'
            for i in range(0,data_arr.shape[1]):
                labelname = plotname + ': ' + label_vals[i]
                plt.hist(data_arr[:,i],bins=200,range=(min_val[i],max_val[i]),alpha=0.5,label=labelname,density=True,stacked=True) 
            plt.xlabel('\u03C3^2')
            plt.ylabel('density')
            plt.title(col)
            
        #plt.show()
        #x = np.linspace(np.min(data_arr[0,:]),np.max(data_arr[0,:]),len(pdf))
        #plt.plot(x,pdf,'k')
        #for count,row in enumerate(data_arr):
        #    plt.hist(row,bins=200,range=hist_range,alpha=0.5,label=col + ': ' + str(vals[count]),density=True,stacked=True)
        print_str = '{} bw values: ['.format(plotname)
        for i in range(0,data_arr.shape[1]):
            print_str += ' {:.3f}'.format(h[i])
        print_str += ']'
        print(print_str)
    return multivariate_kernel

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
    elif (val == 'fg'):
        return 1
    elif (val == 'bg'):
        return 0
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
            #data_truncated = data[data < range*0.20]
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