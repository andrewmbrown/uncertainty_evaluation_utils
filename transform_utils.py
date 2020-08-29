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
        [sl*c,-sw*s,  0,tx],
        [sl*s, sw*c,  0,ty],
        [   0,    0, sh,tz],
        [   0,    0,  0, 1]])

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

def bbdet3d_to_bbdet2d(df,top_crop):
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
            bbox2D = [-1,-1,-1,-1]  # not inside image frame
            bbox2D_col.append(bbox2D)
            continue
        bbox2D_clipped = clip_2d_bounding_box(im_shape, bbox2D)
        truncation     = compute_truncation(bbox2D,bbox2D_clipped)
        if truncation > .90:  # last check if box is too far clipped
            bbox2D = [-1,-1,-1,-1]  # not inside image frame
            bbox2D_col.append(bbox2D)
            continue
        bbox2D = [bbox2D[0], bbox2D[1]-top_crop, bbox2D[2], bbox2D[3]-top_crop]
        bbox2D_col.append(bbox2D)
    df['bbdet3d_2d'] = bbox2D_col
    return df


if __name__ == '__main__':
    print('cannot run file stand-alone')