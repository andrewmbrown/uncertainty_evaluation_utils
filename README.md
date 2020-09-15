Hello! Welcome to this repository :) 
This work showcases the work Mathew Hildebrand (@mathild7) and I acomplished during my Summer Resarch at UToronto. 
Mat Hildebrand's thesis: Multimodal Object Detection (2D/3D) with Uncertainty Analysis


Description: This repo serves as a library of functions to evaluate uncertainty outputs from a deep learning model (RCNN). These could be used for 2D or 3D predictors (we used img and LiDAR). To see sample outputs we were able to acomplish, check out the 'plots' directory. 
The general flow of this script is:
1. Build a pandas dataframe of the uncertainty output file 
2. Add extra data to dataframe from our labels file (tod, ground truth info...)
3. Execute helper functions on the dataframe (histogram plotting, PDF curve fititng, Average Precison calculation, contour plotting, Kernel Density Estimation)

Usage: To use this repository (FULLY) two files are needed:
        1. text file specifying model output with uncertainty values (a_bbox_var, a_cls_var ...) 
        2. A labels file specifying extra info (such as time of day) (this file is optional) 
        
Realistically, this repository should be used as a helper library for somebody preforming post-processing. This script was customized 
for Mat and I, so it would be difficult to implement the entire script. 
Use this script as a template for performing (histogram plotting, PDF curve fititng, Average Precison calculation, contour plotting, Kernel Density Estimation)

If somebody stumbles upon this repo and has questions, message me on github or email me (andrew.brown.csf@gmail.com) 
Thanks! 
