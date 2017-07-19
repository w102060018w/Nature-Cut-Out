# Nature-Cut-Out

This is the python implementation of cutting out a nature contour aroung human body.

## Result

I've tested on several images and in most cases the results work pretty well.

<div align="center">
<img src="./input/testImg8.jpg" height="280px">
<img src="./output/8_Bezier_modified_contour.jpg" height="280px">
<img src="./input/testImg11.jpg" height="280px">
<img src="./output/11_Bezier_modified_contour.jpg" height="280px">
</div>

<div align="center">
<img src="./input/testImg13.jpg" height="280px">
<img src="./output/13_Bezier_modified_contour.jpg" height="280px">
<img src="./input/testImg19.jpg" height="280px">
<img src="./output/19_Bezier_modified_contour.jpg" height="280px">
</div>

<div align="center">
<img src="./input/testImg23.jpg" height="280px">
<img src="./output/23_Bezier_modified_contour.jpg" height="280px">
<img src="./input/testImg9.jpg" height="280px">
<img src="./output/9_Bezier_modified_contour.jpg" height="280px">
</div>

## Setup

Recommended running environment:
* Mac OS X El Capitan (version 10.11.6) 
* Python 3.6.1

Library:
* OpenCV 3.3.0-rc
* Scipy 0.19.1
* Shapely 1.5.17
* math
* Numpy 1.13.1
* Bezier

Optional library:
* descartes
* matplotlib

Using pip to install all these library would be recommended:
```
pip install the-lib-you-want-to-install
```
Also, if you stuck in some problems when installing OpenCV with Python bindings, I will recommend following [this tutorial](http://www.pyimagesearch.com/2016/12/05/macos-install-opencv-3-and-python-3-5/) written by [Adrian Rosebrock](http://www.pyimagesearch.com/author/adrian/).

## Usage
#### Basic usage:
```
python HPE_NatureCutout.py
```
It will run 27 images in the **./input** folder at one time, and show an output image once at a time, press 'esc' to see next output image.

#### Output:

Output will all be saved to the **./New_Output** folder. Each input will generate 3 outputs, including the result simple base on Human-Pose-Estimation, the result after applying Alpha-Shape and the result after using 4-point Bézier curve.

<div align="center">
<img src="./output/7_Base_On_HPE.jpg" height="280px">
<img src="./output/7_Alpha_Shape_modified_contour.jpg" height="280px">
<img src="./output/7_Bezier_modified_contour.jpg" height="280px">
</div>
<div align="center">
<img src="./output/20_Base_On_HPE.jpg" height="280px">
<img src="./output/20_Alpha_Shape_modified_contour.jpg" height="280px">
<img src="./output/20_Bezier_modified_contour.jpg" height="280px">
</div>


#### Run on your own images:

Please first go to [this website](https://fling.seas.upenn.edu/~xiaowz/dynamic/wordpress/shapeconvex/) and scroll down to the bottom to download the matlab code on constructing 2D and 3D human pose. Save the 2D human pose result as the **.mat** file:
```
filename = 'testImg21'
fname = strcat('./pred_2d',filename(8:end));
save(fname,'preds_2d');
```
which 'preds_2d' is the parameter's name.

Put both your .mat files and images in the input folder, and make sure the format and the name of files are the same as mine in the input folder. 
