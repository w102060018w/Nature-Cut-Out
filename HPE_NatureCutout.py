'''
Author: Winnie Hong
Date: 07.13.2017 16:36
Model of human pose estimaton : Sparseness Meets Deepness (CVPR2016) 
Goal : Try to see if human pose can cut out in a pretty nature way(but not really close to the edge).
'''
import os
import sys
sys.path.append('/Users/pointern/.virtualenvs/cv3/lib/python3.6/site-packages')

import scipy.io
import numpy as np
import cv2
import pandas as pd
from descartes import PolygonPatch
from matplotlib.collections import LineCollection
from shapely.ops import cascaded_union
import bezier

import matplotlib as mpl
mpl.use('TkAgg') # in order to avoid default backends error
import matplotlib.pyplot as plt

from Alpha_shape import alpha_shape



## Preprocess of Data
# read image
fdir = './input/'
# file_id = '01'
file_id_list = list(range(23,24))
file_id_list = [str(x) for x in file_id_list]

for k, file_id in enumerate(file_id_list):

	filename = 'testImg'+file_id+'.jpg'
	img = cv2.imread(fdir+filename)
	hei,wid,dep = img.shape

	# read data from human pose 16 points in the .mat file
	mat = scipy.io.loadmat(fdir+'pred_2d'+file_id+'.mat')
	landmark = mat['preds_2d']
	x_posi = landmark[0]
	y_posi = landmark[1]
	maxDif_ydir = max(y_posi)-min(y_posi)
	landmark_N = len(x_posi) # calculate the human pose landmark number (in this case should be 16 points)

	# save each position as the value in Dict
	posi_Dict = {}
	idx_Dict = {}
	posi_name = ['RAnk','RKne','RHip','LHip','LKne','LAnk','Pelv','Thrx','Neck','Head','RWri','RElb','RSho','LSho','LElb','LWri']
	for i in range(landmark_N):
		posi_Dict[posi_name[i]] = np.array([int(x_posi[i]),int(y_posi[i])])
		idx_Dict[posi_name[i]] = i

	# add special landmark of 'stom' to make it more easy on finding contour points between stomach
	stom_x = (posi_Dict['Thrx'][0]+posi_Dict['Pelv'][0])/2 
	stom_y = (posi_Dict['Thrx'][1]+posi_Dict['Pelv'][1])/2
	posi_Dict['Stom'] = [stom_x,stom_y]

	def ConnectLandmark(str1,str2,dataframe):
		dataframe.set_value(str1,str2,1)
		dataframe.set_value(str2,str1,1)

	## Build Connection Matrix (but actually only need the info in the upper-triangle)
	columns = posi_name
	index = posi_name
	df = pd.DataFrame(0,index = index, columns = columns)
	ConnectLandmark('Head','Neck',df)
	ConnectLandmark('Thrx','Neck',df)
	ConnectLandmark('Thrx','LSho',df)
	ConnectLandmark('Thrx','RSho',df)
	ConnectLandmark('Thrx','Pelv',df)
	ConnectLandmark('LSho','LElb',df)
	ConnectLandmark('LElb','LWri',df)
	ConnectLandmark('RSho','RElb',df)
	ConnectLandmark('RElb','RWri',df)
	ConnectLandmark('Pelv','RHip',df)
	ConnectLandmark('Pelv','LHip',df)
	ConnectLandmark('RHip','RKne',df)
	ConnectLandmark('RAnk','RKne',df)
	ConnectLandmark('LKne','LHip',df)
	ConnectLandmark('LKne','LAnk',df)

	# print(df)

	## Build Vector between landmarks
	vec_Dict = {}
	idx = 0
	for row, col in df.iterrows():
		for j in range(idx,len(col)): # only search upper triangular part
			if col[j] == 1: # find connection, then build vector.
				vecH = row
				vecT = col.index[j]
				vec_Dict[vecH+'_'+vecT] = posi_Dict[vecT]-posi_Dict[vecH]

				# normalization
				vec_Dict[vecH+'_'+vecT] = vec_Dict[vecH+'_'+vecT]/np.linalg.norm(vec_Dict[vecH+'_'+vecT])
		idx += 1

	## Start marking out the extending point
	# set the width we would like to extend from the considered-point
	width = np.linalg.norm(posi_Dict['LHip']-posi_Dict['RHip'])

	# build clear relation between vector and its considered-point 
	# (i.e. a vector is built from 2 points(considered-points), we have to make it clear which point is we would like to extend out base on this vector)
	VecExtenPt_Dict = {}
	VecExtenPt_Dict['RAnk_RKne'] = ['RKne','RAnk']
	# VecExtenPt_Dict['RKne_RHip'] = ['RKne','RHip']
	VecExtenPt_Dict['RKne_RHip'] = ['RHip']
	# VecExtenPt_Dict['RHip_Pelv'] = 'RHip'
	# VecExtenPt_Dict['LHip_LKne'] = ['LKne','LHip']
	VecExtenPt_Dict['LHip_LKne'] = ['LHip']
	# VecExtenPt_Dict['LHip_Pelv'] = 'LHip'
	VecExtenPt_Dict['LKne_LAnk'] = ['LKne','LAnk']
	VecExtenPt_Dict['Pelv_Thrx'] = ['Stom']
	VecExtenPt_Dict['Thrx_Neck'] = ['Neck']
	# VecExtenPt_Dict['Thrx_RSho'] = ['RSho']
	# VecExtenPt_Dict['Thrx_LSho'] = ['LSho']
	VecExtenPt_Dict['Neck_Head'] = ['Head']
	VecExtenPt_Dict['RWri_RElb'] = ['RElb','RWri']
	VecExtenPt_Dict['RElb_RSho'] = ['RSho']
	VecExtenPt_Dict['LSho_LElb'] = ['LSho']
	VecExtenPt_Dict['LElb_LWri'] = ['LElb','LWri']

	# calculate extended-points regardless of its order
	extendPt = []
	extendPt_Draw = []
	vtxs = ['Head','LWri','RWri','LAnk','RAnk'] # special cases on the 5 vertex pts
	vtxs_WidFactor = [0.7]+[1.5]*4 # special 
	vtxs_PairPts = ['Neck','LElb','RElb','LKne','RKne']

	for key, value in vec_Dict.items():
		# eg. key = RAnk_RKne; VecExtenPt_Dict['RAnk_RKne'] = 'RAnk'; posi_Dict['RAnk'] will get the posi.
		try :
			for i,ele in enumerate(VecExtenPt_Dict[key]): # usually enumerate once, but like in ['LKne','LHip'], we have to apply same norm on multi-considered-pts.
				extendpt = posi_Dict[ele]

				# special case on 5 extending pts(head,LWrist,RWrist,LAnkle,RAnkle), 
				'''
				Now I recalculate again the vector of these special vertex, since I am not sure 
				if the direction of previous calculated vector is what I want(have to look at the vec_Dict key name though..), 
				if I need choose on testing on its direction and then adjust it, it would be too non-intuitive.
				'''
				for idx, vertex in enumerate(vtxs):
					if ele == vertex:
						pair = vtxs_PairPts[idx]
						slope = (posi_Dict[vertex] - posi_Dict[pair])/np.linalg.norm(posi_Dict[vertex] - posi_Dict[pair])
						extendPt.append([int(x) for x in (extendpt+width*slope*vtxs_WidFactor[idx])]) # why multiply by 1.5 -> simply the exp. result, so it will look more nature.
				
				norm = np.array([value[1],-value[0]])
				extendPt.append([int(x) for x in (extendpt+width*norm)])
				extendPt.append([int(x) for x in (extendpt+width*(-norm))])
		except KeyError:
			continue

	# for drawing purpose
	extendPt_Draw = [[x] for x in extendPt]
	extendPt = np.array(extendPt)

	## Start Connecting These extend-pts (using Bezier Curve)
	# use Alpha shape to remain those really important pts.
	# [Note]: with larger alpha-value, we can find more fitting contour (but can't be too large(i.e. threshold would be too small, and so no more triangles could be added into the contour set, which will cause error.))
	alpha = 3.5/maxDif_ydir # why 3.5? -> it's just exp. result. [eg. 1200 pt in 'maximum-y-diff', we will use 0.003 as alpha-value]
	triangles, edge_points = alpha_shape(extendPt, alpha)

	## shoaw result of Delauny-triangular
	lines = LineCollection(edge_points,linewidths=(0.5, 1, 1.5, 2))
	plt.figure()
	plt.title('Alpha=2.0 Delaunay triangulation')
	plt.plot(extendPt[:,0], extendPt[:,1], 'o', hold=1, color='#f16824')
	plt.gca().add_collection(lines)

	## show result of connected contour
	plt.figure()
	plt.title("Alpha=2.0 Hull")
	plt.gca().add_patch(PolygonPatch(cascaded_union(triangles), alpha=0.5))
	plt.gca().autoscale(tight=False)
	plt.plot(extendPt[:,0], extendPt[:,1], 'o', hold=1)
	plt.show()

	#%%
	# extract vertices of Polygon
	ExtPts_x, ExtPts_y = cascaded_union(triangles).exterior.coords.xy #extract these control points from alpha-shape-result
	ExtPts = []
	for i in range(len(ExtPts_x)):
	    ExtPts.append([int(ExtPts_x[i]),int(ExtPts_y[i])])

	ExtPts_Draw = [[x] for x in ExtPts]


	def CheckInOrderOrNot(Pts):
		colorAry = [(0,0,204),(0,0,255),(51,51,255),(102,102,255),(153,153,255),(204,204,255),\
		(0,102,204),(0,128,255),(51,153,255),(102,178,255),(153,204,255),(204,229,255),\
		(0,204,204),(0,255,255),(51,255,255),(102,255,255),(153,255,255),(204,255,255),\
		(0,204,102),(0,255,128),(51,255,153),(102,255,178),(153,255,204),(204,255,229)]

		filename = 'testImg'+file_id+'.jpg'
		img = cv2.imread(fdir+filename)	
		hei,wid,dep = img.shape
		newHei = 600
		newWid = int(wid*newHei/hei)

		imcopy = img.copy()
		imcopy = cv2.resize(imcopy,(newWid, newHei), interpolation = cv2.INTER_CUBIC)

		for i in range(len(Pts)):
			imcopy = cv2.circle(imcopy,(Pts[i][0],Pts[i][1]),5,colorAry[i%len(colorAry)],3)

		cv2.imshow('Check_contour_inOrder_OrNot',imcopy)

		k = cv2.waitKey(0)

		if k==27:
			cv2.destroyAllWindows()	

	# check if it's in clock wise or counter clockwise
	# CheckInOrderOrNot(ExtPts)

	#%%

	# use 4 points Bezier Curve
	# inserts more points
	Dense_BezierCurve = []
	insertN = 30
	for idx,ele in enumerate(ExtPts):
	    Dense_BezierCurve.append(ele)
	    if idx < len(ExtPts)-1:
	        for i in range(1,insertN+1):
	            insert_x = int(ele[0]*((insertN-i)/insertN)+ExtPts[idx+1][0]*(i/insertN))
	            insert_y = int(ele[1]*((insertN-i)/insertN)+ExtPts[idx+1][1]*(i/insertN))
	            Dense_BezierCurve.append([insert_x,insert_y])

	# draw Bezier Curve Result      
	BezierCurve = bezier.Curve(np.array(Dense_BezierCurve), degree = 4)
	ax = BezierCurve.plot(num_pts=256)
	plt.show()

	s_vals = np.linspace(0.0, 1.0, 100)
	curve_pts = BezierCurve.evaluate_multi(s_vals)
	curve_pts_draw = [[[int(x[0]),int(x[1])]] for x in curve_pts]
	# CheckInOrderOrNot(Dense_BezierCurve)  



	## Show the result
	# in case the image is too big to show on the screen.
	newHei = 600
	ratio = newHei/hei
	newWid = int(wid*ratio)

	ptS = int(hei/100) # why 100? -> it's just exp. result.

	# img_HPE_extend = img.copy()
	# cv2.drawContours(img_HPE_extend, np.asarray(extendPt_Draw), -1, (128,255,0), ptS)
	# img_HPE_extend = cv2.resize(img_HPE_extend,(newWid, newHei), interpolation = cv2.INTER_CUBIC)
	# cv2.imshow('Base_On_HPE',img_HPE_extend)

	# img_AlphaShape = img.copy()
	# cv2.drawContours(img_AlphaShape, np.asarray(ExtPts_Draw), -1, (128,255,0), ptS)
	# img_AlphaShape = cv2.resize(img_AlphaShape,(newWid, newHei), interpolation = cv2.INTER_CUBIC)
	# cv2.imshow('Alpha_Shape_modified_contour',img_AlphaShape)

	img_NatureCurve = img.copy()
	cv2.drawContours(img_NatureCurve, np.asarray(curve_pts_draw), -1, (128,255,0), ptS)
	img_NatureCurve = cv2.resize(img_NatureCurve,(newWid, newHei), interpolation = cv2.INTER_CUBIC)
	cv2.imshow('Bezier_modified_contour',img_NatureCurve)

	k = cv2.waitKey(0)
	if k==27:
		cv2.destroyAllWindows()	

	## Save Output Images
	outdir = './New_Output/'
	if not os.path.exists(outdir):
	    os.makedirs(outdir)
	cv2.imwrite( outdir+file_id+"_"+"Base_On_HPE.jpg",img_HPE_extend);
	cv2.imwrite( outdir+file_id+"_"+"Alpha_Shape_modified_contour.jpg",img_AlphaShape);
	cv2.imwrite( outdir+file_id+"_"+"Bezier_modified_contour.jpg",img_NatureCurve);


