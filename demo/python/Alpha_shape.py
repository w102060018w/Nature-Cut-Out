'''
author : Sean Gillies
edit : Winnie Hong
create date : 0714/2017
'''

import sys
sys.path.append('/Users/pointern/.virtualenvs/cv3/lib/python3.6/site-packages')

from scipy.spatial import Delaunay
from descartes import PolygonPatch

from matplotlib.collections import LineCollection
from shapely.ops import cascaded_union, polygonize
from shapely.geometry import MultiLineString
import shapely.geometry as geometry
import math
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def add_edge(edges,edge_points, coords, i, j):
    """
    Add a line between the i-th and j-th points,
    if not in the list already
    """
    if (i, j) in edges or (j, i) in edges:
        # already added
        return

    edges.add( (i, j) )
    edge_points.append(coords[ [i, j] ])


# loop over triangles:
# ia, ib, ic = indices of corner points of the triangle
def alpha_shape(points,alpha):
#    circum_lst = []
    tri = Delaunay(points)
    
    edges = set()
    edge_points = []
    
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
    
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
    
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
    
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
    
        circum_r = a*b*c/(4.0*area)
    
        # Here's the radius filter.
        if circum_r < 1.0/alpha: 
            add_edge(edges,edge_points, points, ia, ib)
            add_edge(edges,edge_points, points, ib, ic)
            add_edge(edges,edge_points, points, ic, ia)
            
#    print('ave = ',sum(circum_lst)/len(circum_lst))
            
    m = MultiLineString(edge_points)
    triangles = list(polygonize(m))
    
#    ## Show Result
#    # shoaw result of triangular
#    lines = LineCollection(edge_points,linewidths=(0.5, 1, 1.5, 2))
#    plt.figure()
#    plt.title('Alpha=2.0 Delaunay triangulation')
#    plt.plot(points[:,0], points[:,1], 'o', hold=1, color='#f16824')
#    plt.gca().add_collection(lines)
#    
#    # show result of connected contour
#    plt.figure()
#    plt.title("Alpha=2.0 Hull")
#    plt.gca().add_patch(PolygonPatch(cascaded_union(triangles), alpha=0.5))
#    plt.gca().autoscale(tight=False)
#    plt.plot(points[:,0], points[:,1], 'o', hold=1)
#    plt.show()
    
    return triangles, edge_points

if __name__ == '__main__' :

    # pts of extending points calculated from HPE result
    points = np.array([[79, 441], [85, 370], [150, 447], [98, 360], [173, 427], [166, 285], [241, 352], \
    [372, 320], [337, 413], [270, 282], [235, 375], [501, 445], [480, 377], [433, 466], [205, 238], \
    [298, 276], [254, 234], [299, 145], [214, 216], [257, 125], [338, 157], [307, 252], [342, 81], \
    [274, 98], [359, 149], [58, 68], [127, 52], [74, 137], [190, 91], [145, 180], [365, 163], [344, 260], \
    [446, 194], [387, 154], [406, 253]])
    

    alpha = 0.013
    
    triangles, edge_points = alpha_shape(points, alpha)
    
    



