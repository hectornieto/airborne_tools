# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 09:17:01 2017

@author: hector
"""
import numpy as np
from osgeo import ogr
import osr
import os
from airborne_tools import georeference as georef

def write_flight_runs_vector(start_points,end_points,outshapefile,prj):


    DriverName = "ESRI Shapefile"
    driver = ogr.GetDriverByName(DriverName)
    if os.path.exists(outshapefile):
        driver.DeleteDataSource(outshapefile)
    
    data_source = driver.CreateDataSource(outshapefile)
    # create the layer
    srs=osr.SpatialReference()
    srs.ImportFromWkt(prj)
    layer = data_source.CreateLayer("Transformation Vectors", srs, ogr.wkbLineString)

    for start_point,end_point in zip(start_points,end_points):
        # create the feature
        feature = ogr.Feature(layer.GetLayerDefn())
 
        line = ogr.Geometry(ogr.wkbLineString)
        line.AddPoint(start_point[0],start_point[1])
        line.AddPoint(end_point[0],end_point[1])
          # Set the feature geometry using the point
        feature.SetGeometry(line)
        # Create the feature in the layer (shapefile)
        layer.CreateFeature(feature)
        # Destroy the feature to free resources
        feature.Destroy()
    # Destroy the data source to free resources
    data_source.Destroy()
    return True
    
def get_flight_lines_paramters(altitude,total_swath,camera_size,focal_length,overlap=0.7):
    pixel=[0,0]# Use the top left corner of the image to calculate maximum acros and along distances
    on_ground_coord=[0,0,0]
    c_xy=np.asarray(camera_size)/2.# Assume Principal Point at the center of the camera
    rotation=np.eye(3)# Assume nadir viewing rotation matrix
    half_across,half_along=georef.coordinates_collinearity(pixel, c_xy, altitude, on_ground_coord, rotation, focal_length)
    
    witdh=2*abs(half_across)
    height=2*abs(half_along)
    image_size=(witdh,height)
    separation=(1.0-overlap)*witdh
    if witdh>total_swath:
        n_runs=1
    else:
        n_runs=int(np.ceil(np.abs(total_swath)/separation))
    resolution=witdh/camera_size[0],height/camera_size[1]
    return n_runs,separation,image_size,resolution
    
def get_total_swath(flight_azimuth, point_1, point_2):
    
    if abs(flight_azimuth)==0 or abs(flight_azimuth)==180 or abs(flight_azimuth)==360:
        total_swath=abs(point_1[0]-point_2[0])# If plane flight N-S total_swath is maxX-minX
    elif abs(flight_azimuth)==90 or abs(flight_azimuth)==270:
        total_swath=abs(point_1[1]-point_2[1])# If plane flight E-W total_swath is maxY-minY
    else:
        # We calculate the distance between two parallel lines at the two extremes of a diagonal
        #cartesian_angle=np.radians(-(flight_azimuth)+90)# Convert azimuthal angle to cartesian, i.e. 90 azimumth is 0
        slope=np.cos(np.radians(flight_azimuth))/np.sin(np.radians(flight_azimuth))
        total_swath=get_distance_between_parallel(point_1, point_2, slope)
    return total_swath
            
def point_slope_formula(x1,y1,m):
    #y-y1=slope(x-x1)
    b=-m*x1+y1
    return b
    
def get_distance_between_parallel(point_1, point_2, slope):
    b_1=point_slope_formula(point_1[0],point_1[1],slope)
    x_parallel_1,y_parallel_1=get_perpendicular_intersection(point_2, b_1, slope)
    # Compute the cartesian distance between x_parallel and point_2
    distance=np.sqrt((point_2[0]-x_parallel_1)**2+(point_2[1]-y_parallel_1)**2)
    return distance          

def get_perpendicular_intersection(point_2,b_1, slope):

    b_2_perpendicular, slope_perpendicular=get_perpendicular_line_at_point(point_2, slope)
    # We find the interesection point at the first parallel from the perpendicullar at point 2
    x_parallel_1=(b_2_perpendicular-b_1)/(slope-slope_perpendicular)
    y_parallel_1=slope*x_parallel_1+b_1
    return x_parallel_1,y_parallel_1

def get_perpendicular_line_at_point(point, slope):
    slope_perpendicular=-1.0/slope # a perpencicular slope is the negative reciprocal
    b_perpendicular=point_slope_formula(point[0],point[1],slope_perpendicular)
    return b_perpendicular,slope_perpendicular
    
def get_fligh_lines_coordinates(polygon,flight_azimuth, n_runs, separation, buffer_dist=0, start_acquisition=0):
    
    #cartesian_angle=np.radians(-(flight_azimuth)+90)# Convert azimuthal angle to cartesian, i.e. 90 azimumth is 0
    #slope=np.tan(cartesian_angle)
    slope=np.cos(np.radians(flight_azimuth))/np.sin(np.radians(flight_azimuth))
    normal_slope=-1/slope
    normal_angle=np.arctan(normal_slope)
    start_points=[]
    end_points=[]
    point_1,point_2=get_extreme_points(polygon,flight_azimuth)
    if abs(np.cos(np.radians(flight_azimuth)))==1:
        start=np.asarray([point_1[0] + separation/2. - buffer_dist, point_1[1] - start_acquisition])
        end=np.asarray([point_2[0] + separation/2. + buffer_dist, point_2[3] + start_acquisition])
        start_points.append(start)
        end_points.append(end)
        for run in range(n_runs-1):
            start=start+separation*np.asarray([1,0])    
            end=end+separation*np.asarray([1,0])
            start_points.append(start)
            end_points.append(end)        

    else:
        if slope >= 0:
            ref=get_point_in_distance_line([point_1[0] ,point_1[1]],normal_slope, separation/2. - buffer_dist)  
            point_down,point_up=get_extreme_points(polygon,flight_azimuth-90.)
        else:
            ref=get_point_in_distance_line([point_1[0],point_1[1]],normal_slope, separation/2. - buffer_dist)  
            point_down,point_up=get_extreme_points(polygon,flight_azimuth+90.)

        b_ref=point_slope_formula(ref[0],ref[1],slope)
        start0=get_perpendicular_intersection(point_down, b_ref,slope)
        end0=get_perpendicular_intersection(point_up, b_ref, slope)
        start=get_point_in_distance_refpoint(start0,ref, start_acquisition)
        end=get_point_in_distance_refpoint(end0,ref, start_acquisition)
        start_points.append(start)
        end_points.append(end)
        for run in range(n_runs-1):
            start=get_point_in_distance_line(start,normal_slope,separation)         
            end=get_point_in_distance_line(end,normal_slope,separation)             
            start_points.append(start)
            end_points.append(end)

    return start_points, end_points
        

def get_point_in_distance_refpoint(point_ini,ref_point, distance):
    point_ini=np.asarray(point_ini)    
    ref_point=np.asarray(ref_point)
    v=point_ini-ref_point
    u=v/np.linalg.norm(v)
    point=point_ini+distance*u
    return point

def get_point_in_distance_line(point_ini,slope,distance):
    b=point_slope_formula(point_ini[0],point_ini[1],slope)
    x=point_ini[0]+distance/np.sqrt(1+slope**2)
    y=slope*x+b
    return x,y

def get_extreme_points(polygon,flight_azimuth):
    flight_azimuth=np.radians(flight_azimuth)
    rayDirection=np.asarray([np.sin(flight_azimuth),np.cos(flight_azimuth)])
    points=[]
    
    for i,point in enumerate(polygon):
        intersect_bool=False
        for j in range(1,len(polygon)-1):
            index = i + j
            if index > len(polygon)-1:
                index = index - len(polygon)
            index2 = i + j + 1
            if index2 > len(polygon)-1:
                    index2 = index2 - len(polygon)
            
            intersect = lineRayIntersectionPoint(point, rayDirection, polygon[index], polygon[index2])
            if len(intersect) > 0:
                intersect_bool=True
                
            intersect = lineRayIntersectionPoint(point, -rayDirection, polygon[index], polygon[index2])
            if len(intersect) > 0:
                intersect_bool=True 
        if not intersect_bool:
            points.append(point)
    if points[0][0]<= points[1][0]:
        point_1,point_2=points
    else:
        point_2,point_1=points
    return point_1,point_2

def lineRayIntersectionPoint(rayOrigin, rayDirection, point1, point2):
    """
    >>> # Line segment
    >>> z1 = (0,0)
    >>> z2 = (10, 10)
    >>>
    >>> # Test ray 1 -- intersecting ray
    >>> r = (0, 5)
    >>> d = norm((1,0))
    >>> len(lineRayIntersectionPoint(r,d,z1,z2)) == 1
    True
    >>> # Test ray 2 -- intersecting ray
    >>> r = (5, 0)
    >>> d = norm((0,1))
    >>> len(lineRayIntersectionPoint(r,d,z1,z2)) == 1
    True
    >>> # Test ray 3 -- intersecting perpendicular ray
    >>> r0 = (0,10)
    >>> r1 = (10,0)
    >>> d = norm(np.array(r1)-np.array(r0))
    >>> len(lineRayIntersectionPoint(r0,d,z1,z2)) == 1
    True
    >>> # Test ray 4 -- intersecting perpendicular ray
    >>> r0 = (0, 10)
    >>> r1 = (10, 0)
    >>> d = norm(np.array(r0)-np.array(r1))
    >>> len(lineRayIntersectionPoint(r1,d,z1,z2)) == 1
    True
    >>> # Test ray 5 -- non intersecting anti-parallel ray
    >>> r = (-2, 0)
    >>> d = norm(np.array(z1)-np.array(z2))
    >>> len(lineRayIntersectionPoint(r,d,z1,z2)) == 0
    True
    >>> # Test ray 6 --intersecting perpendicular ray
    >>> r = (-2, 0)
    >>> d = norm(np.array(z1)-np.array(z2))
    >>> len(lineRayIntersectionPoint(r,d,z1,z2)) == 0
    True
    """
    # Convert to numpy arrays
    rayOrigin = np.asarray(rayOrigin)
    rayDirection=np.asarray(rayDirection)
    #rayDirection =np.linalg.norm(rayDirection)
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)

    # Ray-Line Segment Intersection Test in 2D
    # http://bit.ly/1CoxdrG
    v1 = rayOrigin - point1
    v2 = point2 - point1
    v3 = np.asarray([-rayDirection[1], rayDirection[0]])
    #v3 = rayDirection
    t1 = np.cross(v2, v1) / np.dot(v2, v3)
    t2 = np.dot(v1, v3) / np.dot(v2, v3)
    if t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0:
        return [rayOrigin + t1 * rayDirection]
    return []

def line_intersect(p0, p1, m0=None, m1=None, q0=None, q1=None):
    ''' intersect 2 lines given 2 points and (either associated slopes or one extra point)
    Inputs:
        p0 - first point of first line [x,y]
        p1 - fist point of second line [x,y]
        m0 - slope of first line
        m1 - slope of second line
        q0 - second point of first line [x,y]
        q1 - second point of second line [x,y]
    '''
    if m0 is  None:
        if q0 is None:
            raise ValueError('either m0 or q0 is needed')
        dy = q0[1] - p0[1]
        dx = q0[0] - p0[0]
        lhs0 = [-dy, dx]
        rhs0 = p0[1] * dx - dy * p0[0]
    else:
        lhs0 = [-m0, 1]
        rhs0 = p0[1] - m0 * p0[0]

    if m1 is  None:
        if q1 is None:
            raise ValueError('either m1 or q1 is needed')
        dy = q1[1] - p1[1]
        dx = q1[0] - p1[0]
        lhs1 = [-dy, dx]
        rhs1 = p1[1] * dx - dy * p1[0]
    else:
        lhs1 = [-m1, 1]
        rhs1 = p1[1] - m1 * p1[0]

    a = np.array([lhs0, 
                  lhs1])

    b = np.array([rhs0, 
                  rhs1])
    try:
        px = np.linalg.solve(a, b)
    except:
        px = np.array([np.nan, np.nan])

    return px

def degrees2ddmmss(decdegrees, is_latitude = True):
    if is_latitude == True:
        if np.sign(decdegrees) == -1:
            zone = 'S'
        else:
            zone = 'N'
    
    else:    
        if np.sign(decdegrees) == -1:
            zone = 'W'
        else:
            zone = 'E'
    
    decdegrees = np.abs(decdegrees)
    
    degrees = int(decdegrees)
    decminutes = 60. * (decdegrees - degrees)
    minutes = int(decminutes)
    seconds = 60 * (decminutes - minutes)
    
    return zone, degrees, minutes, seconds

def sensorlink_coordinates(zone, degrees, minutes, seconds, is_latitude = True):
    if is_latitude:
        fill = 2
    else:
        fill = 3
        
    degrees_str = str(int(degrees)).zfill(fill)
    minutes_dec_str = str(int(1000 * seconds/60.))
    minutes_str = str(int(minutes)).zfill(2)
    
    out_string = '%s.%s.%s,%s'%(degrees_str, minutes_str, minutes_dec_str, zone) 
    return out_string
