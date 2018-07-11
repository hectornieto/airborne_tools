# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:36:53 2017

@author: hnieto
"""
import numpy as np
from pyTSEB import meteo_utils as met
import os
import os.path as pth
import osr
import airborne_tools.save_runs as fp

#==============================================================================
# # INPUT PARAMETERS
#==============================================================================
workdir=os.getcwd()
output_vector=pth.join(workdir,'AlcarrasFLIR_%sm_%sRuns_solar_plane_UTM.shp')
output_gps_file=pth.join(workdir,'AlcarrasFLIR_%sm_%sRuns_solar_plane_WGS.txt')
output_sensorlink_file=pth.join(workdir,'sensorlink_AlcarrasFLIR_%sm_%sRuns_solar_plane_WGS.txt')

# Set site location to compute solar angles
lat, lon = 41.61763 ,0.87218 # Mollerusa
lon, lat=0.52636,41.36809 # Maials
lon, lat=0.51173,41.66265 # Raimat
# Set flight date
doy=193 
stdlon=0 # GMT time
ftime=12.
# Set the desired flight direction or type None for using the solar plane
flight_direction = None
#to change the flight direction, instead of using None, we should write the angle (i.e. 30)


# Set the polygon to cover
raimat_polygon=([291687.3,4618506.2],[291090.2,4618506.1],
                [291099.5,4617998.3],[291680.3,4617985.2])

Alcarras_polygon = ( [ 283171.300875827437267, 4609057.553501397371292 ], 
                  [ 283202.964008267736062, 4609110.880882349796593 ], 
                  [ 283256.291389219695702, 4609104.214959730394185 ], 
                  [ 283297.953405588399619, 4609070.052106307819486 ], 
                  [ 283392.109562581521459, 4608718.424688147380948 ], 
                  [ 283289.621002314321231, 4608690.927757344208658 ])

polvori_polygon = ([ 292610.3933971747756, 4615985.997668545693159 ], 
                   [ 294154.318742099858355, 4615935.549106797203422 ], 
                   [ 294083.688699191436172, 4614260.440288106910884 ], 
                   [ 292586.854469431738835, 4614378.162484897300601 ])

flight_polygon = Alcarras_polygon

#  Set the coordinates EPSG code
input_coords = 32631

# Set camera properties
# FLIR fx from Photoscan
focal_length=790 
camera_size=np.asarray([640,480])
c_xy=camera_size/2.

#if we want to do the flight plan, based on the MACAW, we should remove the comment (#)
#==============================================================================
# MACAW fx from Photoscan
# focal_length=2030.42 
# camera_size=np.asarray([1280,1024])
# c_xy=camera_size/2.
#==============================================================================

# Fligth altitude
altitude = 180
buffer_dist = 0 # Keep at zero for now. Need to be revised
start_acquisition = 200
overlap = .60

#==============================================================================
# # END INPUT PARAMETERS
#==============================================================================
if flight_direction=='E':
    flight_azimuth=90.
elif flight_direction=='N':
    flight_azimuth=180.
elif type(flight_direction)==type(None):
    _, flight_azimuth = met.calc_sun_angles(lat, lon, stdlon, doy, ftime)
else:
    flight_azimuth=flight_direction
    
    
prj = osr.SpatialReference()
prj.ImportFromEPSG(input_coords)
prj_wkt=prj.ExportToPrettyWkt()

out_prj = osr.SpatialReference()
out_prj.ImportFromEPSG(4326)
coordTransform = osr.CoordinateTransformation(prj, out_prj)

point_1,point_2=fp.get_extreme_points(flight_polygon, flight_azimuth)
  
total_swath=fp.get_total_swath(flight_azimuth, point_1, point_2)+2*buffer_dist
n_runs,separation,image_size,resolution=fp.get_flight_lines_paramters(altitude,total_swath,camera_size,focal_length,overlap=overlap)
print('Flight altitude: %s m'%altitude)
print('Image Size: %sm x %sm'%(image_size))
print('Total swath: %s m'%total_swath)
print('resolution (m):',resolution)
print('Separation between runs with %s %% overlap (m): %s'%(100*overlap,separation))
print('Number of runs with %s %% overlap: %s'%(100*overlap,n_runs))
#final_separation=image_size[0]/(n_runs+1.0)

fid=open(output_gps_file%(altitude,n_runs),'w')
fid.write('Flight altitude: %s m a.g.l. \n'%altitude)
fid.write('Image Size: %sm x %sm \n'%(image_size))
fid.write('Flight Swath: %sm \n'%(total_swath))
fid.write('Average resolution (m): %s x %s \n'%(resolution))
fid.write('Separation between runs with %s %% overlap (m): %s \n'%(100*overlap,separation))
fid.write('Number of runs with %s %% overlap: %s \n\n'%(100*overlap,n_runs))
fid.write('line \t start_latitude \t start_longitude \t end_latitude \t end_longitude \n')

if n_runs == 1:
    separation = total_swath - 2*buffer_dist
    buffer_dist = 0.
    
start_points, end_points = fp.get_fligh_lines_coordinates(flight_polygon,
                                                          flight_azimuth,
                                                          n_runs, separation,
                                                          buffer_dist=buffer_dist,
                                                          start_acquisition=start_acquisition)

for line, (start, end) in enumerate(zip(start_points,end_points)):
    # transform point
    X_0,Y_0,Z_0=coordTransform.TransformPoint(start[0],start[1],0)
    X_1,Y_1,Z_1=coordTransform.TransformPoint(end[0],end[1],0)
    Y_0, Y_1 = map(fp.degrees2ddmmss, [Y_0, Y_1] , [True, True])
    X_0, X_1 = map(fp.degrees2ddmmss, [X_0, X_1] , [False, False])
    fid.write('%s\t%s%sº%s\t%s%sº%s\t%s%sº%s\t%s%sº%s\n'%(line,
                                        Y_0[0], Y_0[1], Y_0[2]+Y_0[3]/60., 
                                       X_0[0], X_0[1], X_0[2]+X_0[3]/60.,
                                       Y_1[0], Y_1[1], Y_1[2]+Y_1[3]/60., 
                                       X_1[0], X_1[1], X_1[2]+X_1[3]/60.))
    
fid.flush()
fid.close()
outshapefile=pth.join(output_vector%(altitude,n_runs))
fp.write_flight_runs_vector(start_points,end_points,outshapefile,prj_wkt)

#==============================================================================
# # FLIR tau    
# FOV=(32,26)
# separation=(3.63,1.2)
# altitudes=np.asarray([10,15,20])
# print(2*altitudes[2]*np.tan(np.radians(np.asarray(FOV)/2)))
# print(2*altitudes[2]*np.tan(np.radians(np.asarray(FOV)/2))/np.asarray(separation))
#==============================================================================


fid=open(output_sensorlink_file%(altitude,n_runs),'w')
reverse = 0
for line, (start, end) in enumerate(zip(start_points,end_points)):
    # transform point
    X_0,Y_0,Z_0=coordTransform.TransformPoint(start[0],start[1],0)
    X_1,Y_1,Z_1=coordTransform.TransformPoint(end[0],end[1],0)
    lat_str_0 = fp.sensorlink_coordinates(*fp.degrees2ddmmss(Y_0, is_latitude = True), is_latitude = True) 
    lon_str_0 = fp.sensorlink_coordinates(*fp.degrees2ddmmss(X_0, is_latitude = False), is_latitude = False)  
    lat_str_1 = fp.sensorlink_coordinates(*fp.degrees2ddmmss(Y_1, is_latitude = False), is_latitude = True) 
    lon_str_1 = fp.sensorlink_coordinates(*fp.degrees2ddmmss(X_1, is_latitude = True), is_latitude = False)  

    if reverse:
        fid.write('%s,%s,T, %s agl\n'%(lat_str_1, lon_str_1, altitude))
        fid.write('%s,%s,T, %s agl\n'%(lat_str_0, lon_str_0, altitude))
        reverse = 0
    else:
        fid.write('%s,%s,T, %s agl\n'%(lat_str_0, lon_str_0, altitude))
        fid.write('%s,%s,T, %s agl\n'%(lat_str_1, lon_str_1, altitude))
        reverse = 1

fid.flush()
fid.close()