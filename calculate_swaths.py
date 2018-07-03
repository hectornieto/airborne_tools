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
output_vector=pth.join(workdir,'P128RaimatMACAW_%sm_%sRuns_solar_plane_UTM.shp')
output_gps_file=pth.join(workdir,'P128RaimatMACAW_%sm_%sRuns_solar_plane_WGS.txt')

# Set site location to compute solar angles
lat,lon=41.61763,0.87218 # Mollerusa
lon,lat=0.52636,41.36809 # Maials
lon,lat=0.51173,41.66265 # Raimat
# Set flight date
doy=185 
stdlon=0 # GMT time
ftime=12.
# Set the desired flight direction or type None for using the solar plane
flight_direction = None


# Set the polygon to cover
raimat_polygon=([291687.3,4618506.2],[291090.2,4618506.1],
                [291099.5,4617998.3],[291680.3,4617985.2])

P128_polygon = ( [ 291094.218133997754194, 4618507.910944790579379 ], 
                [ 291683.701694083865732, 4618503.70635449141264 ], 
                [ 291678.656185723666567, 4617984.018993387930095 ], 
                [ 291097.58180623780936, 4617995.791846227832139 ])

polvori_polygon = ([ 292610.3933971747756, 4615985.997668545693159 ], 
                   [ 294154.318742099858355, 4615935.549106797203422 ], 
                   [ 294083.688699191436172, 4614260.440288106910884 ], 
                   [ 292586.854469431738835, 4614378.162484897300601 ])

flight_polygon = P128_polygon

#  Set the coordinates EPSG code
input_coords = 32631

# Set camera properties
focal_length=790 # FLIR fx from Photoscan
camera_size=np.asarray([640,480])
c_xy=camera_size/2.

focal_length=2030.42 # MACAW fx from Photoscan
camera_size=np.asarray([1280,1024])
c_xy=camera_size/2.

# Fligth altitude
altitude = 180
buffer_dist = 0
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
start_points, end_points=fp.get_fligh_lines_coordinates(flight_polygon,flight_azimuth, n_runs, separation, buffer_dist=buffer_dist, start_acquisition=start_acquisition)

for line, (start, end) in enumerate(zip(start_points,end_points)):
    # transform point
    X_0,Y_0,Z_0=coordTransform.TransformPoint(start[0],start[1],0)
    X_1,Y_1,Z_1=coordTransform.TransformPoint(end[0],end[1],0)
    fid.write('%s\t%s\t%s\t%s\t%s\n'%(line,Y_0,X_0,Y_1,X_1))
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
