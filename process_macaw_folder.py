# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:19:18 2017

@author: hnieto
"""

import os.path as pth
from glob import glob
import os
import numpy as np
from airborne_tools import macaw

#==============================================================================
# # INPUT PARAMETERS
#==============================================================================
sync_imu = True # set True to synchronize GPS_IMU data with the images
hour_lag = 0 # set a non zero value if you suspect that the images are not well synchronized

# Set working folders
workdir = pth.join('H:\\\\', '2018', 'Mollerussa', 'Vuelo20180710', 'images')
# Location of the GPS IMU data
imu_file = pth.join(workdir, 'GPS', 'MOLLE10072018_IG-500N_007000277.txt')
# Folder where the raw macaw images are stored
macaw_dir = pth.join(workdir, 'MACAW', 'L0')
#==============================================================================
# # END INPUT PARAMETERS
#==============================================================================

image_list = glob(pth.join(macaw_dir, '*.TIF'))

if sync_imu:
    print('Reading IMU data')
    imu_data = np.genfromtxt(imu_file, dtype=None, names=True, delimiter='\t')

fid = open(pth.join(workdir, 'imu_log_macaw.txt'), 'w')
header = 'camera\tlat\tlon\talt\tyaw\tpitch\troll\n'
fid.write(header)

for image in image_list:
    image_name = pth.basename(image)
    print('Synchronizing image %s'%image_name)
    header_dict = macaw.read_macaw_header(image)
    year_str = str(header_dict['year']).zfill(4)
    month_str = str(header_dict['month']).zfill(2)
    day_str = str(header_dict['day']).zfill(2)
    hour_str = str(header_dict['hour']-hour_lag).zfill(2)
    minute_str = str(header_dict['minute']).zfill(2)
    second_str = str(header_dict['second']).zfill(2)
	
    if sync_imu:
	    imu = macaw.sync_macaw_IMU(header_dict, imu_data, hour_lag=hour_lag)#lat, lon, alt, yaw, pitch, roll
	    out_string = image_name
	    for item in imu:
			out_string += '\t%s'%item
	    fid.write(out_string + '\n')    
    
fid.flush()
fid.close()
