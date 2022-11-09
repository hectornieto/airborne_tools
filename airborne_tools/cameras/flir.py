'''
; name:
;   read_fpf.pro
;
; purpose:
;   reads FLIR Public Format files from FLIR thermal cameras
;
; calling sequence:
;   fpf = read_fpf(file)
;   
; keywords
;   header_only : if set only header of fpf file is read, but not data.
;   subset:       array [4]. If set, only part of the image is read into 
;                 memory, where [x0,y0,x1,y1] is the subset to be retained.
;
; output:
;   fpf         : structure with all tags and data from .fpf file.
;
; example:
;   file = 'Example.FPF'
;   fpf = read_fpf(file)
;   tvscl, rotate(fpf.data,7)
;   
; revision history:
;   14-May-09  : andreas.christen@ubc.ca
'''
import struct
import numpy as np
from osgeo import gdal

TOTAL_HEADER=892 
IMAGE_HEADER=120
CAMERA_HEADER=360
OBJECT_HEADER=104
DATE_HEADER=92
SCALING_HEADER=88
FLIR_TYPE_DICT={0:'I',2:'f',3:'f',4:'H'}
#----------------------------
# set-up header of fpf file
#----------------------------
fpf_image_headers=(('fpf_id',32,'32c'),
				('version', 4,'l'),
				('pixel_offset',4,'l') ,
				('image_type',2,'h'),
				('pixel_format',2,'h'),
				('x_size',2,'h'),
				('y_size',2,'h'),
				('trig_count',4,'l' ),
				('frame_count',4,'l' ),
				('spare_long',4*16,'16l'))
				
fpf_camera_headers= (('camera_name',32,'32s'),
				('camera_partn', 32,'32s'),
				('camera_sn', 32,'32s'),
				('camera_range_t_min', 4,'f'),
				('camera_range_t_max', 4,'f'),
				('camera_lens_name', 32,'32s'), 
				('camera_lens_partn', 32,'32s'), 
				('camera_lens_sn', 32,'32s'), 
				('filter_name', 32,'32s'), 
				('filter_part_n', 32,'32s'), 
				('filter_part_sn', 32,'32s'), 
				('spare_long', 4*16,'16l'))

fpf_object_headers= (('emissivity',4 ,'f'),
				('object_distance', 4 ,'f'),
				('amb_temp',4 ,'f'),       
				('atm_temp',4 ,'f'), 
				('rel_hum', 4 ,'f'),       
				('compu_tao',4 ,'f'),  
				('estim_tao',4 ,'f'),  
				('ref_temp', 4 ,'f'),
				('ext_opt_temp',4 ,'f'), 
				('ext_opt_trans',4 ,'f'), 
				('spare_long', 4*16,'16l'))
				
fpf_date_time_headers  = (('year',4,'l'),
				('month', 4,'l'),
				('day', 4,'l'),
				('hour', 4,'l'),
				('minute', 4,'l'),
				('second', 4,'l'),
				('millisecond', 4,'l'),
				('spare_long', 4*16,'16l'))

fpf_scaling_headers = (('t_min_cam',4 ,'f'),  
				('t_max_cam',4 ,'f'),  
				('t_min_calc',4 ,'f'),       
				('t_max_calc',4 ,'f'),  
				('t_min_scale', 4 ,'f'),       
				('t_max_scale',4 ,'f'),   
				('spare_long',4*16 ,'16l')) 

def get_total_bytes(header_lenght):
    total=0
    for byte in header_lenght:
        total+=byte
    return total

def read_fpf(file, header_only=False):
    


    headers=(fpf_image_headers,fpf_camera_headers,fpf_object_headers,
             fpf_date_time_headers,fpf_scaling_headers)                        
    
    #read file
    with open(file, 'rb') as fid:
        fileContent =fid.read()
    header_binary=fileContent[:892]
    image_binary=fileContent[892:]
    # Read header
    byte=0
    header_dict=dict()
    for header in headers:
        for name,lenght,fmt in header:
            header_dict[name]=struct.unpack(fmt,header_binary[byte:byte+lenght])[0]
            byte+=lenght
    
    # Read image
    # Get the total number of cells
    total_elements=header_dict['x_size']*header_dict['y_size']
    fmt='%s%s'%(total_elements,FLIR_TYPE_DICT[int(header_dict['pixel_format'])])
    image=np.asarray(struct.unpack(fmt,image_binary))
    #min_value=image.min()
    #max_value=image.max()
    #slope=float(header_dict['t_max_scale']-header_dict['t_min_scale'])/float(max_value-min_value)
    image=image.reshape((header_dict['y_size'],header_dict['x_size']))
    #image=header_dict['t_min_scale']+(image-min_value)*slope
                     
    return header_dict,image
	
def Convert_16_to_8_bit(image):

    #Converts the 16 bit image array into a 8 bit for creating an RGB image for Photoscan
    from scipy.stats import scoreatpercentile
    import numpy
    
    #  2% Linear stretching
    im_min=scoreatpercentile(image.flatten(),2)
    im_max=scoreatpercentile(image.flatten(),98)

    shape=numpy.shape(image)

    if im_max==im_min:
        im256=numpy.zeros(shape)
    else:
        im256=255*((image-im_min)/(im_max-im_min))
        im256[image<im_min]=0
        im256[image>im_max]=255

    return im256 
	
def saveImg(data, outPath, scale=100, data_type=gdal.GDT_UInt16):
    
    # Start the gdal driver for GeoTIFF
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(outPath, data.shape[1], data.shape[0], 1, data_type)
    ds.GetRasterBand(1).WriteArray(data*scale)
    print('Saved ' +outPath )
    ds = None

    return True

def sync_flir_IMU(header_dict,imu_data,hour_lag=0):
    
    index=np.logical_and.reduce((imu_data['year']==header_dict['year'],
                                 imu_data['month']==header_dict['month'],
                                 imu_data['day']==header_dict['day'],
                                 imu_data['hour']==header_dict['hour']-hour_lag,
                                 imu_data['min']==header_dict['minute'],
                                 imu_data['sec']==header_dict['second']))
    lats=imu_data['latitude'][index]
    lons=imu_data['longitude'][index]
    alts=imu_data['altitude'][index]
    yaws=imu_data['yaw'][index]
    pitchs=imu_data['pitch'][index]
    rolls=imu_data['roll'][index]
		
    # Find the closest data in time
    nano_seconds=imu_data['nano'][index]*1e-9

    dif_nano=np.abs(nano_seconds-header_dict['millisecond']*1e-3)
    min_diff=np.min(dif_nano)
    index=dif_nano==min_diff
    
    lat=lats[index][0]
    lon=lons[index][0]
    alt=alts[index][0]
    yaw=yaws[index][0]
    pitch=pitchs[index][0]
    roll=rolls[index][0]
    return lat, lon, alt, yaw, pitch, roll
