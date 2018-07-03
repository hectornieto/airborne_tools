# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:40:02 2016

@author: hector
"""
import cv2
import os.path as pth
import UAV_tools as uav
import os

# Define folders
basedir=os.getcwd()
#basedir='/media/hector/TOSHIBA EXT/'
workdir=basedir+'/Projects/UAV_ET/Database/Gallo/gallo_lodi_2014_2015_data_delivery-selected/'
outdir=workdir
date='20140809'
dist_threshold=1.5 # Matching points between images must be closer than [dist_threshold] to be considered a reliable control poing
pixel_threshold=10 # Ground Control Points must be separated at least [pixel_threshold] pixels
ratio=0.75 # Set threshold similarity ratio
prj='PROJCS["WGS 84 / UTM zone 10N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-123],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],AUTHORITY["EPSG","32610"],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'

tir_image=workdir+'/'+date+'/'+date+'_gallo_lodi_pm_450_tir.tif'
fid=gdal.Open(tir_image,gdal.GA_ReadOnly)
tir_geo=fid.GetGeoTransform()
tir=fid.GetRasterBand(1).ReadAsArray().astype(np.float32)
#tir[tir==0]=np.nan
del fid

hyper_image=workdir+'/'+date+'/'+date+'_gallo_lodi_pm_450_rgbn.tif'
fid=gdal.Open(hyper_image,gdal.GA_ReadOnly)
nir=fid.GetRasterBand(4).ReadAsArray()
nir[nir>1]=1
red=fid.GetRasterBand(1).ReadAsArray()
blue=fid.GetRasterBand(3).ReadAsArray()
ndvi_geo=fid.GetGeoTransform()
del fid

# Scale images
tir_scaled=np.zeros(tir.shape)
tir_scaled[tir>0]= (255*(tir[tir>0]-np.amin(tir[tir>0]))/(np.amax(tir[tir>0])-np.amin(tir[tir>0]))).astype(np.uint8)
#tir_scaled=tir_scaled.astype(np.uint8)
tir_scaled=cv2.equalizeHist(tir.astype(np.uint8))

#ndvi_scaled=np.zeros(ndvi.shape)
#print(np.amin(tir[tir>0]),np.amax(tir[tir>0]))
#ndvi_scaled[ndvi>0]= (255*(ndvi[ndvi>0]-np.amin(ndvi[ndvi>0]))/(np.amax(ndvi[ndvi>0])-np.amin(ndvi[ndvi>0]))).astype(np.uint8)
#ndvi_scaled=ndvi_scaled.astype(np.uint8)
nir_scaled=cv2.equalizeHist((nir*255).astype(np.uint8))
red_scaled=cv2.equalizeHist((red*255).astype(np.uint8))
blue_scaled=cv2.equalizeHist((blue*255).astype(np.uint8))
img=cv2.merge([nir_scaled,red_scaled,blue_scaled])
ndvi_scaled=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#==============================================================================
# # Save images for error checking
# test=osr.SpatialReference()
# test.ImportFromEPSG(32610)
# prj=test.ExportToPrettyWkt()
# driver = gdal.GetDriverByName('GTiff')
# 
# outfile=outdir+'/'+date+'_TIR_8bit.tif'
# ds = driver.Create(outfile, tir_scaled.shape[1], tir_scaled.shape[0],1, gdal.GDT_Byte)
# ds.SetGeoTransform([extent[0],tir_geo[1],0,extent[3],0,tir_geo[5]])
# ds.SetProjection(prj)
# ds.GetRasterBand(1).WriteArray(tir_scaled)
# ds.FlushCache()
# del ds
# 
# outfile=outdir+'/'+date+'_NDVI_8bit.tif'
# ds = driver.Create(outfile, ndvi_scaled.shape[1], ndvi_scaled.shape[0],1, gdal.GDT_Byte)
# ds.SetGeoTransform([extent[0],ndvi_geo[1],0,extent[3],0,ndvi_geo[5]])
# ds.SetProjection(prj)
# ds.GetRasterBand(1).WriteArray(ndvi_scaled)
# ds.FlushCache()
# del ds
#==============================================================================

#==============================================================================
# # testing images
# tir_scaled = cv2.imread('box.png',0)          # queryImage
# ndvi_scaled = cv2.imread('box_in_scene.png',0) # trainImage
#==============================================================================