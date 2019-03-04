# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 10:13:51 2016

@author: hector
"""
from __future__ import print_function
import gdal
import numpy as np
import cv2
from os.path import  basename,dirname,join,exists
import os

def collocate_image(master_file, slave_file, output_file, slave_bands=None, noData=0,
                    dist_threshold=50, pixel_threshold=20, angle_threshold=90, 
                    std_translation_thres=2, filter_intersect=True,
                    search_window_size=1000, manual_GCP_file=None, transform=0,
                    useSIFT=True, match_factor=0.75,image_to_georreference=None, bands_to_georreference=None):
    outdir=dirname(output_file)
    # Read the master TIR mosaic
    masterfid=gdal.Open(master_file,gdal.GA_ReadOnly)
    master_geo=masterfid.GetGeoTransform()
    masterXsize=masterfid.RasterXSize
    masterYsize=masterfid.RasterYSize
    
    # Set the extent for the output image equal as the master image
    xmin,ymax=get_map_coordinates(0,0,master_geo)
    xmax,ymin=get_map_coordinates(masterYsize,masterXsize,master_geo)
    output_extent=(xmin,ymin,xmax,ymax)
    
    slavefid=gdal.Open(slave_file,gdal.GA_ReadOnly)
    slaveCols=slavefid.RasterXSize
    slaveRows=slavefid.RasterYSize
    slave_geo=slavefid.GetGeoTransform()
    slave_prj=slavefid.GetProjection()
    if slave_bands==None:
        slave_bands= range(slavefid.RasterCount)
    # Get the upper and lower bounds for each tile in the input image
    upperRows=range(0,slaveRows,search_window_size)
    upperCols=range(0,slaveCols,search_window_size)
    nXwindows=len(upperCols)
    nYwindows=len(upperRows)
    # Create the empty list with valid GCPs and count for valid GCPs
    GCP_valid=[]
    total=0
    distance=0
    azimuth=0
    intersect=0
    proximity=0
    # Loop all the tiles to get the GCPs
    for i,upperRow in enumerate(upperRows):
        #print('Searching tiles in row %s,'%(upperRow))
        if i>=nYwindows-1:
            lowerRow=slaveRows
            win_ysize=lowerRow-upperRow
            # Last tile must fit the image size
        else:
            lowerRow=upperRows[i+1]
            win_ysize=search_window_size
        for j,upperCol in enumerate(upperCols):
            print('Searching tile row %s, col %s'%(upperRow,upperCol))
            if j>=nXwindows-1:
                lowerCol=slaveCols # Last tile must fit the image size
                win_xsize=lowerCol-upperCol
            else:
                lowerCol=upperCols[j+1]
                win_xsize=search_window_size
            xmin,ymax=get_map_coordinates(upperRow,upperCol,slave_geo)
            xmax,ymin=get_map_coordinates(lowerRow,lowerCol,slave_geo)
            # Get the pixel coordinates of the master image
            xmin-=dist_threshold # THe search window has a buffer equal to the distance threshold
            ymin-=dist_threshold # THe search window has a buffer equal to the distance threshold
            xmax+=dist_threshold # THe search window has a buffer equal to the distance threshold
            ymax+=dist_threshold # THe search window has a buffer equal to the distance threshold
            upperMasterRow,upperMasterCol=np.floor(get_pixel_coordinates(xmin,ymax,master_geo)).astype(np.int16)
            lowerMasterRow,lowerMasterCol=np.ceil(get_pixel_coordinates(xmax,ymin,master_geo)).astype(np.int16)
            upperMasterRow=int(np.clip(upperMasterRow,0,masterYsize))# Avoid negative pixel coordinates
            upperMasterCol=int(np.clip(upperMasterCol,0,masterXsize))# Avoid negative pixel coordinates
            lowerMasterRow=int(np.clip(lowerMasterRow,0,masterYsize))# Avoid pixel coordinates beyond the image extent
            lowerMasterCol=int(np.clip(lowerMasterCol,0,masterXsize))# Avoid pixel coordinates beyond the image extent
            win_MasterXsize=int(lowerMasterCol-upperMasterCol)
            win_MasterYsize=int(lowerMasterRow-upperMasterRow)
            # Read the master image array and subset
            master_scaled=masterfid.GetRasterBand(1).ReadAsArray(
                        upperMasterCol,upperMasterRow, win_MasterXsize, 
                        win_MasterYsize).astype(np.float)
            master_scaled=scale_grayscale_image(master_scaled, noData=noData)
            UL_X,UL_Y=get_map_coordinates(upperMasterRow,upperMasterCol,master_geo)
            # Get the master subset geotransform
            master_window_geo=(UL_X,master_geo[1],master_geo[2],UL_Y,master_geo[4],master_geo[5])
            if master_scaled.any() == noData: # If all pixels have no data skip tile
                continue
            # Loop all the Principal Componets
            GCP_region=[]
            for band in slave_bands:
                slave_scaled=slavefid.GetRasterBand(band+1).ReadAsArray(
                                                    upperCol,upperRow, 
                                                    win_xsize, win_ysize).astype(np.float)
                if slave_scaled.any() == noData: # If all pixels have no data skip tile
                    continue
                slave_scaled=scale_grayscale_image(slave_scaled, noData=noData)
                if np.any(slave_scaled)==0:
                    continue
                # Find features and matches
                GCPs=find_GCPs(master_scaled,slave_scaled,master_window_geo,
                                   UL_offset=(upperRow,upperCol),
                                    match_factor=match_factor,useSIFT=useSIFT)
                total+=len(GCPs)
                if len(GCPs)>0 and dist_threshold>0:
                    GCPs=filter_GCP_by_map_proximity(GCPs,slave_geo,dist_threshold)
                    print('Found %s valid GPCs'%np.asarray(GCPs)[:,0].shape)
                    distance+=float(np.asarray(GCPs)[:,0].size)
                for GCP in np.asarray(GCPs).tolist():
                    GCP_region.append(GCP)
            if len(GCP_region) > 0:
                if angle_threshold > 0:
                    # Filter GCPs based on angular deviations from the mean translation direction
                    GCP_region=filter_GCP_by_azimuth(GCP_region,slave_geo,angle_threshold)
                    print('Filtered by angle %s GPCs'%GCP_region[:,0].shape)
                    azimuth+=float(GCP_region[:,0].size)
                if filter_intersect:
                    #Filter GCPs based on whether they intersec with other GCPs   
                    GCP_region=filter_GCP_by_intersection(GCP_region,slave_geo)
                    print('Found %s GCPs that do not intersect' %(len(GCP_region)))
                    intersect+=len(GCP_region)
                if pixel_threshold > 0:
                    # Filter GCPs based on image proximity
                    GCP_region=filter_GCP_by_GCP_proximity(GCP_region,pixel_threshold)
                    print('Found %s GCPs separated enough' %(len(GCP_region)))
                    proximity+=len(GCP_region)
                for GCP in np.asarray(GCP_region).tolist():
                    GCP_valid.append(tuple(GCP))

    nGCPs={'total':total,'distance':distance,'azimuth':azimuth,'intersect':intersect,'proximity':proximity}   
    if std_translation_thres > 0:                
        # Filter GCPs by removing outliers in the tranlation distance between source and destination
        GCP_valid=filter_GCP_by_tranlation(GCP_valid,slave_geo,stdThres=std_translation_thres)
        print('Filtered %s GCPs with normal tranlation' %(len(GCP_valid))) 
        nGCPs['tranlation']=len(GCP_valid)
   
    # Remove GCPs with exactly the same map coordinates
    GCP_valid=filter_GCP_by_unique_coordinates(GCP_valid)
    print('Filtered %s GCPs with same coordinates' %(len(GCP_valid))) 
    nGCPs['coordinates']=len(GCP_valid)

    print(nGCPs)
    
    # Add manual GCPs
    if manual_GCP_file:
        GCP_valid=np.asarray(GCP_valid)[:,:4]
        GCP_valid=ASCII_to_GCP(manual_GCP_file,GCP=GCP_valid.tolist())
    slaveXCoord,slaveYCoord=get_map_coordinates(np.asarray(GCP_valid)[:,2],np.asarray(GCP_valid)[:,3],slave_geo)
    
    if not exists(outdir+'/GCPs/'):
        os.mkdir(outdir+'/GCPs/')
    outshapefile=outdir+'/GCPs/'+basename(output_file)[:-4]+'_Transform.shp'
    write_transformation_vector((slaveXCoord,slaveYCoord),(np.asarray(GCP_valid)[:,0],np.asarray(GCP_valid)[:,1]),outshapefile,slave_prj)
    
    # Write the GCP to ascii file
    outtxtfile=outdir+'/GCPs/'+basename(output_file)[:-4]+'_GCPs.txt'    
    GCPs_to_ASCII(GCP_valid,outtxtfile)
    # Reproject image
    if image_to_georreference:
        slave_file=image_to_georreference
    warp_image_with_GCPs(slave_file,GCP_valid,output_extent,transform=transform,
                             subsetBands=bands_to_georreference,outfile=output_file)

def georref_image(master_file, slave_file, output_file, slave_bands=None, noData=0,
                    pixel_threshold=20, 
                    manual_GCP_file=None, transform=0,
                    useSIFT=True, match_factor=0.75,image_to_georreference=None, bands_to_georreference=None,
                    resolution=None):
    outdir=dirname(output_file)
    # Read the master TIR mosaic
    masterfid=gdal.Open(master_file,gdal.GA_ReadOnly)
    master_geo=masterfid.GetGeoTransform()
    masterXsize=masterfid.RasterXSize
    masterYsize=masterfid.RasterYSize
    
    # Set the extent for the output image equal as the master image
    xmin,ymax=get_map_coordinates(0,0,master_geo)
    xmax,ymin=get_map_coordinates(masterYsize,masterXsize,master_geo)
    output_extent=(xmin,ymin,xmax,ymax)
    if not resolution:
       resolution= master_geo[1],master_geo[5]
    
    slavefid=gdal.Open(slave_file,gdal.GA_ReadOnly)
    if slave_bands==None:
        slave_bands= range(slavefid.RasterCount)
    # Create the empty list with valid GCPs and count for valid GCPs
    GCP_valid=[]
    total=0
    distance=0
    azimuth=0
    intersect=0
    proximity=0
    master_scaled=masterfid.GetRasterBand(1).ReadAsArray().astype(np.float)
    master_scaled=scale_grayscale_image(master_scaled, noData=noData)
    GCP_region=[]
    for band in slave_bands:
        slave_scaled=slavefid.GetRasterBand(band+1).ReadAsArray().astype(np.float)
        if slave_scaled.any() == noData: # If all pixels have no data skip tile
            continue
        slave_scaled=scale_grayscale_image(slave_scaled, noData=noData)
        if np.any(slave_scaled)==0:
            continue
        # Find features and matches
        GCPs=find_GCPs(master_scaled,slave_scaled,master_geo,
                            match_factor=match_factor,useSIFT=useSIFT)
        total+=len(GCPs)
        for GCP in np.asarray(GCPs).tolist():
            GCP_region.append(GCP)
    if len(GCP_region) > 0:
        if pixel_threshold > 0:
            # Filter GCPs based on image proximity
            GCP_region=filter_GCP_by_GCP_proximity(GCP_region,pixel_threshold)
            print('Found %s GCPs separated enough' %(len(GCP_region)))
            proximity+=len(GCP_region)
        for GCP in np.asarray(GCP_region).tolist():
            GCP_valid.append(tuple(GCP))

    nGCPs={'total':total,'distance':distance,'azimuth':azimuth,'intersect':intersect,'proximity':proximity}   
   
    # Remove GCPs with exactly the same map coordinates
    GCP_valid=filter_GCP_by_unique_coordinates(GCP_valid)
    print('Filtered %s GCPs with same coordinates' %(len(GCP_valid))) 
    nGCPs['coordinates']=len(GCP_valid)

    print(nGCPs)
    
    # Add manual GCPs
    if manual_GCP_file:
        GCP_valid=np.asarray(GCP_valid)[:,:4]
        GCP_valid=ASCII_to_GCP(manual_GCP_file,GCP=GCP_valid.tolist())
    
    if not exists(outdir+'/GCPs/'):
        os.mkdir(outdir+'/GCPs/')
    # Write the GCP to ascii file
    outtxtfile=outdir+'/GCPs/'+basename(output_file)[:-4]+'_GCPs.txt'    
    GCPs_to_ASCII(GCP_valid,outtxtfile)
    # Reproject image
    if image_to_georreference:
        slave_file=image_to_georreference
    warp_image_with_GCPs(slave_file,GCP_valid,output_extent,transform=transform,
                             subsetBands=bands_to_georreference,outfile=output_file,
                             resolution=resolution)

def scale_grayscale_image(image, noData=None):
    
    if noData==None:
        index=np.ones(image.shape,dtype=np.bool)
    else:
        index=image!=noData
    if np.sum(index)>30:
        image[index]= 255*((image[index]-np.amin(image[index]))/(np.amax(image[index])-np.amin(image[index])))
        image=image.astype(np.uint8)
        image=cv2.equalizeHist(image.astype(np.uint8))
    else:
        image*=0
    return image

def get_map_coordinates(row,col,geoTransform):
    X=geoTransform[0]+geoTransform[1]*col+geoTransform[2]*row
    Y=geoTransform[3]+geoTransform[4]*col+geoTransform[5]*row
    return X,Y

def get_pixel_coordinates(X,Y,geoTransform):
    row=(Y - geoTransform[3]) / geoTransform[5]
    col=(X - geoTransform[0]) / geoTransform[1]
    return row,col

def compute_PCA(imageFile,pcaComponents=None,outfile=None,bandsPCA=None,normalize=False):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    #from sknn.platform import gpu32
    
    driver = gdal.GetDriverByName('GTiff')
    
    fid=gdal.Open(imageFile,gdal.GA_ReadOnly)
    if bandsPCA:    
        bands=list(bandsPCA)
        nbands=len(bands)
    else:
        nbands=fid.RasterCount
        bands=range(nbands)
    if not pcaComponents:
        pcaComponents=nbands
    cols=fid.RasterXSize
    rows=fid.RasterYSize
    hyper_geo=fid.GetGeoTransform()
    hyper_prj=fid.GetProjection()
    input_array=np.zeros((rows*cols,nbands))
    for i,band in enumerate(bands):
        print('Reading hyperspectral band %s'%band)
        array=fid.GetRasterBand(band+1).ReadAsArray()
        mask=array>0
        if normalize:
            scalerInput=StandardScaler()
            scalerInput.fit(array[array>0].reshape(-1,1))
            input_array[:,i]=scalerInput.transform(array.reshape(-1,1)).reshape(-1)
            input_array[:,i]*=mask.reshape(-1)
        else:
            input_array[:,i]=array.reshape(-1)
        del array
    del fid
    pca = PCA(n_components=pcaComponents)
    input_array=np.ma.masked_array(input_array,mask=input_array==0)
    pca.fit(input_array)
    print('Explained variance per component,'+str(pca.explained_variance_ratio_))
    print('Explained variance total,'+str(np.sum(np.asarray(pca.explained_variance_ratio_))))
    
    output_array=pca.transform(input_array)*mask.reshape(-1,1)
    output_array=output_array.reshape((rows,cols,pcaComponents))
    if outfile:
        ds = driver.Create(outfile, cols, rows ,pcaComponents, gdal.GDT_Float32)
        ds.SetGeoTransform(hyper_geo)
        ds.SetProjection(hyper_prj)
        for band in range(pcaComponents):
            ds.GetRasterBand(band+1).WriteArray(output_array[:,:,band])
            ds.FlushCache()
        del ds
    
    return output_array,pca.explained_variance_ratio_

def warp_image_with_GCPs(input_file,GCP_list,output_extent=None,subsetBands=None,transform=0, outfile=None, data_format=2, resolution=None):
    from os.path import exists,dirname, splitext
    from glob import glob
    from os import remove
    import subprocess
    
    GDALGCPs=create_gdal_GCPs(GCP_list)  
    
    infid=gdal.Open(input_file,gdal.GA_ReadOnly)
    prj=infid.GetProjection()
    if not resolution:
        geo=infid.GetGeoTransform()
        xres,yres=geo[1],geo[5]
    else:
        xres,yres=resolution
    
    rows=infid.RasterYSize
    cols=infid.RasterXSize
    driver = gdal.GetDriverByName('GTiff')
    tempfile=dirname(input_file)+'/temp.tif'
    if exists(tempfile): 
        [remove(i) for i in glob(splitext(tempfile)[0]+'.*')]
    if not output_extent:
        xmin_out,ymax_out=get_map_coordinates(0,0,geo)
        xmax_out,ymin_out=get_map_coordinates(rows,cols,geo)
        output_extent=(xmin_out,ymin_out,xmax_out,ymax_out)

    nbands=infid.RasterCount
    if subsetBands==None:
        subsetBands=range(1,nbands+1)
    nbands=len(subsetBands)
    ds = driver.Create(tempfile, cols, rows, nbands, data_format)
    for i,band in enumerate(subsetBands):
        print('Saving Band ' +str(band))
        array=infid.GetRasterBand(band).ReadAsArray()
        band=ds.GetRasterBand(i+1)
        band.WriteArray(array)
        band.SetNoDataValue(0)
        #band.FlushCache()
        del band
        del array
    del infid
    ds.SetGCPs(GDALGCPs,prj)
    ds.SetProjection(prj)
    ds.FlushCache()
    del ds
    # Run GDAL Warp
    if not outfile:
        outfile=input_file[:-4]+'_Georref.bsq'
    if exists(outfile):
        [remove(i) for i in glob(splitext(outfile)[0]+'.*')]
    if transform==0:
        gdal_command='gdalwarp -tps -overwrite -r bilinear -of ENVI -srcnodata 0 -dstnodata 0 -multi -tr %s %s '%(xres,yres) +' -te %s %s %s %s '%output_extent+' "'+ tempfile + '" "'  + outfile +'"'
    else:
        gdal_command='gdalwarp -order ' +str(transform)+' -overwrite -r bilinear -of ENVI -srcnodata 0 -dstnodata 0 -multi -tr %s %s '%(xres,yres) +' -te %s %s %s %s '%output_extent+' "'+ tempfile + '" "'  + outfile +'"'
    print(gdal_command)
    proc=subprocess.Popen(gdal_command,shell=True,stdout=subprocess.PIPE,
         stdin=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
    while True:
        out = proc.stdout.read(1)
        if out == '' and proc.poll() != None:
        	break
        if out != '':
            print(out,end='')
    proc.communicate()
    remove(tempfile)
    return True

def find_GCPs(masterImage,slaveImage,masterGeo,UL_offset=(0,0),match_factor=0.75,useSIFT=False):

    GCP_list=[]
    # Create the feature detector/descriptor and matching objects
    if useSIFT==True:
        # Initiate SIFT detector
        detector = cv2.xfeatures2d.SIFT_create()
        # FLANN parameters for SIFT
        FLANN_INDEX_KDTREE = 2
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        
    else:
        detector=cv2.ORB_create()
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 12, # 12
                   key_size = 20,     # 20
                   multi_probe_level =2) #2
        search_params = dict(checks=50)   # or pass empty dictionary
    
    matcher = cv2.FlannBasedMatcher(index_params,search_params)
    # Finde features and their descriptors
    kp_master, des_master = detector.detectAndCompute(masterImage,None)
    kp_slave, des_slave = detector.detectAndCompute(slaveImage,None)
    if len(kp_master) < 2 or len(kp_slave) < 2:
        return GCP_list
    # Get the 2 best matches per feature
    matches = matcher.knnMatch(des_master,des_slave,k=2)
    GCP_id=0
    GCP_list=[]
    for i,m_n in enumerate(matches):
        if len(m_n) != 2:
            continue
        (m,n) = m_n
        dist_ratio=m.distance/n.distance
        if  dist_ratio < match_factor:
            master_pt=np.float32( kp_master[m.queryIdx].pt)
            Xmaster,Ymaster = get_map_coordinates(float(master_pt[1]),float(master_pt[0]),masterGeo)
            slave_pt=np.float32( kp_slave[m.trainIdx].pt)
            GCP_list.append((Xmaster,Ymaster,UL_offset[0]+float(slave_pt[1]),UL_offset[1]+float(slave_pt[0]),dist_ratio))
            GCP_id+=1
    return GCP_list
    
def filter_GCP_by_map_proximity(GCPs,slaveGeo,distThres):

    GCPs=np.asarray(GCPs)
    if len(GCPs.shape)==1:
        GCPs=GCPs.reshape(1,-1)
    Xslave,Yslave = get_map_coordinates(GCPs[:,2],GCPs[:,3],slaveGeo)
    dist=np.sqrt((Xslave-GCPs[:,0])**2+(Yslave-GCPs[:,1])**2)
    GCPs=GCPs[dist<=distThres]
    return GCPs
    
def filter_GCP_by_GCP_proximity(GCPs,pixelThres):

    # Filter GCPs based on image proximity
    GCPs_good=[]
    for i,gcp_test in enumerate(GCPs):
        good =True
        if i==len(GCPs)-2: continue
        for j in range(i+1,len(GCPs)):
            #print(i,j)
            dist=np.sqrt((gcp_test[2]-GCPs[j][2])**2+(gcp_test[3]-GCPs[j][3])**2)
            if dist<pixelThres:# GCPs closer to each other are discarded to avoid overfitting
                good=False
                break
        if good==True:
            GCPs_good.append(gcp_test)

    return GCPs_good

def filter_GCP_by_intersection(GCPs,slave_geo):

    # Filter GCPs based on image proximity
    GCPs_good=[]
    GCPs=np.asarray(GCPs)
    if len(GCPs.shape)==1:
        GCPs=GCPs.reshape(1,-1)

    X_slave,Y_slave=get_map_coordinates(GCPs[:,2],GCPs[:,3],slave_geo)
    obs_vec=(GCPs[:,0]-X_slave,GCPs[:,1]-Y_slave)
    azimuth=calc_azimuth(obs_vec)
    cos_azimuth=np.cos(np.radians(azimuth))
    sin_azimuth=np.sin(np.radians(azimuth))
    mean_azimuth=np.degrees(np.arctan2(np.mean(sin_azimuth),np.mean(cos_azimuth)))
    diff=np.abs(calc_azimuth_difference(azimuth,mean_azimuth))
    indices=diff.argsort()[::-1]
    for i,index in enumerate(indices):
        good =True
        if i==len(indices)-2: continue
        for j in range(i+1,len(indices)):
            #print(i,j)
            intercept=calc_vector_intersection((X_slave[index],Y_slave[index]),(GCPs[index,0],GCPs[index,1]),
                                             (X_slave[indices[j]],Y_slave[indices[j]]),(GCPs[indices[j],0],GCPs[indices[j],1]))
            if intercept:# GCPs closer to each other are discarded to avoid overfitting
                good=False
                break
        if good==True:
            GCPs_good.append(GCPs[index])

    return GCPs_good

def filter_GCP_by_azimuth(GCPs,slave_geo,angleThres):

    # Filter GCPs based on image proximity
    GCPs=np.asarray(GCPs)
    if len(GCPs.shape)==1:
        GCPs=GCPs.reshape(1,-1)
    X_slave,Y_slave=get_map_coordinates(GCPs[:,2],GCPs[:,3],slave_geo)
    obs_vec=(GCPs[:,0]-X_slave,GCPs[:,1]-Y_slave)
    azimuth=calc_azimuth(obs_vec)
    cos_azimuth=np.cos(np.radians(azimuth))
    sin_azimuth=np.sin(np.radians(azimuth))
    mean_azimuth=np.degrees(np.arctan2(np.mean(sin_azimuth),np.mean(cos_azimuth)))
    diff=np.abs(calc_azimuth_difference(azimuth,mean_azimuth))
    GCPs_good=GCPs[diff<=angleThres]
    return GCPs_good

def filter_GCP_by_tranlation(GCPs,slave_geo,stdThres=2):

    # Filter GCPs based on image proximity
    GCPs=np.asarray(GCPs)
    if len(GCPs.shape)==1:
        GCPs=GCPs.reshape(1,-1)
    X_slave,Y_slave=get_map_coordinates(GCPs[:,2],GCPs[:,3],slave_geo)
    dist=np.log(np.sqrt((X_slave-GCPs[:,0])**2+(Y_slave-GCPs[:,1])**2))
    mean_dist=np.mean(dist)
    std_dist=np.std(dist)
    GCPs_good=GCPs[dist<=mean_dist+stdThres*std_dist]
    return GCPs_good

def filter_GCP_by_unique_coordinates(GCPs):
    GCPs=np.asarray(GCPs)
    Coord_good=[]
    GCPs_good=[]
    for GCP in GCPs:
        if (GCP[0],GCP[1]) not in Coord_good:
            Coord_good.append((GCP[0],GCP[1]))
            GCPs_good.append(GCP)
    return GCPs_good

def calc_vector_intersection(start1,end1,start2,end2):
        
    return _ccw(start1,start2,end2) != _ccw(end1,start2,end2) and _ccw(start1,end1,start2) != _ccw(start1,end1,end2)

def _ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def filter_GCP_by_warp_error(GCPs,errorThres):
    def fit_polynomial_warp(GCPs):
    
        GCPs=np.asarray(GCPs)
        if GCPs.shape[0]<15:
            return None,None
        rows=GCPs[:,2]
        cols=GCPs[:,3]
        rows2=rows**2
        cols2=cols**2
        rowscols=rows*cols
        rows2cols=rows**2*cols
        rowscols2=rows*cols**2
        rows3=rows**3
        cols3=cols**3
        
        X=np.matrix([np.ones(rows.shape),rows,cols,rowscols,rows2,cols2,rows2cols,rowscols2,rows3,cols3]).T
        mapX=GCPs[:,0].reshape(-1,1)
        mapY=GCPs[:,1].reshape(-1,1)
        thetaX=(X.T*X).I*X.T*mapX    
        thetaY=(X.T*X).I*X.T*mapY    
        return np.asarray(thetaX).reshape(-1),np.asarray(thetaY).reshape(-1)

    def calc_warp_erors(GCPs,thetaX,thetaY):
        def polynomial_warp(rows,cols,thetaX,thetaY):
        
            X=thetaX[0]+thetaX[1]*rows+thetaX[2]*cols+thetaX[3]*rows*cols+thetaX[4]*rows**2+thetaX[5]*cols**2+thetaX[6]*rows**2*cols+thetaX[7]*rows*cols**2+thetaX[8]*rows**3+thetaX[9]*cols**3
            Y=thetaY[0]+thetaY[1]*rows+thetaY[2]*cols+thetaY[3]*rows*cols+thetaY[4]*rows**2+thetaY[5]*cols**2+thetaY[6]*rows**2*cols+thetaY[7]*rows*cols**2+thetaY[8]*rows**3+thetaY[9]*cols**3
            return X,Y
    
        GCPs=np.asarray(GCPs)
        if len(GCPs.shape)==1:
            GCPs=GCPs.reshape(1,-1)
    
        rows=GCPs[:,2]
        cols=GCPs[:,3]
        X_model,Y_model=polynomial_warp(rows,cols,thetaX,thetaY)
        error=np.sqrt((X_model-GCPs[:,0])**2+(Y_model-GCPs[:,1])**2)
        return error

    GCPs=np.asarray(GCPs)
    if GCPs.shape[0]<15:
        return GCPs
    thetaX,thetaY=fit_polynomial_warp(GCPs)
    error=calc_warp_erors(GCPs,thetaX,thetaY)
    while np.max(error)>errorThres and len(error)>30:
        index=error.argsort()[::-1]
        GCPs=GCPs[index[1:]]
        thetaX,thetaY=fit_polynomial_warp(GCPs)
        error=calc_warp_erors(GCPs,thetaX,thetaY)

    return GCPs
    
def calc_azimuth(obs_vec):
    ''' Calculates the azimuth navigation heading between two positions

    Parameters
    ----------   
    lon_0, lat_0 : Initial longitude and latitude (degrees)
    lon_1, lat_1 : Final longitude and latitude (degrees)
    
    Returns
    -------
    azimuth : Azimutal heading (degrees from North)'''

    # Get the view azimuth angle
    azimuth = np.degrees(np.arctan2(obs_vec[0],obs_vec[1]))
    return azimuth

def calc_azimuth_difference(azimut1,azimut2):
    ''' Calculates the azimuth navigation heading between two positions

    Parameters
    ----------   
    lon_0, lat_0 : Initial longitude and latitude (degrees)
    lon_1, lat_1 : Final longitude and latitude (degrees)
    
    Returns
    -------
    azimuth : Azimutal heading (degrees from North)'''

    # Get the view azimuth angle
    azimuth = np.degrees(np.arctan2(np.sin(np.radians(azimut1-azimut2)),np.cos(np.radians(azimut1-azimut2))))
    return azimuth

def create_gdal_GCPs(GCPs):

    gdalGCP=[]
    for i,gcp in enumerate(GCPs):
        gdalGCP.append(gdal.GCP(gcp[0], gcp[1],0, gcp[3], gcp[2],'',str(i)))
    return gdalGCP

def GCPs_to_ASCII(GCPs,outfile):

    header='ID \t X \t Y \t Row \t Col'
    fid=open(outfile,'w')
    fid.write(header+'\n')
    for i,gcp in enumerate(GCPs):
        fid.write('%s \t %s \t %s \t %s \t %s \n'%(i,gcp[0],gcp[1],gcp[2],gcp[3]))
    fid.flush()
    fid.close()
    return True

def ASCII_to_GCP(infile,GCP=[]):

    indata=np.genfromtxt(infile,names=True,dtype=None)
    for data in indata:
        GCP.append([float(data['X']),float(data['Y']),float(data['Row']),float(data['Col'])])
    return GCP

def write_transformation_vector(slaveCoords,masterCoords,outshapefile,prj):
    from osgeo import ogr
    import osr
    import os

    DriverName = "ESRI Shapefile"
    driver = ogr.GetDriverByName(DriverName)
    if os.path.exists(outshapefile):
        driver.DeleteDataSource(outshapefile)
    
    data_source = driver.CreateDataSource(outshapefile)
    # create the layer
    srs=osr.SpatialReference()
    srs.ImportFromWkt(prj)
    layer = data_source.CreateLayer("Transformation Vectors", srs, ogr.wkbLineString)

    for i,slaveXCoord in enumerate(slaveCoords[0][:]):
        # create the feature
        feature = ogr.Feature(layer.GetLayerDefn())
 
        line = ogr.Geometry(ogr.wkbLineString)
        line.AddPoint(slaveXCoord,slaveCoords[1][i])
        line.AddPoint(masterCoords[0][i],masterCoords[1][i])
          # Set the feature geometry using the point
        feature.SetGeometry(line)
        # Create the feature in the layer (shapefile)
        layer.CreateFeature(feature)
        # Destroy the feature to free resources
        feature.Destroy()
    # Destroy the data source to free resources
    data_source.Destroy()
    return True
