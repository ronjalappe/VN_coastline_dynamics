import numpy as np
import shapely as shp
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio as rio 
from rasterio.mask import mask 
from rasterio.warp import calculate_default_transform, reproject, Resampling
from skimage import measure
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

def download_from_drive(export_folder, out_path, tile_name):
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    # download rasters from Google drive
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth() # Creates local webserver and auto handles authentication.
    drive = GoogleDrive(gauth)

    folder_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    exportFolderID = [folder['id'] for folder in folder_list if folder['title']==export_folder][0]
    file_list = drive.ListFile({'q': "'" + exportFolderID + "' in parents and trashed=false"}).GetList()
    # select only files that have been created with under the current tile name 
    files = []
    for file in file_list:
        if file['title'].endswith(tile_name+".tif"):
            files.append(file)
    # download selected files
    for file in files:
        #print('title: %s, id: %s' % (file['title'], file['id']))
        fname = file['title']
        print(fname)
        if not os.path.exists(os.path.join(out_path,fname)):
            print('Downloading {}...'.format(fname))
            f = drive.CreateFile({'id': file['id']})
            f.GetContentFile(fname)
        else:
            print('File', fname, 'already exists.')

def reproject_raster(raster_path, out_path, dst_crs):
    """Reprojects GeoTIFF to a desired CRS. The functions reads the GeoTIFF, reprojects
       to CRS and saves it as specified in out_path.

    Args:
        raster_path (string): Path to GeoTIFF raster file (1 Band)
        out_path (string): Path to reprojected GeoTIFF that will be created     
        dst_crs (string): CRS of type "EPSG:xxxx"
    """
    with rio.open(raster_path) as src:
        if not dst_crs == src.crs:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'compress':'LZW'
            })
            with rio.open(out_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rio.band(src, i),
                        destination=rio.band(dst, i),   
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)
            print(os.path.basename(raster_path),'reprojected.')
        else:
            print(os.path.basename(raster_path),'already projected to given CRS.')

def mask_single_observation_pixel(raster_path):
    with rio.open(raster_path, "r") as src:  # src has meta that can be accessed through src.meta or directly, e.g. src.height
        meta = src.meta
        binary = src.read(1)
        nobs = src.read(2)
        avg_aq = int(np.nanmean(nobs))    
        binary_masked = binary.copy()
        binary_masked[nobs == 1] = np.nan
        print(os.path.basename(raster_path),"masked.")
        meta.update(
            count=1,
            compress="lzw")
        out_file = os.path.splitext(raster_path)[0]+"_"+str(avg_aq).zfill(2)+"avg_aq.tif"
        print(os.path.basename(out_file),"saved.")
        with rio.open(out_file, "w", **meta) as dst:
            dst.write(binary_masked, 1)



def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def crop_raster(raster_path, gdf_path, out_path):
    raster = rio.open(raster_path)
    crs = raster.crs
    test_box = gpd.read_file(gdf_path)
    test_box = test_box.to_crs(crs)
    coords = getFeatures(test_box)
    out_img, out_transform = rio.mask.mask(dataset=raster, shapes=coords, crop=True, filled=False)
    out_meta = raster.meta.copy()
    out_meta.update({"driver": "GTiff",
                    "height": out_img.shape[1],
                    "width": out_img.shape[2],
                    "transform": out_transform,
                    "crs": crs,
                    "compress":"LZW"})
    with rio.open(out_path, "w", **out_meta) as dest:
        dest.write(out_img)

def polygonize_raster(raster_path,raster_value,min_length):
    # Read input band with Rasterio
    with rio.open(raster_path) as src:
        crs = src.crs
        src_band = src.read(1)
        src_band[src_band==0]=2
        src_band[src_band==1]=0
        src_band[src_band==2]=1
        # Polygonize with Rasterio. `shapes()` returns an iterable
        # of (geom, value) as tuples
        shapes = list(rio.features.shapes(src_band, transform=src.transform))
    shp_schema = {
        'geometry': 'MultiPolygon',
        'properties': {'pixelvalue': 'int'}
        }
    # keep polygons with specified raster pixel value    
    polygons = [shp.geometry.shape(geom) for geom, value in shapes
                if value == raster_value]
    # save polygons as geodataframe
    polygons_gdf = gpd.GeoDataFrame(geometry=polygons,crs=crs)
    # remove small polygons with given min length
    polygons_gdf = postprocess.remove_small_lines(polygons_gdf,min_length)
    return polygons_gdf
    
def subpixel_contours(raster_path, scale, threshold=0.5, sigma=3):
    """Uses skimage.measure.find_contours to extract contours from a binary image. 
    Smoothens contour line with scipy.interpolate to achieve a subpixel border
    segmentation. 

    Args:
        raster (string): path to raster file 
        scale (int): Scale of input raster (e.g. 30)
        threshold (float, optional): Threshold to position of subpixel contour line. Defaults to 0.5.
        sigma (int, optional): Level of smoothing (compare scipy.ndimage.gaussian_filter1d). Defaults to 3.

    Returns:
        Geopandas.GeoDataFrame : GeoDataFrame that contains one MultiLineString with all detected, smoothed 
        contours. Projected to the crs of the input raster. 
    """
    with rio.open(raster_path,"r") as raster:
        crs = raster.crs
        # contruct 2d.array from raster image
        array = raster.read(1)
    # extract contours at given threshold
    contours = measure.find_contours(array, threshold)
    # create a list where to store the smoothed lines in 
    smooth_lines= []
    # iterate through each of the extracted contour lines
    for n, contour in enumerate(contours):
        # define the length of each contour line in m
        length = shp.geometry.asLineString(contour).length*scale 
        # define the number of points to interpolate according to the length of the linestring
        if length < 1000:
            # define a minimum of interpolation points
            n_points = 100
        else:
            n_points = int(length/10)
        # get x and y coordinates from contours
        xr = []
        yr = []
        xr.append(contour[:,1])
        yr.append(contour[:,0])
        # get values within array
        x = xr[0]
        y = yr[0]
        # number of points where to interpolate
        t = np.linspace(0, 1, len(x))
        t2 = np.linspace(0, 1, n_points)
        # interpolate coordinates
        x2 = np.interp(t2, t, x)
        y2 = np.interp(t2, t, y)
        # apply gaussian filter for smoothing 
        x3 = gaussian_filter1d(x2, sigma)
        y3 = gaussian_filter1d(y2, sigma)
        # interpolate on smoothed line
        x4 = np.interp(t, t2, x3)
        y4 = np.interp(t, t2, y3)
        # save new x and y coordinates as numpy array     
        smooth_contour = np.array([[x, y] for x, y in zip(x4, y4)])
        # convert to shapely linestring 
        line = shp.geometry.asLineString(smooth_contour)
         # add affine transformation 
        t = raster.transform
        # add half a pixel to xoff and yoff to place the reference coordinate in the pixel center 
        aff_matrix= [t.a, t.b, t.d, t.e,t.xoff+t.a*0.5,t.yoff+t.e*0.5]
        line = shp.affinity.affine_transform(line, aff_matrix)
        # add to list of lines
        smooth_lines.append(line)
    # create geodataframe from the list of contour lines
    smooth_lines_gdf = gpd.GeoDataFrame(geometry = smooth_lines,crs=crs)
    #smooth_lines_gdf = smooth_lines.to_crs(crs)
    return smooth_lines_gdf

def remove_small_lines(gdf, min_size):
    """Remove lines smaller than min_size from LineString or MultiLineString GeoDataFrame

    Args:
        gdf (GeoDataFrame): Geopandas GeoDataFrame containing shapely LineString or MultiLineString geometries.
        min_size (int): Minimum length of lines to keep. Keep units of the chosen coordinate system in mind.
        A projected coordinate system is recommended for any measurements.
                        
    Returns:
        GeoDataFrame: Geopanads GeoDataFrame containismooth_linesng LineString or MultiLineString geometries with added
        'length' information.
    """
    # get the length of each item in dataframe
    new_gdf = gdf.explode()
    new_gdf['length'] = round(new_gdf.geometry.length,2)
    # get indices of items smaller than min_size
    too_small_indices = new_gdf[new_gdf.length < min_size].index
    # remove all entries with a length smaller than min_size
    new_gdf.drop(too_small_indices,inplace=True)
    # save in new dataframe with resetted indices
    new_gdf = new_gdf.reset_index(drop=True)
    return new_gdf
    
def draw_transects_polygon(gdf, length, distance, min_line_length):
    """Create transects perpendicular to a polygons outline. The shape of the
    the polygon is smoothed using a Gaussian Filter (Sigma: 3).  

    Args:
        gdf (GeoDataFrame): Geopandas GeoDataFrame with Polygons or Multipolygons
        length (int): Length of the transects in m
        distance (int): Distance between transects in m
        min_line_length (int): Minimum length of the Polygon line to generate transects at the

    Returns:
        GeoDataFrame: Geopandas GeoDataFrame with transects as LineStrings 
    """
    crs = gdf.crs
    all_transects = []
    if type(gdf.geometry.iloc[0]) == shp.geometry.multipolygon.MultiPolygon:
        print("Yes I'm multiple, sorry.")
        gdf = gdf.explode().reset_index(drop=True)
    polygons = []
    for index, row in gdf.iterrows():
        poly = row.geometry
        if poly.length > min_line_length:
            n_points = int(poly.length/distance)
            print(index, "n_points:", n_points)        
            new_xy = np.transpose([poly.exterior.interpolate(t).xy for t in np.linspace(0,poly.length,n_points,False)])[0]
            x = new_xy[0]
            y = new_xy[1]
            # number of points where to interpolate
            t1 = np.linspace(0, 1, len(x))
            t2 = np.linspace(0, 1, n_points)
            x2 = gaussian_filter1d(x, 3)
            y2 = gaussian_filter1d(y, 3)
            # interpolate on smoothed line
            x3 = np.interp(t1, t2, x2)
            y3 = np.interp(t1, t2, y2)
            new_points = np.array([[x, y] for x, y in zip(x3, y3)])
            new_polygon = shp.geometry.asPolygon(new_points)
            polygons.append(new_polygon)
            transects = []
            for index, point in enumerate(new_points): 
                if index+1 < len(new_points):
                    a = new_points[index]
                    b = new_points[index+1]
                    ab = shp.geometry.LineString([a,b])
                    left = ab.parallel_offset(length/2, 'left')
                    right = ab.parallel_offset(length/2, 'right')
                    c = left.boundary[1]
                    d = right.boundary[0]  # note the different orientation for right offset
                    cd = shp.geometry.LineString([c,d])
                    transects.append(cd)
            transects_gdf = gpd.GeoDataFrame(geometry=transects,crs=crs)
            all_transects.append(transects_gdf)
    all_polygons_gdf = gpd.GeoDataFrame(geometry=polygons,crs=crs)
    all_transects_gdf = pd.concat(all_transects,ignore_index=True)
    all_transects_gdf['id'] = all_transects_gdf.index
    return all_transects_gdf

def compute_intersections(transects, shorelines, reference=None):
    """This functions calculates intersections between shore-perpendicular transects and a GeoDataFrame with shorelines. 
    It calculates the distance of the each intersection point to the origin of the transects and adds it as a property to 
    the output intersections GeoDataFrame. If the parameter "reference" is given, the distance of each intersection point 
    to a reference shoreline is calculated additionally in order to only keep the intersection point of each year which is 
    closest to the reference line.

    Args:
        transects (GeoDataFrame): with LineStrings [required column: "transect_id"]
        shorelines (GeoDataFrame): with LineStrings and/ or MultiLineStrings [recommended column: "year"]
        reference (GeoDataFrame, optional): with LineStrings and/ or MultiLineString. Defaults to None.

    Returns:
        GeoDataFrame: with Points and information on location, transect number, (year) and distance to
    transect origin and the reference line, if given.
    """
    # crs
    crs = shorelines.crs
    transects = transects.to_crs(crs)
    if reference is not None:
        reference = reference.to_crs(crs)
        ref_inter = compute_intersections(transects, reference)
    # empty list to store point dataframes for all transects
    all_intersections = []
    # loop through all transects and compute intersections
    for t, transect in transects.iterrows():
        transect = transect.geometry
        transect_id = t
        # empty list to store intersection points dataframe for each transect
        intersections = []
        # loop through all shorelines 
        for s, shoreline in shorelines.iterrows():
            shoreline = shoreline.geometry
            inter = []
            # handle single linestrings
            if type(shoreline)==shp.geometry.linestring.LineString:
                inter.append(transect.intersection(shoreline))
            # handle mutlilinestrings
            elif type(shoreline)==shp.geometry.multilinestring.MultiLineString:
                for sh in shoreline:
                    inter.append(transect.intersection(sh))
            # create geodataframe from list of intersection points 
            gdf = gpd.GeoDataFrame(geometry=inter, crs=crs)
            # add transect id
            gdf['transect_id'] = transect_id
            # add year
            if 'year' in shorelines:
                gdf['year'] = shorelines.year.loc[s]
            # add to list
            intersections.append(gdf)
        # merge dataframes of each transect to one 
        intersections_gdf =  pd.concat(intersections,ignore_index=True)
        # drop empty geometries
        intersections_gdf = intersections_gdf[~intersections_gdf.is_empty].reset_index(drop=True)
        # separate Multipoint geometries 
        intersections_gdf = intersections_gdf.explode()
        # calculate the distance of intersections points to the (landwards) origin of the transect
        dist = []
        for i, inter in intersections_gdf.iterrows():
            origin = shp.geometry.Point(transect.coords[1])
            dist.append(origin.distance(inter.geometry))
        # add distance information to dataframe
        intersections_gdf['dist_to_transect_origin'] = dist
        # additionally calculate distance to reference shoreline
        if reference is not None:
            dist_to_osm_sl = []
            for p, point in intersections_gdf.iterrows():
                sl_point = point.geometry
                osm_point = ref_inter[ref_inter.transect_id==point.transect_id].geometry.iloc[0]
                dist = sl_point.distance(osm_point)
                dist_to_osm_sl.append(dist)
            intersections_gdf["dist_to_osm_sl"] = dist_to_osm_sl
            intersections_gdf = intersections_gdf.sort_values(by="dist_to_osm_sl")
            intersections_gdf = intersections_gdf.drop_duplicates(subset="year",keep="first")
        #max_dist = np.max(intersections_gdf.dist_to_transect_origin)
        #intersections_gdf['change'] = intersections_gdf.dist_to_transect_origin - max_dist        
        #intersections_gdf = intersections_gdf.drop_duplicates(subset="year",keep="first")
        # sort dataframe by date
        if 'year' in intersections_gdf:
            intersections_gdf = intersections_gdf.sort_values(by="year")
        # add dataframe to list
        all_intersections.append(intersections_gdf)
    # merge all dataframes
    new_gdf = pd.concat(all_intersections,ignore_index=True)
    new_gdf = new_gdf.to_crs(crs)
    return new_gdf


# TEST IT 
#import os
#gee_download_from_drive('GEE',os.path.join(os.getcwd(),'data'),'P0')
#data_dir = os.path.join(os.getcwd(),"test_data")
#raster_file = "1_1988_L5_P1_13aq.tif"
#out_path = os.path.join(data_dir, 'P1',os.path.splitext(raster_file)[0]+"_reproj.tif")



#raster_file_masked = "1_1988_L5_P1_13aq_clip.tif"
###raster_file = os.path.splitext(raster_file_masked)[0]
#vn_crs = "EPSG:3857" #EPSG:3857" #projected coordinate system of the world
#out_path = os.path.join(data_dir, os.path.splitext(raster_file_masked)[0]+"_reproj.tif")
#rfsl_file = "OSM_coastline_VN_simplified"
#box_file = "test_box.geojson"
#poly_file = "VN_processing_polygons_EPSG4326.geojson"


#raster = rio.open(os.path.join(data_dir, 'P1',raster_file))
#reproject_raster(os.path.join(data_dir,'P1',raster_file), out_path, vn_crs)

#raster_masked = rio.open(os.path.join(data_dir, raster_file_masked))
#reproject_raster(os.path.join(data_dir, raster_file_masked), out_path, vn_crs)
#raster_reproj = rio.open(out_path)
#shoreline = subpixel_contours(raster=raster_reproj, scale=30)
#shoreline.to_file(os.path.join(data_dir, os.path.splitext(raster_file_masked)[0]+"_shoreline.geojson"), driver="GeoJSON")
#shoreline_cleaned = remove_small_lines(shoreline, 1000)
#shoreline_cleaned.to_file(os.path.join(data_dir,os.path.splitext(raster_file_masked)[0]+"_shoreline_cleaned.geojson"),driver="GeoJSON")

#poly = gpd.read_file(os.path.join(data_dir,poly_file))
#poly = poly[poly.id == 1]
#poly = poly.to_crs(vn_crs)

#box = gpd.read_file(os.path.join(data_dir,box_file))
#box = box.to_crs(vn_crs)

#rfsl = gpd.read_file(os.path.join(data_dir,"osm_coastline_epsg4326.geojson"))
#rfsl = rfsl.to_crs(vn_crs)
#rfsl_clip = gpd.clip(rfsl,box).reset_index(drop=True)
#rfsl_clip.to_file(os.path.join(data_dir,"osm_coastline_clip"),driver="GeoJSON")


#simpler_lines = simplify_lines(rfsl_clip,200)
#simpler_lines.to_file(os.path.join(data_dir,"osm_coastline_clip_simplified"),driver="GeoJSON")

#simpler_lines_smooth = geosmooth_lines(simpler_lines,500)
#simpler_lines_smooth.to_file(os.path.join(data_dir,"osm_coastline_clip_simplified_smooth"),driver="GeoJSON")

#VN_land = gpd.read_file(os.path.join(data_dir,"country_bounds.geojson"))
#VN_land = VN_land.to_crs("EPSG:3857")
#VN_land_clip = gpd.clip(VN_land,box)

#VN_land_clip = VN_land_clip.buffer(-1000)
#VN_land_clip = gpd.GeoDataFrame(geometry=VN_land_clip,crs=vn_crs)
#VN_land_clip.to_file(os.path.join(data_dir,"country_bounds_clip_buffer"),driver="GeoJSON")

#transects = draw_transects_polygon(VN_land,5000,200,10000)
#transects.to_file(os.path.join(data_dir,"country_bounds_transects"),driver="GeoJSON")

#transects = gpd.read_file(os.path.join(data_dir,"transects_clip"))
#shoreline = gpd.read_file(os.path.join(data_dir, raster_file+"_shoreline_cleaned.geojson"))

#intersections = compute_intersections(transects,shoreline)
#intersections.to_file(os.path.join(data_dir,"intersections"),driver="GeoJSON")

#print('Done!')
