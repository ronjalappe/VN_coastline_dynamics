import numpy as np
import shapely as shp
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio as rio 
from rasterio.warp import calculate_default_transform, reproject, Resampling
from skimage import measure
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

def reproject_raster(raster_path, out_path, dst_crs):
    """Reprojects GeoTIFF to a desired CRS. The functions reads the GeoTIFF, reprojects
       to CRS and saves it as specified in out_path.

    Args:
        raster_path (string): Path to GeoTIFF raster file (1 Band)
        out_path (string): Path to reprojected GeoTIFF that will be created     
        dst_crs (string): CRS of type "EPSG:xxxx"
    """
    with rio.open(raster_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
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


def subpixel_contours(raster, scale, threshold=0.5, sigma=3):
    """Uses skimage.measure.find_contours to extract contours from a binary image. 
    Smoothens contour line with scipy.interpolate to achieve a subpixel border
    segmentation. 

    Args:
        raster (rasterio raster): raster.io Dataset reader (with rasterio.open)
        scale (int): Scale of input raster (e.g. 30)
        threshold (float, optional): Threshold to position of subpixel contour line. Defaults to 0.5.
        sigma (int, optional): Level of smoothing (compare scipy.ndimage.gaussian_filter1d). Defaults to 3.

    Returns:
        Geopandas.GeoDataFrame : GeoDataFrame that contains one MultiLineString with all detected, smoothed 
        contours. Projected to the crs of the input raster. 
    """
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
    new_gdf['length'] = new_gdf.geometry.length
    # get indices of items smaller than min_size
    too_small_indices = new_gdf[new_gdf.length < min_size].index
    # remove all entries with a length smaller than min_size
    new_gdf.drop(too_small_indices,inplace=True)
    # save in new dataframe with resetted indices
    new_gdf = new_gdf.reset_index(drop=True)
    return new_gdf


def draw_transects(gdf, length, distance):

    crs = gdf.crs
    all_transects = []
    for index, line in gdf.iterrows():
        line = gdf.geometry.iloc[index]
        type_MLS = shp.geometry.multilinestring.MultiLineString
        type_LS = shp.geometry.linestring.LineString
        if type(line) == type_MLS or type_LS:
            if line.length > 500:
                n_points = int(line.length/distance)
            print(index, "n_points:", n_points)        
            new_points = [line.interpolate(p/float(n_points - 1), normalized=True) for p in range(n_points)]
            new_line = shp.geometry.LineString(new_points)
            transects = []
            for index, point in enumerate(new_points): 
                if index+1 < len(new_points):
                    a = new_points[index]
                    b = new_points[index+1]
                    ab = shp.geometry.LineString([a,b])
                    left = ab.parallel_offset(length / 2, 'left')
                    right = ab.parallel_offset(length / 2, 'right')
                    c = left.boundary[1]
                    d = right.boundary[0]  # note the different orientation for right offset
                    cd = shp.geometry.LineString([c, d])
                    transects.append(cd)
            transects_gdf = gpd.GeoDataFrame(geometry=transects,crs=crs)
            all_transects.append(transects_gdf)
    all_transects_gdf = pd.concat(all_transects,ignore_index=True)
    all_transects_gdf['id'] = all_transects_gdf.index
    return all_transects_gdf





# TEST IT 
import os
data_dir = os.path.join(os.getcwd(),"test_data")
raster_file_masked = "1_1988_L5_P1_13aq_masked.tif"
vn_crs = "EPSG:3857" #EPSG:3857" #projected coordinate system of the world
out_path = os.path.join(data_dir, os.path.splitext(raster_file_masked)[0]+"_reproj.tif")
rfsl_file = "OSM_coastline_VN_simplified"
box_file = "test_box.geojson"



#raster_masked = rio.open(os.path.join(data_dir, raster_file_masked))
#reproject_raster(os.path.join(data_dir, raster_file_masked), out_path, vn_crs)
#raster_reproj = rio.open(out_path)
#shoreline = subpixel_contours(raster=raster_reproj, scale=30)
#shoreline.to_file(os.path.join(data_dir, os.path.splitext(raster_file_masked)[0]+"_shoreline.geojson"), driver="GeoJSON")
#shoreline_cleaned = remove_small_lines(shoreline, 1000)
#shoreline_cleaned.to_file(os.path.join(data_dir,os.path.splitext(raster_file_masked)[0]+"_shoreline_cleaned.geojson"),driver="GeoJSON")

box = gpd.read_file(os.path.join(data_dir,box_file))
box = box.to_crs(vn_crs)
rfsl = gpd.read_file(os.path.join(data_dir,"osm_coastline_epsg4326.geojson"))
rfsl = rfsl.to_crs(vn_crs)
rfsl = gpd.clip(rfsl,box).reset_index(drop=True)
rfsl.to_file(os.path.join(data_dir,"osm_coastline_clip"),driver="GeoJSON")

simpler_lines = []
for index, line in rfsl.iterrows():
    print(index)
    simpler = rfsl.geometry.iloc[index].simplify(tolerance=200)
    simpler_lines.append(simpler)
simpler_lines_gdf = gpd.GeoDataFrame(geometry=simpler_lines,crs=rfsl.crs)
simpler_lines_gdf.to_file(os.path.join(data_dir,"osm_coastline_clip_simplified"),driver="GeoJSON")

from geosmoothing.geosmoothing import GeoSmoothing
gsm = GeoSmoothing()
simpler_lines_smooth = []
for index, row in simpler_lines_gdf.iterrows():
    geometry = simpler_lines_gdf.geometry.iloc[index]
    if type(geometry) == shp.geometry.linestring.LineString:
        line = geometry
        if line.length > 500:
            smooth = gsm.smoothWkt(line.wkt)
            smooth = shp.wkt.loads(smooth)
            simpler_lines_smooth.append(smooth)
    elif type(geometry) == shp.geometry.multilinestring.MultiLineString:
        for line in geometry:
            if line.length > 500:
                smooth = gsm.smoothWkt(line.wkt)
                smooth = shp.wkt.loads(smooth)
                simpler_lines_smooth.append(smooth)
simpler_lines_smooth_gdf = gpd.GeoDataFrame(geometry=simpler_lines_smooth,crs=vn_crs)
simpler_lines_smooth_gdf.to_file(os.path.join(data_dir,"osm_coastline_clip_simplified_smooth"),driver="GeoJSON")

transects = draw_transects(simpler_lines_smooth_gdf,3000,100)
transects.to_file(os.path.join(data_dir,"osm_coastline_clip_simplified_smooth_transects"),driver="GeoJSON")

print('Done!')
