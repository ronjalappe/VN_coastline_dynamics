
import rasterio as rio 
import numpy as np
import geopandas as gpd
import shapely as shp
from skimage import morphology

def calc_water_extent(files_list,min_file,max_file):
    """Calculate minimum and maximum water extent raster from a list of binary water/ on-water
    files, where water=1 and non-water=0        

    Args:
        files_list (list): List with paths to raster files
        min_file (string): Path to min water extent file, that will be created.
        max_file (string): Path to max water extent file, that will be created. 
    """
    all_masks = None
    for idx, file in enumerate(files_list):#
        print("Eating file: %s" % file)
        with rio.open(file, "r") as src:  # src has meta that can be accessed through 
                                            # src.meta or directly, e.g. src.height
            if all_masks is None:  # we have not defined it yet but we only have do define ones
                all_masks = np.zeros((len(files_list), src.height, src.width), dtype=np.float32)  # np.float32 may have nans
                meta = src.meta
            all_masks[idx] = src.read(1)                    
    min_water_extent = np.nanmin(all_masks, 0)  # water = 1, min water extent
    max_water_extent = np.nanmax(all_masks, 0)  # no water = 0, max water extent
    # write the masks
    for arr, out_file in zip([min_water_extent, max_water_extent], [min_file, max_file]):
        with rio.open(out_file, "w", **meta) as tgt:
            tgt.write(arr,1)

def remove_pixel_cluster(raster_path,out_path,min_size1,min_size0,connectivity=0):
    """The function generalizes a binary raster by removing objects smaller than a specified size.
    using the skimage.morphology.remove_small_objects method. The result is saved as out_path.  

    Args:
        raster_path (string): Path to raster that shall be generalized
        out_path (string): Path of output raster
        min_size1 (int): The smallest allowable object size for water within non-water
        min_size0 (int): The smallest allowable object size for non-water within water
        connectivity (int, optional): The connectivity defining the neighborhood of a pixel. Defaults to 0.
    """
    with rio.open(raster_path,'r') as src:
        im = src.read(1)
        meta = src.meta
        im_rev = im.copy()
        im_rev[im_rev==0]=2
        im_rev[im_rev==1]=0
        im_rev[im_rev==2]=1
        processed_rev = morphology.remove_small_objects(im_rev.astype(bool),min_size=min_size1,connectivity=connectivity).astype('int16')
        im[processed_rev==0]=1
        processed = morphology.remove_small_objects(im.astype(bool), min_size=min_size0, connectivity=connectivity).astype('int16')
        meta.update({
            "compress":"LZW",
            "dtype":"int16"
        })
        with rio.open(out_path,'w',**meta) as dst:
            dst.write(processed, 1)

def vectorize_raster(raster_path,raster_value):
    """Vectorizes all pixel specified as raster_value.

    Args:
        raster_path (string): Path of raster to vectorize_raster
        raster_value (int): Raster value to vectorize

    Returns:
        GeoDataFrame: Geopandas GeoDataFrame with polygons  
    """
    # Read input band with Rasterio
    with rio.open(raster_path) as src:
        crs = src.crs
        src_band = src.read(1)
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
    return polygons_gdf


