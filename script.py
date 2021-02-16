# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import os
import pycrs
import geopandas as gpd
import matplotlib.pyplot as plt 
import rasterio as rio 
from rasterio.mask import mask 
from rasterio.plot import show 


# %%
data_dir = "/Users/Ronjamac/Documents/02_Studium/Masterarbeit/Code/VN_coastline_dynamics/test_data/"
raster_file = "1_1988_L5_P1_13aq.tif"
raster_file_masked = "1_1988_L5_P1_13aq_masked.tif"
box_file = "test_box.geojson"


# %%
def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]
# open input raster
raster = rio.open(os.path.join(data_dir,raster_file))
show(raster,cmap="Blues")
# open test bbox
box = gpd.read_file(os.path.join(data_dir,box_file))
coords = getFeatures(box)
out_img, out_transform = mask(dataset=raster, shapes=coords, crop=True, filled=False)
out_meta = raster.meta.copy()
epsg_code = int(raster.crs.data['init'][5:])
out_meta.update({"driver":"GTiff",
                 "height": out_img.shape[1],
                 "width": out_img.shape[2],
                 "transform": out_transform,
                 "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4()})
with rio.open(os.path.join(data_dir,raster_file_masked), "w",**out_meta) as dest:
        dest.write(out_img)


# %%
# extract contours
# open raster cropped to bbox
from coasty import postprocess

raster_masked = rio.open(os.path.join(data_dir,raster_file_masked))
#show(raster_masked,cmap="Blues")
shoreline = postprocess.subpixel_contours(raster=raster_masked, crs="EPSG:3405", scale=30)
shoreline.to_file(os.path.join(data_dir,raster_file_masked+"_shoreline"),driver="GeoJSON")
shoreline.plot()

# %%
postprocess.subpixel_contours()


