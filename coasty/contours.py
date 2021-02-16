import os
import pycrs
import numpy as np
import geopandas as gpd 
import rasterio as rio 
import matplotlib.pyplot as plt 
from rasterio.mask import mask 
from rasterio.plot import show 
from skimage import measure


data_dir = "/Users/Ronjamac/Documents/02_Studium/Masterarbeit/Code/VN_coastline_dynamics/test_data"
raster_file_masked = "1_1988_L5_P1_13aq_masked.tif"
raster_masked = rio.open(os.path.join(data_dir,raster_file_masked))
#print(raster_masked.read(1))
#show(raster_masked)

raster = raster_masked.read(1)
#contours = measure.find_contours(raster,0.5)

# Construct some test data
x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))

# Find contours at a constant value of 0.8
contours = measure.find_contours(raster, 0.8)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(raster, cmap=plt.cm.gray)

for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

