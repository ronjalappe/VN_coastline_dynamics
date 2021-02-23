import geopandas as gpd
import shapely as shp
from geosmoothing.geosmoothing import GeoSmoothing


def simplify_lines(gdf, tolerance=200):
    simpler_lines = []
    for index, line in gdf.iterrows():
        simpler = gdf.geometry.iloc[index].simplify(tolerance=tolerance)
        simpler_lines.append(simpler)
    simpler_lines_gdf = gpd.GeoDataFrame(geometry=simpler_lines,crs=gdf.crs)
    return simpler_lines_gdf

def geosmooth_lines(gdf,min_length=500):
    gsm = GeoSmoothing()
    smooth_lines = []
    for index, row in gdf.iterrows():
        geometry = gdf.geometry.iloc[index]
        if type(geometry) == shp.geometry.linestring.LineString:
            line = geometry
            if line.length > min_length:
                smooth = gsm.smoothWkt(line.wkt)
                smooth = shp.wkt.loads(smooth)
                smooth_lines.append(smooth)
        elif type(geometry) == shp.geometry.multilinestring.MultiLineString:
            for line in geometry:
                if line.length > min_length:
                    smooth = gsm.smoothWkt(line.wkt)
                    smooth = shp.wkt.loads(smooth)
                    smooth_lines.append(smooth)
    lines_smooth_gdf = gpd.GeoDataFrame(geometry=smooth_lines,crs=gdf.crs)
    return lines_smooth_gdf

def draw_transects(gdf, length, distance):
    crs = gdf.crs
    all_transects = []
    for index, line in gdf.iterrows():
        line = gdf.geometry.iloc[index]
        type_MLS = shp.geometry.multilinestring.MultiLineStringlength
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
                    left = ab.parallel_offset(length/4, 'left') #/2
                    right = ab.parallel_offset(length, 'right') #/2
                    c = left.boundary[1]
                    d = right.boundary[0]  # note the different orientation for right offset
                    cd = shp.geometry.LineString([c, d])
                    transects.append(cd)
            transects_gdf = gpd.GeoDataFrame(geometry=transects,crs=crs)
            all_transects.append(transects_gdf)

    all_transects_gdf = pd.concat(all_transects,ignore_index=True)
    all_transects_gdf['id'] = all_transects_gdf.index
    return all_transects_gdf