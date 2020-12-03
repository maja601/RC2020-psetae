import json
from tqdm import tqdm
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon

"""
A short script to calculate the geometric features mentioned in the paper
"""

with open('/home/maja/ssd/rc2020dataset/Dataset_4_garnot/slovenia_32633.geojson') as f:
    data = json.load(f)

with open('/home/maja/ssd/rc2020dataset/Dataset_4_garnot/eurocrops_as_garnot/META/sizes.json') as f:
    sizes = json.load(f)

geomfeat = {}
for f in tqdm(data['features']):
    id = str(f['properties']['ID'])
    if id in sizes:
        p = Polygon(f['geometry']['coordinates'][0][0])
        perimeter = p.length
        pixel_count = sizes[id]
        bounds = p.bounds
        b_area = (bounds[3] - bounds[1]) * (bounds[2] - bounds[0])
        b_pixel = b_area / 100
        area = p.area
        cover_ratio = pixel_count / b_pixel
        per_sur_ratio = perimeter / area
        geomfeat[id] = [perimeter, b_area, cover_ratio, per_sur_ratio]

with open('/home/maja/ssd/rc2020dataset/Dataset_4_garnot/eurocrops_as_garnot/META/geomfeat.json', 'w') as outfile:
    json.dump(geomfeat, outfile)


