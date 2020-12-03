import subprocess
import os
from tqdm import tqdm
from pathlib import Path

os.environ['GDAL_DATA'] = '/home/maja/anaconda3/envs/eurocropsdatasetpreparation/share/gdal'
os.environ['PROJ_LIB'] = '/home/maja/anaconda3/envs/eurocropsdatasetpreparation/share/proj'

"""
Adapted from 
https://github.com/dairejpwalsh/Sentinel-Scripts/blob/master/Sentinel%202/tiff-generator.py
"""


def generate_geotiffs(SAFE_path, output_path):
    basename = os.path.basename(SAFE_path)
    _, _, datatake_sensing_starttime, _, _, _, _ = basename[:-4].split('_')

    product_name = os.path.basename(basename)[:-5]

    folders_until_imgs = os.path.join(SAFE_path, 'GRANULE')
    folders_until_imgs = [f.path for f in os.scandir(folders_until_imgs) if f.is_dir()]
    folders_until_imgs = os.path.join(folders_until_imgs[0], 'IMG_DATA/')
    generate_all_bands(folders_until_imgs, datatake_sensing_starttime, output_path)


def generate_all_bands(folders_until_imgs, datatake_sensing_starttime, output_path):
    img_name = os.path.join(folders_until_imgs, 'R10m/')
    img_name = [f.path for f in os.scandir(img_name)]
    tile_number_field, prod_discriminator_of_images, band_name, res = os.path.basename(img_name[0])[:-4].split('_')

    granule_band_template = tile_number_field + "_" + prod_discriminator_of_images + "_"

    output_tiff = '/' + datatake_sensing_starttime[:8] + '.tif'
    output_vrt = '/' + datatake_sensing_starttime[:8] + '.vrt'

    output_full_path_tiff = output_path + output_tiff
    output_full_path_vrt = output_path + output_vrt

    bands = {"band_02": folders_until_imgs + "R10m/" + granule_band_template + "B02_10m.jp2",
             "band_03": folders_until_imgs + "R10m/" + granule_band_template + "B03_10m.jp2",
             "band_04": folders_until_imgs + "R10m/" + granule_band_template + "B04_10m.jp2",
             "band_05": folders_until_imgs + "R20m/" + granule_band_template + "B05_20m.jp2",
             "band_06": folders_until_imgs + "R20m/" + granule_band_template + "B06_20m.jp2",
             "band_07": folders_until_imgs + "R20m/" + granule_band_template + "B07_20m.jp2",
             "band_08": folders_until_imgs + "R10m/" + granule_band_template + "B08_10m.jp2",
             "band_8A": folders_until_imgs + "R20m/" + granule_band_template + "B8A_20m.jp2",
             "band_11": folders_until_imgs + "R20m/" + granule_band_template + "B11_20m.jp2",
             "band_12": folders_until_imgs + "R20m/" + granule_band_template + "B12_20m.jp2"}

    cmd = ['gdalbuildvrt', '-resolution', 'user', '-tr', '10', '10', '-separate', output_full_path_vrt]

    for band in sorted(bands.values()):
        cmd.append(band)

    my_file = Path(output_full_path_vrt)
    if not my_file.is_file():
        # file exists
        subprocess.call(cmd)

    # , '-a_srs', 'EPSG:3857'
    cmd = ['gdal_translate', '-of', 'GTiff', output_full_path_vrt, output_full_path_tiff]

    my_file = Path(output_tiff)
    if not my_file.is_file():
        # file exists
        subprocess.call(cmd)


output_path = '/home/maja/Documents/SentinelTiles/T33UWP/output/'

for f in tqdm(os.scandir('/home/maja/Documents/SentinelTiles/T33UWP/L2A')):
    input_path = f.path
    generate_geotiffs(input_path, output_path)

