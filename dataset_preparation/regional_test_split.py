import os
from tqdm import tqdm
import json

org_folder = '/home/maja/ssd/rc2020dataset/Dataset_4_garnot/eurocrops_as_garnot/DATA/'
new_folder = '/home/maja/ssd/rc2020dataset/Dataset_4_garnot/eurocrops_as_garnot/TESTDATA/'
with open('/home/maja/ssd/rc2020dataset/Dataset_4_garnot/slovenia_32633.geojson') as f:
    data = json.load(f)

for f in tqdm(data['features']):
    id = str(f['properties']['ID'])
    nuts = f['properties']['NUTS_ID']
    if nuts == 'SI034':
        print(1)
        os.rename(org_folder + id + '.npy', new_folder + id + '.npy')