import numpy as np
import os
import json

datapath = os.path.join(os.path.dirname(__file__) + '/../datasets/toydata/DATA/')
metapath = os.path.join(os.path.dirname(__file__) + '/../datasets/toydata/META/')

with open(os.path.join(metapath + 'labels.json')) as f:
    labels_json = json.load(f)

print(len(labels_json['label_44class']))

datast = {}
classes = [1, 3, 4, 5, 6, 8, 9, 12, 13, 14, 16, 18, 19, 23, 28, 31, 33, 34, 36, 39]

for key, value in labels_json['label_44class'].items():
    if value in classes:
        datast[key] = value

print(len(datast))
# x = sorted(os.listdir(datapath))
# for filename in os.listdir(datapath):
#     if filename.endswith('.npy'):
#         data_array = np.load(datapath + filename)
#         parcel_nr = filename[:-4]
#         lbl = labels_json['label_19class'][parcel_nr]
#         print('done')