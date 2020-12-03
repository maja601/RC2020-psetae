import json
import copy

"""
Hand-crafted approach to harmonise the France and Slovenia Dataset
"""

labels_to_subst = {
    33200000: 1,
    33101022: 3,
    33101011: 5,
    33101031: 5,
    33101041: 5,
    33101051: 5,
    33101071: 5,
    33101112: 6,
    33101032: 6,
    33101042: 6,
    33101052: 6,
    33101072: 6,
    33101080: 8,
    33101000: 9,
    33101100: 9,
    33101060: 9,
    33101090: 9,
    33105000: 12,
    33106042: 14,
    33106050: 16,
    33106060: 18,
    33102000: 19,
    33103000: 23,
    33107000: 28,
    33110000: 28,
    33301010: 33,
    33304000: 34,
}

with open('/home/maja/ssd/rc2020dataset/Dataset_4_garnot/pixelset/META/labels.json') as json_file:
    data = json.load(json_file)
    subst_data = copy.deepcopy(data)

    for p in data['c_group_co']:
        if data['c_group_co'][p] in labels_to_subst:
            subst_data['c_group_co'][p] = labels_to_subst[data['c_group_co'][p]]

    with open('/home/maja/ssd/rc2020dataset/Dataset_4_garnot/eurocrops_as_garnot/META/labels.json', 'w') as outfile:
        json.dump(subst_data, outfile, indent=4)
