# -*- coding: utf-8 -*-

import optlang
from time import process_time
import json

def 

time_1 = process_time()
with open('sub_timesteps/simple_full_community/mscommfitting.json', 'r') as mscmft:
    mscomfit_json = json.load(mscmft)
time_2 = process_time()
print(f'Done loading the JSON: {(time_2-time_1)/60} min')

cvf = 2
cvt = 3
diff = 200

for arg in mscomfit_json['objective']['expression']['args']:
    if 'cvf' in arg['args'][1]['name']:
        arg['args'][0]['value'] = cvf
    elif 'cvt' in arg['args'][1]['name']:
        arg['args'][0]['value'] = cvt
    elif 'diff' in arg['args'][1]['name']:
        arg['args'][0]['value'] = diff

time_3 = process_time()
print(f'Done editing the objective: {(time_3-time_2)/60} min')

with open('sub_timesteps/simple_full_community/mscommfitting_edited.json', 'w') as mscmft:
    json.dump(mscomfit_json, mscmft, indent=3)
time_4 = process_time()
print(f'Done exporting the model: {(time_4-time_3)/60} min')
    
model = optlang.Model.from_json(mscomfit_json)
time_5 = process_time()
print(f'Done loading the model: {(time_5-time_4)/60} min')  # ~1/2 the defining a new problem