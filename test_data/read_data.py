# %%
import numpy as np
import obspy
import re
import pandas as pd



source = np.load('o_source.npy')
station = np.load('o_station.npy')
tt_p = np.load('tt_P.npy')
print(tt_p)
#hypo = obspy.read('hypocenter.CNV')
#velest = obspy.read('velest.cnv')
#print(station)
#print(velest)
# print(max(source[:,0]))
# print(max(source[:,1]))
# print(min(source[:,0]))
# print(min(source[:,1]))
# print('-------------')
# print(max(station[:,0]))
# print(max(station[:,1]))
# print(min(station[:,0]))
# print(min(station[:,1]))

column = ['event_id','event_lat','event_lon','event_depth_km','station_id','travel_time','phase_type']
picks = pd.DataFrame(columns = column)

events = pd.DataFrame(columns=['event_id','latitude','longitude','depth_km'])
with open('hypocenter.CNV', 'r') as file:
    lines = file.readlines()

index = -1
event_id = -1
for line in lines:
    if line[:2] == ' 0':
        event_info = line.split()
        event_lat = float(re.search(r'\d+\.\d+',event_info[3]).group())
        event_lon = float(re.search(r'-?\d+\.\d+',event_info[4]).group()) * (-1)
        event_depth_km = float(re.search(r'\d+\.\d+',event_info[5]).group())
        event_id += 1
        events.loc[event_id] = [event_id, event_lat, event_lon, event_depth_km]
    if line[:2] == 'ST':
        station_info = line.split('ST')
        for pick in station_info:
            if pick[:1] != '':
                index += 1 
                if re.search('P0', pick):
                    phase_type = 'P'
                else:
                    phase_type = 'S'
                pick_info = re.split('P0|S0', pick)
                station_id = int(re.search(r'\d+',pick_info[0]).group())
                travel_time = float(re.search(r'\d+\.+\d+',pick_info[1]).group())

                picks.loc[index] = [event_id, event_lat, event_lon, event_depth_km, station_id, travel_time, phase_type]
# print(picks)
events.to_csv('catalog.csv', index = False)
picks.to_csv('catalog_picks.csv', index=False)
# %%
