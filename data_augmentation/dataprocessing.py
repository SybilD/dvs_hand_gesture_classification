import numpy as np
import matplotlib.pyplot as plt
import slayerSNN as snn
from dv import LegacyAedatFile
import os
from metavision_core.event_io.raw_reader import RawReader
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from metavision_core.event_io import EventsIterator
from metavision_sdk_base import EventCD

path = '/media/nurulakhira/3103-DF79/new dataset/'

activity_time_ths = 20000  # Length of the time window for activity filtering (in us)
events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
filters = ActivityNoiseFilterAlgorithm(640, 480, activity_time_ths) #keep it to full resolution

actionName = [
    'right_hand_wave', #walk
    'right_arm_clockwise', #roll
]

def readRawEvent(filename):
    record_raw = RawReader(filename)
    #iterator = EventsIterator(input_path=filename, delta_t=1000)
    events_by_time = record_raw.load_delta_t(1500000)
    xEvent = []
    yEvent = []
    pEvent = []
    tEvent = []
    
    for event in events_by_time:
    #for event in iterator:
        #filters.process_events(event, events_buf)
        #events = events_buf.numpy()
        if len(event) != 0:
        #if (event[0]/5) % 1 == 0 and (event[1]/4) % 1 == 0:
            #xEvent.append([e[0]/1.6 for e in event])
            #yEvent.append([e[1]/1.6 for e in event])
            #pEvent.append([e[2] for e in event])
            #tEvent.append([e[3]/1000 for e in event])
            xEvent.append(event[0]/1.6)
            yEvent.append(event[1]/1.6)
            pEvent.append(event[2])
            tEvent.append(event[3]/1000)

    return xEvent, yEvent, pEvent, tEvent
    
def createNPY(filename, path):
    x, y, p, t = readRawEvent(path + filename + '.raw')
    #action, tst, ten = np.loadtxt(path + filename + '_labels.csv', delimiter=',', skiprows=1, unpack=True) 
    #print(action, tst, ten)
    
    #print(actionName[int(action)-1])
    #print(t)
    tst = np.array([0])
    ten = np.array([1500000])
    ind = (t >= tst/1000) & (t < ten/1000) 
    ind_in = np.argmax(ind)
    ind_end = ind_in+np.argmin(ind[(ind.argmax()):-1])
    if ind_end==ind_in: 
           ind_end=0
    TD = snn.io.event(x[ind_in:ind_end-1], y[ind_in:ind_end-1], p[ind_in:ind_end-1], (t[ind_in:ind_end-1] - tst/1000))
    #snn.io.showTD(TD)

    snn.io.encodeNpSpikes(path + '/npy/' + filename + '.npy', TD)

if __name__ == '__main__':

    person = list(range(1, 13))
    person.remove(5)
    
    action = ['w', 'r'] #wave, rotate
    
    lighting = ['r', 'd'] #roomlight, dim
    
    states = ['s', 'w', 'o', 'r'] #standing, walking, staionary roll form, rolling
    
    distance = [1, 2, 3] #1m, 1.5m, 2m
    
    count = 0
    
    for light in lighting: 
        for act in action:
            for name in person:
                for state in states:
                    for d in distance:
                        filename = '{}{}{}{}_{}'.format(act, state, light, str(d), str(name)) 

                        if os.path.isfile(path + filename + '.raw'):
                            print(count, filename)
                            createNPY(filename, path)
                        count += 1
    
    
    
    
    
    
    
    
