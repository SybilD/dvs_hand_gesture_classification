import numpy as np
import matplotlib.pyplot as plt
import slayerSNN as snn
from dv import LegacyAedatFile
import os
from metavision_core.event_io.raw_reader import RawReader
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from metavision_core.event_io import EventsIterator
from metavision_sdk_base import EventCD

path = '/media/nurulakhira/3103-DF79/dataset/'

activity_time_ths = 20000  # Length of the time window for activity filtering (in us)
events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
filters = ActivityNoiseFilterAlgorithm(640, 480, activity_time_ths) #keep it to full resolution

actionName = [
    'right_hand_wave', #walk
    'right_arm_clockwise', #roll
]

def readRawEvent(filename):
    iterator = EventsIterator(input_path=filename, delta_t=1000)
    xEvent = np.array([])
    yEvent = np.array([])
    pEvent = np.array([])
    tEvent = np.array([])
    
    for evs in iterator:
    	filters.process_events(evs, events_buf)
    	events = events_buf.numpy()
    	#events = evs
    	if len(events) != 0:
    	    #for  event in events:
    	    #print(evs)
    	    xEvent = np.append(xEvent, [e[0]/1.6 for e in events])
    	    yEvent = np.append(yEvent, [e[1]/1.6 for e in events])
    	    pEvent = np.append(pEvent, [e[2] for e in events])
    	    tEvent = np.append(tEvent, [e[3]/1000 for e in events])
    	#else:
    	#    print(evs)
        
    return xEvent, yEvent, pEvent, tEvent
    
def createNPY(filename, path):
    x, y, p, t = readRawEvent(path + filename + '.raw')
    
    action, tst, ten = np.loadtxt(path + filename + '_labels.csv', delimiter=',', skiprows=1, unpack=True) 
    print(action, tst, ten)
    
    print(actionName[int(action)-1])
    ind = (t >= tst/1000) & (t < ten/1000) 
    ind_in = np.argmax(ind)
    ind_end = ind_in+np.argmin(ind[(ind.argmax()):-1])
    if ind_end==ind_in: 
        ind_end=0
    TD = snn.io.event(x[ind_in:ind_end-1], y[ind_in:ind_end-1], p[ind_in:ind_end-1], (t[ind_in:ind_end-1] - tst/1000))
    #snn.io.showTD(TD)
    snn.io.encodeNpSpikes(path + 'noiseless/400x300/' + filename + '.npy', TD) #create new folders first

if __name__ == '__main__':

    person = [
        'akhira',
        'raul',
        'melvin',
        'daniela',
        'indra',
        'natasha',
        'ilyas',
        'ameer',
        'jiewei',
        'ryan',
        'cassie',
        'eliana',
        'QF',
        'eduardo',
        'azreena',
        'jin',
        'cheegan',
        'tomo',
        'dongke',
    ]
    
    action = [
        'rotate',
        'wave',
    ]
    
    lighting = [ 
    	'dim',       
        'roomlight',
        'natural',
    ]
    
    count = 0
    
    for light in lighting: 
        for act in action:
            for name in person:
                filename = '{}_{}_{}'.format(act, name, light) 

                if os.path.isfile(path + filename + '.raw'):
                    print(count, filename)
                    createNPY(filename, path)
                count += 1
