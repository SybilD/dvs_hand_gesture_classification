import numpy as np
import matplotlib.pyplot as plt
import slayerSNN as snn
from dv import LegacyAedatFile
import os
from metavision_core.event_io.raw_reader import RawReader
from metavision_sdk_cv import RotateEventsAlgorithm
from metavision_sdk_core import RoiFilterAlgorithm
from metavision_core.event_io import EventsIterator, EventNpyReader
from metavision_sdk_base import EventCD
import math

def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DVS Gesture Data Processing (Cropping)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input', dest='input', default="640x480",
        help="Resolution of input image, default is the full resolution.")
    parser.add_argument(
        '-o', '--output', dest='output', default="400x300",
        help="Resolution of output image, default is 400x300. Cropping will be done about the centre of the image.")
    
    args = parser.parse_args()
    return args

args = parse_args()
inputres = args.input
outputres = args.output

if outputres == "640x480":
    w = 640
    h = 480
elif outputres == "512x384":
    w = 512
    h = 384
elif outputres == "400x300":
    w = 400
    h = 300
    
if inputres == "640x480":
    path = '/media/nurulakhira/3103-DF79/19p_dataset/' 
    full_w = 640
    full_h = 480
elif inputres == "512x384":
    path = '/media/nurulakhira/3103-DF79/19p_dataset/compressed/512x384/' 
    full_w = 512
    full_h = 384
elif inputres == "400x300":
    path = '/media/nurulakhira/3103-DF79/19p_dataset/compressed/400x300/' 
    full_w = 400
    full_h = 300


datasetfilepath = '/media/nurulakhira/3103-DF79/19p_dataset/'
activity_time_ths = 20000  # Length of the time window for activity filtering (in us)
events_buf = RotateEventsAlgorithm.get_empty_output_buffer()
x0 = int((full_w-w)/2)
x1 = x0+w
y0 = int((full_h-h)/2)
y1 = y0+h
filters = RoiFilterAlgorithm(x0,y0,x1,y1) 

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
    	    xEvent = np.append(xEvent, [e[0] for e in events])
    	    yEvent = np.append(yEvent, [e[1] for e in events])
    	    pEvent = np.append(pEvent, [e[2] for e in events])
    	    tEvent = np.append(tEvent, [e[3]/1000 for e in events])
    	#else:
    	#    print(evs)
        
    return xEvent, yEvent, pEvent, tEvent
    
def createNPY(filename, path):
    x, y, p, t = readRawEvent(path + filename + '.raw')
    
    action, tst, ten = np.loadtxt(datasetfilepath + filename + '_labels.csv', delimiter=',', skiprows=1, unpack=True) 
    print(action, tst, ten)
    
    print(actionName[int(action)-1])
    ind = (t >= tst/1000) & (t < ten/1000) 
    ind_in = np.argmax(ind)
    ind_end = ind_in+np.argmin(ind[(ind.argmax()):-1])
    if ind_end==ind_in: 
        ind_end=0
    TD = snn.io.event(x[ind_in:ind_end-1], y[ind_in:ind_end-1], p[ind_in:ind_end-1], (t[ind_in:ind_end-1] - tst/1000))
    #snn.io.showTD(TD)
    snn.io.encodeNpSpikes(datasetfilepath + 'cropped_400x300/' + inputres + '/' + filename + '.npy', TD) #create new folders first

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

                if os.path.isfile(path + filename + '.npy'):
                    print(count, filename)
                    createNPY(filename, path)
                count += 1
