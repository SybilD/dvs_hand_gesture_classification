import numpy as np
import matplotlib.pyplot as plt
import slayerSNN as snn
from dv import LegacyAedatFile
import os
from metavision_core.event_io.raw_reader import RawReader
from metavision_sdk_cv import RotateEventsAlgorithm
from metavision_core.event_io import EventsIterator
from metavision_sdk_base import EventCD
import math

def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DVS Gesture Data Processing (Rotation)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-a', '--angle', type=int, default=20,
        help="Input angle to rotate image by, default is 20")
    
    args = parser.parse_args()
    return args

args = parse_args()
angle = args.angle

if angle == 0:
    rotation = 0
else:
    rotation = math.pi/(180/angle)

if angle < 0:
    folder = 'n' + str(-angle)
else:
    folder = str(angle)
    
path = '/media/nurulakhira/3103-DF79/19p_dataset/original/'

events_buf = RotateEventsAlgorithm.get_empty_output_buffer()
filters = RotateEventsAlgorithm(640, 480, rotation) 

actionName = [
    'right_hand_wave', #walk
    'right_arm_clockwise', #roll
]


def rotate(evs):
    
    filters.process_events(evs, events_buf)
    events = events_buf.numpy()
    #events = evs
    xEvent = np.array([])
    yEvent = np.array([])
    pEvent = np.array([])
    tEvent = np.array([])
    #print(events)

    if len(events) != 0:
        xEvent = np.append(xEvent, [e[0]/1.6 for e in events])
        yEvent = np.append(yEvent, [e[1]/1.6 for e in events])
        pEvent = np.append(pEvent, [e[2] for e in events])
        tEvent = np.append(tEvent, [e[3]/1000 for e in events])
        
    return xEvent, yEvent, pEvent, tEvent
    
def createNPY(iterator):
    
    xEventfull = np.array([])
    yEventfull = np.array([])
    pEventfull = np.array([])
    tEventfull = np.array([])
    time = 0
    
    for evs in iterator:
        x, y, p, t = rotate(evs)
        xEventfull = np.append(xEventfull, x)
        yEventfull = np.append(yEventfull, y)
        pEventfull = np.append(pEventfull, p)
        tEventfull = np.append(tEventfull, t)
        time += 1
        if time > 1500:
            break
    
    TD = snn.io.event(xEventfull, yEventfull, pEventfull, tEventfull)
    npEvent = np.zeros((len(TD.x), 4))
    npEvent[:, 0] = TD.x
    npEvent[:, 1] = TD.y
    npEvent[:, 2] = TD.p
    npEvent[:, 3] = TD.t

    snn.io.encodeNpSpikes('/media/nurulakhira/3103-DF79/19p_dataset/rotated_400x300/' + folder + '/' + filename + '.npy', TD)
    #snn.io.showTD(TD)
     #create new folders first

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
                    print(count, angle, filename)
                    file_path = path + filename + '.raw'
                    iterator = EventsIterator(input_path=file_path, delta_t=1000)
                    createNPY(iterator)
                count += 1
