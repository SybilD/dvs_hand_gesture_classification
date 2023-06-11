import numpy as np
import os
from dataset import *

w = 400
h = 300
path = '/media/nurulakhira/3103-DF79/19p_dataset/rotated_400x300/'
    
def createNPY(filename, path, a):
    events = np.load(path + a + '/' + filename + '.npy')
    filtered = drop_by_time(events)
    np.save('/media/nurulakhira/3103-DF79/19p_dataset/rotated_dropbytime/' + a + '/' + filename + '.npy', filtered) 
    # create new folders first
    
if __name__ == '__main__':
    
    person = [
        #'akhira',
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
        #'rotate',
        'wave',
    ]
    
    lighting = [ 
    	#'dim',       
        #'roomlight',
        'natural',
    ]
    
    angle = ['0', '20', '40', '60', '80', '100', '120', '140', '160', '180', 'n20', 'n40', 'n60', 'n80', 'n100', 'n120', 'n140', 'n160', 'n180']

    count = 0
    
    for light in lighting: 
        for act in action:
            for name in person:
                for a in angle:
                    count += 1
                    filename = '{}_{}_{}'.format(act, name, light) 
                    if os.path.isfile(path + a + '/' + filename + '.npy'):
                         print(count, a, filename)
                         createNPY(filename, path, a)
                

