import numpy as np
import os

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
    path = '/media/nurulakhira/3103-DF79/19p_dataset/rotated_640x480/' 
    full_w = 640
    full_h = 480
elif inputres == "512x384":
    path = '/media/nurulakhira/3103-DF79/19p_dataset/rotated_512x384/' 
    full_w = 512
    full_h = 384
elif inputres == "400x300":
    path = '/media/nurulakhira/3103-DF79/19p_dataset/compressed/400x300/' 
    full_w = 400
    full_h = 300


#datasetfilepath = '/media/nurulakhira/3103-DF79/19p_dataset/'
x0 = int((full_w-w)/2)
x1 = x0+w
y0 = int((full_h-h)/2)
y1 = y0+h
    
def createNPY(filename, path):
    for a in angle:
        if os.path.isfile(path + a + '/' + filename + '.npy'):
            print(a, filename)
            events = np.load(path + a + '/' + filename + '.npy')
            crop = np.array([[0,0,0,0]])
            #loop that is very slow
            for evs in events:
                if evs[0]<x1 and evs[0]>x0 and evs[1]<y1 and evs[1]>y0:
                    crop = np.insert(crop, len(crop), [evs], axis=0)
    
            np.save(path + a + '/' + filename + '.npy', crop)
    
if __name__ == '__main__':

    person = [
        #'akhira',
        #'raul',
        'melvin',
        'daniela',
        'indra',
        'natasha',
        'ilyas',
        #'ameer',
        #'jiewei',
        #'ryan',
        #'cassie',
        #'eliana',
        #'QF',
        #'eduardo',
        #'azreena',
        #'jin',
        #'cheegan',
        #'tomo',
        #'dongke',
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
    
    count = 0
    
    
    
    for light in lighting: 
        for act in action:
            for name in person:
                if name == 'melvin':
                    angle = ['80', '100', '120', '140', '160', '180', 'n20', 'n40', 'n60', 'n80', 'n100', 'n120', 'n140', 'n160', 'n180']
                else:
                    angle = ['0', '20', '40', '60', '80', '100', '120', '140', '160', '180', 'n20', 'n40', 'n60', 'n80', 'n100', 'n120', 'n140', 'n160', 'n180']
     
                #count += 1
                #if count > 96:
                filename = '{}_{}_{}'.format(act, name, light) 
                #print(count, filename)
                createNPY(filename, path)
                

