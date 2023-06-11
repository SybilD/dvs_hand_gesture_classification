# Inference code
import sys, os
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../../src")

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
import pandas as pd
import math

from dv import LegacyAedatFile


# Define the network
class Network(torch.nn.Module):
    def __init__(self, netParams, d):
        super(Network, self).__init__()
        # initialize slayer
        slayer = snn.loihi(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # define network functions
        self.conv1 = slayer.conv(2, 16, 5, padding=2, weightScale=10)
        self.conv2 = slayer.conv(16, 32, 3, padding=1, weightScale=50)
        self.pool1 = slayer.pool(4)
        self.pool2 = slayer.pool(2)
        self.pool3 = slayer.pool(2)
        a = math.ceil(40/1.6) ## 1.6 is the denominator I divide the original resolution by, to decrease to a smaller resolution
        b = math.ceil(30/1.6) ## The dimension of output before flattening for the original resolution is 40x30x32. So for smaller resolutions, we can just scale 40 and 30 by 1/1.6
        self.fc1   = slayer.dense((a*b*32), 512) 
        self.fc2   = slayer.dense(512, 2)
        self.drop  = slayer.dropout(0.1)

    ##Dimension of spikes after layer is given on the right
    def forward(self, spikeInput):
        spike = self.slayer.spikeLoihi(self.pool1(spikeInput )) # 100, 75, 2
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.drop(spike)
        spike = self.slayer.spikeLoihi(self.conv1(spike)) # 100, 75, 16
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.slayer.spikeLoihi(self.pool2(spike)) # 50, 38, 16
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.drop(spike)
        spike = self.slayer.spikeLoihi(self.conv2(spike)) # 50, 38, 32
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.slayer.spikeLoihi(self.pool3(spike)) # 25, 19, 32
        spike = spike.reshape((spike.shape[0], -1, 1, 1, spike.shape[-1]))
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.drop(spike)
        spike = self.slayer.spikeLoihi(self.fc1  (spike)) # 512
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.slayer.spikeLoihi(self.fc2  (spike)) # 2
        spike = self.slayer.delayShift(spike, 1)
        
        return spike

##Define the arguments to include when running the inference code
##To run the code: python inference.py -s 'Trained_400x300_bs3_LR001_82split'
def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DVS Gesture Inference Code.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ##Specify folder where the state is saved in 
    parser.add_argument(
        '-s', '--state', dest='state', default='Trained_400x300_bs3_LR001_82split',
        help="Path to ibmGestureNet.pt, if not specified, Trained_400x300_bs3_LR001_82split will be used.")
    
    args = parser.parse_args()
    return args
 
##Format the raw spikes accordingly for SNN input    
class dataset(Dataset):
    
    def __init__(self, iterator):
        self.event_array = iterator

    def __getitem__(self, index):
        # Read input and label
        classLabel = 1 ##dummy classLabel, classLabel is not required when conducting inference 
        # Read input spike
        inputSpikes = snn.spikeFileIO.event(self.event_array[:, 0], self.event_array[:, 1], self.event_array[:, 2], self.event_array[:, 3])
        x = int(640/1.6) ##to get the correct input resolution
        y = int(480/1.6)
        spiketensor = inputSpikes.toSpikeTensor(torch.zeros((2,y,x,1450)), samplingTime=1.0) ##tensor shape: (channels, height, width, number of time bins)
        desiredClass = torch.zeros((2, 1, 1, 1))
        desiredClass[classLabel,...] = 1
        
        return spiketensor, desiredClass, classLabel
        
    def __len__(self):
        return 1 ##length of sample size (not important) 

##Function to run the classification process and get the predicted class for 1 sample    
def classify(device, net, npyfile, error):
    
    mv_iterator = np.load(npyfile)

    test = dataset(mv_iterator)
    testLoader = DataLoader(dataset=test, batch_size=1, num_workers=1)
    for input, target, label in testLoader:
        net.eval()
        with torch.no_grad():
            input = input.to(device)
            target = target.to(device)
	
        output = net.forward(input)
        predicted = snn.predict.getClass(output)  
        loss = error.numSpikes(output, target)
        #print('Loss:', loss.cpu().data.item())
        #print('Predicted class:', int(predicted))
    
    return int(predicted)

def main():
    
    ##Retrieve the arguments and set network parameters
    args = parse_args()
    state_path = '/home/users/nurul_akhira/slayerPytorch/exampleLoihi/03_IBMGesture/' + args.state + '/ibmGestureNet.pt'
    netParams = snn.params('/home/users/nurul_akhira/slayerPytorch/exampleLoihi/03_IBMGesture/network.yaml')

    # Define the cuda device to run the code on.
    device = torch.device('cuda')
    deviceIds = [0] 

    # Create network instance and load the saved state
    #net = Network(netParams, d).to(device)
    net = torch.nn.DataParallel(Network(netParams, d).to(device), device_ids=deviceIds) #works for single GPU also 
    net.load_state_dict(torch.load(state_path))
    error = snn.loss(netParams, snn.loihi).to(device)
    
    ##Path to samples to classify 
    path = "/mnt/beegfs/Scratch/nurul_akhira/19p_dataset/400x300/"
    
    ##Classifying only the test samples 
    test = ['ryan',
        'QF',
        'tomo',
        'raul',
        ]
        
    action = [
        'wave',
        'rotate',
        ]
    
    lighting = [        
        'roomlight',
        'dim',
        'natural',
        ]
    
    ##Uncomment only original if classifying only the orignal samples
    ##Uncomment the specific data augmentation if you wish to classify the samples with that data augmentation
    data_augmentation = ['original',
                     #'eventdrop/randomdrop', 
                     #'eventdrop/dropbyarea', 
                     #'eventdrop/dropbytime', 
                     #'flip/xflip', 
                     #'flip/yflip', 
                     #'rotated',
                     #'xyshift',
                     #'cropped/512x384',
                     #'cropped/640x480'
                     ]
    
    angles = np.arange(-45, 50, 5)
    #angles = np.arange(-180, 200, 20)
    angles = np.delete(angles, 9) #delete zero
    angles_str = []

    for a in angles:
        if a < 0:
            positive = -a
            angles_str.append('n{}'.format(positive))
        else:
            angles_str.append(str(a))
        
    angle_list = angles_str ##for when you wanna run the inference test for all angles for each sample
    ##Uncomment below for when you want to run the inference test for a randomised angle for each sample
    #angle_list = np.array([])  
    #for c in range(2):
    #    angle_list = np.append(angle_list, angles_str)
    
    print(args.state)
    
    i = 0
    #Hand wave classification
    print('Hand wave classification:')
    for light in lighting:
        correct = 0 ##to count the number of correctly classified samples
        for DA in data_augmentation:
            if DA == 'rotated':
                for a in angle_list: ## comment this line out if using random angles
                    for name in test:
                        #filename = 'rotated/{}/wave_{}_{}.npy'.format(angle_list[i], name, light) ## uncomment this line if using random angles
                        filename = 'rotated/{}/wave_{}_{}.npy'.format(a, name, light) ## comment this line out if using random angles
                        i += 1
                        raw_path = path + filename
                        if classify(device, net, raw_path, error) == 0:
                            correct += 1
                        #else:
                        #    print(filename) ##uncomment else line if you want to know which file is incorrectly classified
            else:            
                for name in test:
                    filename = '{}/wave_{}_{}.npy'.format(DA, name, light) 
                    raw_path = path + filename
                    if classify(device, net, raw_path, error) == 0:
                        correct += 1
                    #else:
                    #    print(filename)
        
        print(light + ':') ##printing out total number of correctly classified samples for each lighting condition
        print(correct)
            
       
    #Hand rotate classification
    ##Same format as the above
    print('Hand rotate classification:')   
    for light in lighting:
        correct = 0
        for DA in data_augmentation:
            if DA == 'rotated':
                for a in angle_list: 
                    for name in test:
                        #filename = 'rotated/{}/rotate_{}_{}.npy'.format(angle_list[i], name, light)
                        filename = 'rotated/{}/rotate_{}_{}.npy'.format(a, name, light)
                        i += 1
                        raw_path = path + filename
                        if classify(device, net, raw_path, error) == 1: #1 for hand rotation, 0 for hand wave
                            correct += 1
                        #else:
                        #    print(filename)
            else:            
                for name in test:
                    filename = '{}/rotate_{}_{}.npy'.format(DA, name, light) 
                    raw_path = path + filename
                    if classify(device, net, raw_path, error) == 1:
                        correct += 1
                    #else:
                    #    print(filename)
                    
        print(light + ':')
        print(correct)
    
if __name__ == "__main__":
    main()
        
    
        

    

       
