# Training code 1
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

# Define dataset module
class HANDGestureDataset(Dataset):
    def __init__(self, datasetPath, sampleFile, samplingTime, sampleLength):
        self.path = datasetPath 
        self.samples = pd.read_csv(sampleFile)
        self.samplingTime = samplingTime
        self.nTimeBins    = int(sampleLength / samplingTime)

    def __getitem__(self, index):
        # Read input and label
        inputIndex = self.samples['sample'][index]
        classLabel = self.samples['labels'][index]
        # Read input spike
        inputSpikes = snn.io.readNpSpikes(
                       self.path + str(inputIndex) + '.npy'
                       ).toSpikeTensor(torch.zeros((2,300,400,self.nTimeBins)),
                       samplingTime=self.samplingTime)
        # Create one-hot encoded desired matrix
        desiredClass = torch.zeros((2, 1, 1, 1)) 
        desiredClass[classLabel,...] = 1
        
        return inputSpikes, desiredClass, classLabel

    def __len__(self):
        return self.samples.shape[0]
		
# Define the network
class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        # initialize slayer
        slayer = snn.loihi(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # define network layers
        self.conv1 = slayer.conv(2, 16, 5, padding=2, weightScale=10)
        self.conv2 = slayer.conv(16, 32, 3, padding=1, weightScale=50)
        self.pool1 = slayer.pool(4)
        self.pool2 = slayer.pool(2)
        self.pool3 = slayer.pool(2)
        self.fc1   = slayer.dense((25*19*32), 512) ##round up after division for each convolution when calculating length of flattened layer 
        self.fc2   = slayer.dense(512, 2)
        self.drop  = slayer.dropout(0.1)

    def forward(self, spikeInput):
        ##after each layer, the dimension of output is commented on the right (for input resolution of 400 x 300 
        
        spike = self.slayer.spikeLoihi(self.pool1(spikeInput)) # 100, 75, 2
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.drop(spike)
        spike = self.slayer.spikeLoihi(self.conv1(spike)) # 100, 75, 16
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.slayer.spikeLoihi(self.pool2(spike)) # 50, 38, 16
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.drop(spike)
        spike = self.slayer.spikeLoihi(self.conv2(spike)) # 50, 38, 32
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.slayer.spikeLoihi(self.pool3(spike)) #  25, 19, 32
        spike = spike.reshape((spike.shape[0], -1, 1, 1, spike.shape[-1]))
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.drop(spike)
        spike = self.slayer.spikeLoihi(self.fc1  (spike)) # 512
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.slayer.spikeLoihi(self.fc2  (spike)) # 2
        spike = self.slayer.delayShift(spike, 1)
        
        return spike
		
# Define Loihi parameter generator
def genLoihiParams(net):
    fc1Weights   = snn.utils.quantize(net.module.fc1.weight  , 2).flatten().cpu().data.numpy()
    fc2Weights   = snn.utils.quantize(net.module.fc2.weight  , 2).flatten().cpu().data.numpy()
    conv1Weights = snn.utils.quantize(net.module.conv1.weight, 2).flatten().cpu().data.numpy()
    conv2Weights = snn.utils.quantize(net.module.conv2.weight, 2).flatten().cpu().data.numpy()
    pool1Weights = snn.utils.quantize(net.module.pool1.weight, 2).flatten().cpu().data.numpy()
    pool2Weights = snn.utils.quantize(net.module.pool2.weight, 2).flatten().cpu().data.numpy()
    pool3Weights = snn.utils.quantize(net.module.pool3.weight, 2).flatten().cpu().data.numpy()
    
    ##Change folder name here to specify the file to save the resulting weights, accuracy, loss plots and state of network 
    np.save('Trained_400x300_allDA_180_bs3_LR0001_82split/fc1.npy'  , fc1Weights) 
    np.save('Trained_400x300_allDA_180_bs3_LR0001_82split/fc2.npy'  , fc2Weights)
    np.save('Trained_400x300_allDA_180_bs3_LR0001_82split/conv1.npy', conv1Weights)
    np.save('Trained_400x300_allDA_180_bs3_LR0001_82split/conv2.npy', conv2Weights)
    np.save('Trained_400x300_allDA_180_bs3_LR0001_82split/pool1.npy', pool1Weights)
    np.save('Trained_400x300_allDA_180_bs3_LR0001_82split/pool2.npy', pool2Weights)
    np.save('Trained_400x300_allDA_180_bs3_LR0001_82split/pool3.npy', pool3Weights)
    
    plt.figure(11)
    plt.hist(fc1Weights  , 256)
    plt.title('fc1 weights')
    
    plt.figure(12)
    plt.hist(fc2Weights  , 256)
    plt.title('fc2 weights')
    
    plt.figure(13)
    plt.hist(conv1Weights, 256)
    plt.title('conv1 weights')
    
    plt.figure(14)
    plt.hist(conv2Weights, 256)
    plt.title('conv2 weights')
    
    plt.figure(15)
    plt.hist(pool1Weights, 256)
    plt.title('pool1 weights')
    
    plt.figure(16)
    plt.hist(pool2Weights, 256)
    plt.title('pool2 weights')
    
    plt.figure(17)
    plt.hist(pool3Weights, 256)
    plt.title('pool3 weights')
	
if __name__ == '__main__':
    netParams = snn.params('network.yaml')
	
	  # Define the cuda device to run the code on.
    device = torch.device('cuda')
    ##Increase the number of device IDs according to the number of GPUs used e.g. [0, 1, 2, 3] for 4 GPUs
    deviceIds = [0, 1, 2]

  	# Create network instance.
    #net = Network(netParams).to(device) ##when using a single GPU 
    net = torch.nn.DataParallel(Network(netParams).to(device), device_ids=deviceIds) ##when using multiple GPUs
    #net.load_state_dict(torch.load('Trained_400x300_rotated_bs3_LR0001_82split/handGestureNet.pt')) ##uncomment to load a saved state and continue training  
  	# Create snn loss instance.
    error = snn.loss(netParams, snn.loihi).to(device)
  
  	# Define optimizer module.
  	# optimizer = torch.optim.Adam(net.parameters(), lr = 0.01, amsgrad = True) 
    optimizer = snn.utils.optim.Nadam(net.parameters(), lr = 0.001, amsgrad = True) 
  
  	# Dataset and dataLoader instances.
    ##Change datasetPath and sampleFile accordingly for trainingSet and testingSet
    trainingSet = HANDGestureDataset(datasetPath ='/mnt/beegfs/Scratch/nurul_akhira/19p_dataset/400x300/', 
  									sampleFile  ='/mnt/beegfs/Scratch/nurul_akhira/19p_dataset/400x300/train_allDA_180.txt',
  									samplingTime=netParams['simulation']['Ts'],
  									sampleLength=netParams['simulation']['tSample'])
    ##change batch_size accordingly here                                            
    trainLoader = DataLoader(dataset=trainingSet, batch_size=3, shuffle=True, num_workers=1)
  								   
    testingSet = HANDGestureDataset(datasetPath  ='/mnt/beegfs/Scratch/nurul_akhira/19p_dataset/400x300/', 
  								   sampleFile  ='/mnt/beegfs/Scratch/nurul_akhira/19p_dataset/400x300/test_allDA_180.txt',
  								   samplingTime=netParams['simulation']['Ts'],
  								   sampleLength=netParams['simulation']['tSample'])
    ##change batch_size accordingly here
    testLoader = DataLoader(dataset=testingSet, batch_size=3, shuffle=True, num_workers=1)

  	# Learning stats instance.
    stats = snn.utils.stats()
  
  	# Visualize the input spikes (first five samples).
  	#for i in range(5):
  	#	input, target, label = trainingSet[i]
  	#	snn.io.showTD(snn.io.spikeArrayToEvent(input.reshape((2, 128, 128, -1)).cpu().data.numpy()))

    ##Change number of epochs here
    for epoch in range(150): 
        tSt = datetime.now()
   
        # Training loop.
        for i, (input, target, label) in enumerate(trainLoader, 0):
      	    # Move the input and target to correct GPU.
            input  = input.to(device)
            target = target.to(device) 
            
            # Forward pass of the network.
            output = net.forward(input)
            
            # Gather the training stats.
            stats.training.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
            stats.training.numSamples     += len(label)
            
            # Calculate loss.
            loss = error.numSpikes(output, target)
            
            # Reset gradients to zero.
            optimizer.zero_grad()
            
            # Backward pass of the network.
            loss.backward()
            
            # Update weights.
            optimizer.step()
            
            # Gather training loss stats.
            stats.training.lossSum += loss.cpu().data.item()
            
            # Display training stats.
            stats.print(epoch, i, (datetime.now() - tSt).total_seconds())
    
        # Testing loop.
        for i, (input, target, label) in enumerate(testLoader, 0):
            net.eval()
            with torch.no_grad():
                input  = input.to(device)
                target = target.to(device) 
            
            output = net.forward(input)
            
            stats.testing.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
            stats.testing.numSamples     += len(label)
            
            loss = error.numSpikes(output, target)
            stats.testing.lossSum += loss.cpu().data.item()
            stats.print(epoch, i)
        
        # Update stats.
        ##Change folder name here to specify the file to save the resulting weights, accuracy, loss plots and state of network 
        stats.update()
        stats.plot(saveFig=True, path='Trained_400x300_allDA_180_bs3_LR0001_82split/')
        if stats.training.bestLoss is True:
            torch.save(net.state_dict(), 'Trained_400x300_allDA_180_bs3_LR0001_82split/handGestureNet.pt')
            print("New state saved")
    
  	# Save training data
    ##Change folder name here to specify the file to save the resulting weights, accuracy, loss plots and state of network 
    stats.save('Trained_400x300_allDA_180_bs3_LR0001_82split/')
    net.load_state_dict(torch.load('Trained_400x300_allDA_180_bs3_LR0001_82split/handGestureNet.pt'))
    genLoihiParams(net)
    
    # Plot the results.
    # Learning loss
    plt.figure(1)
    plt.semilogy(stats.training.lossLog, label='Training')
    plt.semilogy(stats.testing .lossLog, label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Learning accuracy
    plt.figure(2)
    plt.plot(stats.training.accuracyLog, label='Training')
    plt.plot(stats.testing .accuracyLog, label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()
