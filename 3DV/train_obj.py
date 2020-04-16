import dataset
import util
import numpy as np
import cv2
import torch
from Model_obj import OBJ_CNN
import Model_obj
import Customized_Datasets
import time
from torchvision import transforms
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # Parameters setting
    inputSize = 42
    channels =3
    trainingLimit = 30000
    trainingImages = 100
    trainingPatches = 512
    BATCHSIZE = 64
    lrInitPre = 0.0001
    storeIntervalPre = 1000
    lrInterval = 50000
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training path setting
    dataDir = './'
    trainingDir = dataDir + 'training/'
    trainingSets = util.getSubPaths(trainingDir)

    # Test path setting
    dataDir = './'
    TestDir = dataDir + 'test/'
    TestSets = util.getSubPaths(TestDir)

    # transform for data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Initialize training dataset
    training = dataset.Dataset()
    training.readFileNames(trainingSets[0])
    training.SetObjID(1)
    TrainData = Customized_Datasets.RGB_DATASET(training, inputSize, transform)

    # Initialize test dataset
    Test = dataset.Dataset()
    Test.readFileNames(TestSets[0])
    Test.SetObjID(1)
    TestData = Customized_Datasets.RGB_DATASET(Test, inputSize, transform)

    # Construction Model
    OBJ_NET = OBJ_CNN().to(DEVICE)
    # # OBJ_NET.apply(Model_obj.weight_init())

    # Training
    trainCounter = 0
    round = 0
    StoreCounter = 0
    optimizer = torch.optim.Adam(OBJ_NET.parameters(), lr=lrInitPre)

    # For recording
    trainfile = open('./Model parameter/training_loss_obj.txt', 'a')
    time_start = time.time()

    # Validation parameter
    ValiCounter = 0
    ValiLimit = 40000

    # Iteration for training
    while trainCounter <= trainingLimit:
        round += 1
        print('Starting Round:', round)
        # For training
        train_loader = torch.utils.data.DataLoader(TrainData, batch_size=BATCHSIZE, shuffle=True, num_workers=8)
        loss, trainCounter = Model_obj.train(OBJ_NET, train_loader, optimizer, trainCounter, DEVICE)
        # For testing
        # test_loader = torch.utils.data.DataLoader(TestData, batch_size=BATCHSIZE, shuffle=True, num_workers=8)
        # loss, ValiCounter = Model_obj.test(OBJ_NET, test_loader, DEVICE, ValiCounter)
        # For recording
        np.savetxt(trainfile, np.array([loss]))
        time_end = time.time()
        print('Time Cost:', time_end - time_start)
    trainfile.close()














