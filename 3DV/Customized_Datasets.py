from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import math
import cnn
import TYPE
from Hypothesis import Hypothesis
from properties import GlobalProperties


class RGB_DATASET(Dataset):
    def __init__(self, dataset, inputSize, transform):
        self.dataset = dataset
        self.inputSize = inputSize
        self.transform = transform

    def __len__(self):
        return 51200

    def __getitem__(self, item):
        imgIdx = np.random.randint(0, self.dataset.size())
        imgBGR = self.dataset.getBGR(imgIdx)
        imgObj = self.dataset.getObj(imgIdx)
        width = np.size(imgBGR, 1)
        height = np.size(imgBGR, 0)
        x = np.random.randint(self.inputSize / 2, width - self.inputSize / 2)
        y = np.random.randint(self.inputSize / 2, height - self.inputSize / 2)
        data = imgBGR[int(y - self.inputSize/2): int(y + self.inputSize/2),
               int(x - self.inputSize/2): int(x + self.inputSize/2), :]
        img = self.transform(data)
        label = imgObj[y, x]/1000.0
        return img, label


class GetCoorData(Dataset):
    def __init__(self, sampling, patchsize, colorData, transform):
        self.patchsize = patchsize
        self.sampling = sampling
        self.colorData = colorData
        self.transform = transform

    def __len__(self):
        return np.size(self.sampling, 0)**2

    def __getitem__(self, item):
        (width_samp, height_samp) = np.shape(self.sampling[:, :, 0])
        x_samp, y_samp = int(item % width_samp), int(item // width_samp)
        (origX, origY) = self.sampling[y_samp, x_samp, :]
        data = self.colorData[int(origY - self.patchsize/2):int(origY + self.patchsize/2),
               int(origX - self.patchsize/2):int(origX + self.patchsize/2), :]
        data_tensor = self.transform(data)
        return data_tensor


# Function for SCORE DATASET
def getRandHyp(gaussRot, gaussTrans):
    trans = np.array([np.random.normal(0, gaussTrans), np.random.normal(0, gaussTrans), np.random.normal(0, gaussTrans)])
    rotAxis = np.array(([np.random.rand(), np.random.rand(), np.random.rand()]))
    rotAxis = rotAxis/np.linalg.norm(rotAxis)
    rotAxis = rotAxis * np.random.normal(0, gaussRot) * math.pi/180
    # Construct rot vector and translation vector
    RotVec = np.zeros(6)
    RotVec[:3] = rotAxis
    RotVec[3:6] = trans
    # Construct a hypothesis
    h = Hypothesis()
    h.RodvecandTrans(RotVec)
    return h


class SCORE_DATASET(Dataset):
    def __init__(self, dataset, objInputSize, rgbInputSize, model, temperature, transform):
        self.dataset = dataset
        self.objInputSize = objInputSize
        self.rgbInputSize = rgbInputSize
        self.imgIdx = np.random.randint(0, dataset.size())
        self.imgBGR = dataset.getBGR(self.imgIdx)
        self.info = dataset.getInfo(self.imgIdx)
        self.model = model
        self.temperature = temperature
        self.transform = transform

    def __len__(self):
        return 1600

    def __getitem__(self, item):
        # Get parameter
        gp = GlobalProperties()
        camMat = gp.getCamMat()
        # Sampling for reprojection error image
        sampling = cnn.stochasticSubSample(self.imgBGR, targetsize=self.objInputSize, patchsize=self.rgbInputSize)
        estObj = cnn.getCoordImg(self.imgBGR, sampling, self.rgbInputSize, self.model)
        # Produce GroundTruth Label
        poseGT = Hypothesis()
        poseGT.Info(self.info)
        driftLevel = np.random.randint(0, 3)
        if not driftLevel:
            poseNoise = poseGT * getRandHyp(2, 2)
        else:
            poseNoise = poseGT * getRandHyp(10, 100)
        data = cnn.getDiffMap(TYPE.our2cv([poseNoise.getRotation(), poseNoise.getTranslation()]),
                              estObj, sampling, camMat)
        data = self.transform(data)
        label = -1 * self.temperature * max(poseGT.calcAngularDistance(poseNoise),
                                            np.linalg.norm(poseGT.getTranslation() - poseNoise.getTranslation())/10.0)
        return data, label
