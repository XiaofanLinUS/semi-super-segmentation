import pydicom
# read dicom file
import glob
# retrive filename from a directory
import numpy as np
# numpy matrices
import os
# deal with file and directory stuffs
import operator
# for dictionary sorting

import matplotlib.pyplot as plt

'''
Given the root directory of dicom files, generate a file name
sorted in instance number 

input: rootDir -- target directory where it holds dicom files
output: sortedDic -- an array of file names where they are sorted according to the instance number 

'''
def getSortedFilenames(rootDir):
    dicomDict = {}
    for filename in glob.glob(rootDir+'/*.dcm'):
        rawDicom = pydicom.dcmread(filename)
        dicomDict[filename] = rawDicom.InstanceNumber
    dicomDict = sorted(dicomDict.items(), key=operator.itemgetter(1))
    print(dicomDict)
    sortedFilenames = []
    for k, v in dicomDict:
        sortedFilenames.append(k)

    return sortedFilenames

'''
convert raw dicom file to np object

input: rootDir -- target directory where it holds dicom files
ouput: rawImgs -- a numpy array of size N x W X H with int 16 type
'''
def preprocessDicom(rootDir):
    rawDicoms = []
    rawImgs = []
    for filename in glob.glob(rootDir+'/*.dcm'):
        rawDicom = pydicom.dcmread(filename)
        rawDicoms.append(rawDicom)
    rawDicoms = sorted(rawDicoms, key=lambda rawDicom: rawDicom.InstanceNumber)
    for rawDicom in rawDicoms:
        rawImgs.append(rawDicom.pixel_array)
    rawImgs_int16 = np.array(rawImgs)
    return rawImgs_int16

'''
Given the root directory of an mri case scaned, we load the image paired with label data
If no label is presented, it will return a tuple with second value set to none

input: rootDir -- target directory where it hold a mri case 
output: imgLabelPair -- a tuple of numpy images and its mask
'''
def loadImgPairs(rootDir):
    hasLabel = os.path.isdir(rootDir+"/label")
    rawImgs = preprocessDicom(rootDir+"/raw")
    if hasLabel:
        masks = preprocessDicom(rootDir+"/label")    
    else:
        masks = None

    return (rawImgs, masks)

'''
Load all mri cases from a directory

input: rootDir -- target directory holding all mri data labeled or not
output: thunksPair -- a pair of labeled and unlabel numpy matrix holding all mri data
'''
def loadThunks(rootDir):
    labeledDir = rootDir + '/labeled'
    unlabeledDir = rootDir + '/unlabeled'

    labeledDir_code = os.fsencode(labeledDir)
    unlabeledDir_code = os.fsencode(unlabeledDir)

    labelThunks = []
    unlabeledThunks = []

    for subdir in os.listdir(labeledDir_code):
        subdir = os.fsdecode(subdir)
        labelThunks.append(loadImgPairs(labeledDir + '/' + subdir))

    for subdir in os.listdir(unlabeledDir_code):
        subdir = os.fsdecode(subdir)
        unlabeledThunks.append(loadImgPairs(unlabeledDir + '/' + subdir))

    return (labelThunks, unlabeledThunks)