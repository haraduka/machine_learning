#!/usr/bin/python
# -*- coding: utf-8 -*-

# 8×8の数字の画像の配列をdataset.datに保存してくれるscript
# format
# datanum
# 正解LABEL
# dataのlist


from sklearn.datasets import load_digits
import pylab as pl

digits = load_digits()
outputFile = open('dataset.dat', 'w')

dataNum = 300
dataSize = 8*8

outputFile.write(str(dataNum) + '\n')

pl.gray()

for i in xrange(dataNum):
    outputFile.write(str(i%10) + '\n')
    data = digits.data[i]
    for j in xrange(len(data)):
        outputFile.write(str(data[j]/16.0) + ' ')
    outputFile.write('\n')

#pl.matshow(test)
#pl.show()
