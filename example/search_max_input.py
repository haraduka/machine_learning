#!/usr/bin/python
# -*- coding: utf-8 -*-

# 重みベクトルから、各labelについての最大出力を出すようなinputを求め、それらを図として出力するようなscript

import pylab as pl
import sys
from math import sqrt

filename = 'nn1_weight.dat'
n_in = 784
n_hid = 300
n_out = 10
is_bias = False

if is_bias:
    n_in += 1
    n_hid += 1

sys.stdout.write('input digit : ')
digit = int(raw_input())

inputFile = open(filename, 'r').read().split('\n')
for i in xrange(len(inputFile)):
    inputFile[i] = inputFile[i].split(' ')

print len(inputFile[0])
print len(inputFile[1])

weights_hid = [[] for i in xrange(n_in)]
weights_out = [[] for i in xrange(n_hid)]

for i in xrange(n_in):
    for j in xrange(n_hid):
        weights_hid[i].append(float(inputFile[0][i*n_hid + j]))

for i in xrange(n_hid):
    for j in xrange(n_out):
        weights_out[i].append(float(inputFile[1][i*n_out + j]))

inputs = [0 for i in xrange(n_in)]
for i in xrange(n_hid):
    for j in xrange(n_in):
        inputs[j] += weights_out[i][digit] * weights_hid[j][i]

inputs_mx = max(inputs)
inputs_normalized = inputs
for i in xrange(len(inputs)):
    inputs_normalized[i] /= inputs_mx

size = int(sqrt(n_in))
image = [[0.0 for j in xrange(size)] for i in xrange(size)]
for i in xrange(size):
    for j in xrange(size):
        image[i][j] = inputs_normalized[i*size+j]


pl.gray()
pl.matshow(image)
#pl.show()
pl.savefig(str(digit) + ".png")
