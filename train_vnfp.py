import chainer
import chainer.functions as F
import chainer.iterators as I
import chainer.links as L
import chainer.optimizers as O
from chainer import training
import chainer.training.extensions as E
from chainer import serializers
import pickle

import numpy as np
import time
import math
import sys
import argparse
import copy
import os
import six


import model2
import load_data_ecfp

def cal_auc_score(predicted, y):
    thresh = np.linspace(1, 0, 101)
    ROC = np.zeros((101, 2))
    for i in range(101):
        t = float(thresh[i])
        TP_t = np.logical_and(predicted > t, y == 1).sum()
        TN_t = np.logical_and(predicted <= t, y == 0).sum()
        FP_t = np.logical_and(predicted > t, y == 0).sum()
        FN_t = np.logical_and(predicted <= t, y == 1).sum()

        FPR_t = FP_t / float(FP_t + TN_t)
        ROC[i, 0] = FPR_t

        TPR_t = TP_t / float(TP_t + FN_t)
        ROC[i, 1] = TPR_t

    AUC = 0.
    for i in range(100):
        AUC += (ROC[i + 1, 0] - ROC[i, 0]) * (ROC[i + 1, 1] + ROC[i, 1])
    AUC *= 0.5
    return AUC


C = 12
nfp_hidden_dim = 100
nfp_out_dim = 100
max_degree = 5
radius = 6
mlp_hid_dim = 100
batchsize = 200
n_epoch = 40

train, val, test, atom2id = load_data_ecfp.make_dataset()
nfp = model2.NFP(nfp_hidden_dim, nfp_out_dim, max_degree, len(atom2id), radius)
#model =  model2.Predictor(nfp, mlp_hid_dim, C)
model = model2.VPredictor(nfp, mlp_hid_dim, C)
#model = L.Classifier(predictor,
#                     lossfun=F.sigmoid_cross_entropy,
#                     accfun=F.binary_accuracy)

optimizer = O.Adam()
optimizer.setup(model)
N = len(train)
N_val = len(val)

recog_valid_loss = []
recog_train_loss = []
kl_valid_loss = []
kl_train_loss = []
valid_loss = []
train_loss = []
for epoch in range(1, n_epoch + 1):
    print('epoch:', epoch)

    model.setMode(True)
    indexes = np.random.permutation(len(train))

    sum_loss = 0
    sum_rec_loss = 0
    sum_kl_loss = 0
    sum_accuracy = 0
    start = time.time()
    for i in range(0, len(train), batchsize):
        maxid = min(i + batchsize, N)
        batch = train[indexes[i:maxid]]

        adjs = [d[0] for d in train[indexes[i : maxid]]]
        atom_array = [d[1] for d in train[indexes[i: maxid]]]
        labels = [d[2] for d in train[indexes[i: maxid]]]
        adjs = np.asarray(adjs)
        atom_array = np.asarray(atom_array)
        labels = np.asarray(labels)

        #model(adjs, atom_array, labels)
        optimizer.update(model, adjs, atom_array, labels)

        sum_rec_loss += float(model.rec_loss.data) * len(labels.data)
        sum_kl_loss += float(model.kl_dv.data) * len(labels.data)
        sum_loss += float(model.loss.data) * len(labels.data)
        sum_accuracy += float(model.accuracy.data) * len(labels.data)


    end = time.time()
    elapsed = end - start
    throughput = N / elapsed
    print('rec loss={}, Kl loss={}, acc = {}'.format(
        sum_rec_loss / N, sum_kl_loss / N , sum_accuracy / N))

    recog_train_loss.append(sum_rec_loss/N)
    kl_train_loss.append(sum_kl_loss/N)
    train_loss.append(sum_loss/N)



    # evalluation
    sum_accuracy = 0
    sum_rec_loss = 0
    sum_kl_loss = 0
    sum_loss = 0
    model.setMode(False)
    for i in six.moves.range(0, N_val, batchsize):
        maxid = min(i + batchsize, N_val)
        index = np.asarray(list(range(i, maxid)))
        adjs = [d[0] for d in val[index]]
        atom_array = [d[1] for d in val[index]]
        labels = [d[2] for d in val[index]]
        adjs = np.asarray(adjs)
        atom_array = np.asarray(atom_array)
        labels = np.asarray(labels)


        loss = model(adjs, atom_array, labels)
        sum_rec_loss += float(model.rec_loss.data) * len(labels.data)
        sum_kl_loss += float(model.kl_dv.data) * len(labels.data)
        sum_loss += float(model.loss.data) * len(labels.data)
        sum_accuracy += float(model.accuracy.data) * len(labels.data)

    print('val loss = {}, accuracy = {}'.format(sum_loss/N_val, sum_accuracy/N_val))

    # save valid losses
    recog_valid_loss.append(sum_rec_loss/N)
    kl_valid_loss.append(sum_kl_loss/N)
    valid_loss.append(sum_loss/N)

print('save the model')
serializers.save_npz('model.model', model)

#test
adjs = [d[0] for d in test]
atom_array = [d[1] for d in test]
labels = [d[2] for d in test]
adjs = np.asarray(adjs)
atom_array = np.asarray(atom_array)
labels = np.asarray(labels)


test_loss = model(adjs, atom_array, labels)
rec_test_loss = float(model.rec_loss.data)
kl_test_loss = float(model.kl_dv.data)
accuracy = float(model.accuracy.data)

print("rec_loss = ", rec_test_loss)
print("kl_loss=", kl_test_loss)
print("acc = ", accuracy)



#save for illustration
# for training
with open("variational_train_2x100nfp_3x100mlp_rec", "wb") as fp:
    pickle.dump(recog_train_loss, fp)
with open("variational_train_2x100nfp_3x100mlp_kl", "wb") as fp:
    pickle.dump(kl_train_loss, fp)
with open("variational_train_2x100nfp_3x100mlp_loss", "wb") as fp:
    pickle.dump(train_loss, fp)

# for valid
with open("variational_valid_2x100nfp_3x100mlp_rec", "wb") as fp:
    pickle.dump(recog_valid_loss, fp)
with open("variational_valid_2x100nfp_3x100mlp_kl", "wb") as fp:
    pickle.dump(kl_valid_loss, fp)
with open("variational_valid_2x100nfp_3x100mlp_loss", "wb") as fp:
    pickle.dump(train_loss, fp)

# for test
with open("variational_test_2x100nfp_3x100mlp_rec", "wb") as fp:
    pickle.dump(rec_test_loss, fp)
with open("variational_test_2x100nfp_3x100mlp_kl", "wb") as fp:
    pickle.dump(rec_test_loss, fp)
with open("variational_test_2x100nfp_3x100mlp_acc", "wb") as fp:
    pickle.dump(accuracy, fp)






# auc score for each task
label_names = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
               'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
               'SR-HSE', 'SR-MMP', 'SR-p53']


filename = 'tox21_10k_challenge_test.sdf'
model.setMode(False)
for task in label_names:
    dataset = load_data_ecfp.load_one_task(task, filename, atom2id)
    print(len(dataset))
    adjs = [d[0] for d in dataset]
    atom_array = [d[1] for d in dataset]
    labels = [d[2] for d in dataset]
    adjs = np.asarray(adjs)
    atom_array = np.asarray(atom_array)
    labels = np.asarray(labels)
    predicted = model.predict(adjs, atom_array)
    print(predicted.data)
    score = cal_auc_score(predicted.data, labels)
    print('Task:', task, "  AUC score:", score)








