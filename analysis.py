#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:06:39 2023

@author: ehua
"""

from spynal import spikes, info, randstats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cluster_count(labels, comp_num):
    counts = np.zeros(comp_num)
    for label in labels['0']:
        counts[label] += 1
    return counts

def pev_func(data, labels):
    pev = info.neural_info(data, labels)
    return pev

def anova(pev, labels):
    p = np.squeeze(randstats.one_way_test(pev, labels))
    #time_vec = np.linspace(-1.5, 2.5, 60)
    #plt.figure()
    #plt.plot(time_vec, p)
    return p

def ttest(data, labels):
    plt.figure()
    comp_num = max(labels)
    fig, axs = plt.subplots(comp_num, 1)
    fig.tight_layout()
    colors = ['xkcd:azure', 'mediumseagreen', 'tab:olive', 'xkcd:lavender']
    time_vec = np.linspace(-1, 0.5, 30)
    
    for i in range(comp_num):
        ax = axs[i]
        cluster_units = labels == i
        p = randstats.one_sample_test(data[cluster_units, :])
        ax.plot(time_vec, np.squeeze(p), color = colors[i])

    
def main():
    allPEVDf = pd.read_csv('/home/ehua/clustering/allPEV_samp_V4.csv', index_col = 0)
    allPEV = allPEVDf.to_numpy()
    
    labels_df = pd.read_csv('/home/ehua/clustering/V4_labels.csv', index_col = 0)
    labels = labels_df['labels']
    
    p = anova(allPEV, labels)
    print(p)    
    
    ttest(allPEV, labels)
    
    
if __name__ == "__main__":
    main()
    