#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:14:04 2022

@author: huange
"""

import os
os.chdir('/Users/huange/clustering')

from spynal import spikes, utils
from spynal.matIO import loadmat
from preProcessing import preProcessing, featExtract
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from statistics import mode
from sklearn.preprocessing import StandardScaler
import copy

#%%

def load_data(path):
    spike_times, spike_times_schema, unit_info, trial_info, session_info, spike_waves, spike_waves_schema = \
    loadmat(path,
        variables=['spikeTimes','spikeTimesSchema','unitInfo','trialInfo', 'sessionInfo', 'spikeWaves', 'spikeWavesSchema'],
        typemap={'unitInfo':'DataFrame', 'trialInfo':'DataFrame'})\
        
    for i in range(spike_times.shape[0]):
       for j in range(spike_times.shape[1]):
           spike = spike_times[i,j]
           wave = spike_waves[i,j]
           if type(spike) != float:
               trunc_spike = np.where((-1 < spike) & (spike < 2))[0]
               trunc_wave = wave[:, trunc_spike]
               spike_times[i,j] = np.atleast_1d(spike[trunc_spike])
               spike_waves[i,j] = np.atleast_2d(trunc_wave)
           else:
               if -1 < spike < 2:
                   spike_times[i,j] = [spike]  
                   spike_waves[i,j] = np.expand_dims(wave, 1)
               else:
                   spike_times[i,j] = []
                   spike_waves[i,j] = np.empty((48, 0))
    
    return spike_times, spike_times_schema, unit_info, trial_info, session_info, spike_waves, spike_waves_schema

from preProcessing import preProcessing, featExtract

#path = '/Volumes/common/scott/laminarPharm/mat/Lucky-DMS_ACSF_PFC-10162020-001.mat'
#path = '/Volumes/common/scott/laminarPharm/mat/Lucky-DMS_ACSF_PFC-09192020-001.mat'
path = '/Volumes/common/scott/laminarPharm/mat/Lucky-DMS_SALINE_PFC-07052021-001.mat'
spike_times, spike_times_schema, unit_info, trial_info, session_info, spike_waves, spike_waves_schema = load_data(path)
#%%
def cluster_plots(cluster_stats, all_params):
    colors = ["blue", "orange", "turquoise", "purple"]
    num_params = len(all_params)
    fig, axes = plt.subplots(3, 2, figsize = (15, 15), tight_layout = True)
    cluster_stats.boxplot(by = 'Cluster', return_type = 'axes', ax = axes)
    pass

def GMM(features, num_reps):
    components = np.arange(2, 24) # 2-9 clusters
    bics = np.zeros((num_reps, len(components)))
    aics = np.zeros((num_reps, len(components)))
    min_comps = []
    plt.figure(0)
    for rep in range(num_reps): # what is num_reps
        min_bic = np.inf 
        min_aic = np.inf
        for comp in components: # for each cluster #
            gmm = GaussianMixture(comp)
            gmm_copy = copy.deepcopy(gmm)
            gmm_copy.fit(features)
            
            bic = gmm_copy.bic(features)
            aic = gmm_copy.aic(features)
            
            bics[rep, comp-2] = bic
            aics[rep, comp-2] = aic
            
            if bic < min_bic:
                min_bic = bic
                min_comp = comp
            if aic < min_aic: 
                min_aic = aic
        min_comps.append(min_comp)
        plt.plot(components, bics[rep, :], label=str(rep), marker="o")
        plt.title('all bics, 40 reps')
    
    
    plt.figure(1)
    plt.boxplot(bics)
    plt.figure(2)
    plt.hist(min_comps)
    average_min_comp = mode(min_comps)
    print(min_comps)
    print(average_min_comp)
    bics_mean = np.mean(bics, axis=0) #average per cluster (over reps)
    aics_mean = np.mean(aics, axis=0)
    plt.figure(3)
    plt.plot(components, bics_mean)
    plt.title('bics_mean')
    plt.xlabel('cluster #')
    plt.figure(4)
    plt.plot(components, aics_mean)
    plt.title('aics_mean')
    plt.xlabel('cluster #')
# =============================================================================
#    
#     average_min_comp = mode(min_comps)
#     gmm_min = GaussianMixture(average_min_comp)
#     gmm_min.fit(features)
#     min_labels = gmm_min.predict(features)
#     return min_labels, average_min_comp
# =============================================================================

def main():    
    validTrials, validNeurons, meanRates, ISIs, meanAlignWaves, smpRate, rates = preProcessing(spike_times, trial_info, session_info, spike_waves, spike_waves_schema)
    featuresDF = featExtract(meanRates, ISIs, meanAlignWaves, smpRate, rates)
    
    all_params = ['meanRates', 'troughToPeak', 'repolTime', 'CV', 'LV']
    
    cluster_stats = featuresDF[all_params].to_numpy()
    num_params = len(all_params)
    rows, = np.shape(featuresDF['meanRates'][0])
    cluster_stats_new = np.zeros((rows, num_params))
    for i in range(num_params):
        cluster_stats_new[:, i] = cluster_stats[0, i]
    scaler = StandardScaler() #do we need this
    cluster_stats_norm = scaler.fit_transform(cluster_stats_new)
    GMM(cluster_stats_norm, 40)
    
    #featuresDF_params = featuresDF[all_params]
    #featuresDF_params['Cluster'] = labels
    #cluster_plots(featuresDF_params, all_params)

if __name__ == "__main__":
    main()
    