#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 13:25:19 2022

@author: huange
"""
from spynal.matIO import loadmat
import numpy as np
import pandas as pd
from preProcessing import preProcessing, featExtract
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from clustering import GMM

def select_area(unit_info, spike_times, spike_waves, area_name):
    areas = unit_info['area'].to_numpy()
    area_idx = np.where(areas == area_name)[0]
    
    spike_times = spike_times[:, area_idx]
    spike_waves = spike_waves[:, area_idx]
    return spike_times, spike_waves

def shape_data(spike_times, spike_waves):
    
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
    
    return spike_times, spike_waves

def load_data(path):
    spike_times, spike_times_schema, unit_info, trial_info, session_info, spike_waves, spike_waves_schema = \
    loadmat(path,
        variables=['spikeTimes','spikeTimesSchema','unitInfo','trialInfo', 'sessionInfo', 'spikeWaves', 'spikeWavesSchema'],
        typemap={'unitInfo':'DataFrame', 'trialInfo':'DataFrame'})\
    
    shape_spikes, shape_waves = shape_data(spike_times, spike_waves)
    
    return shape_spikes, spike_times_schema, unit_info, trial_info, session_info, shape_waves, spike_waves_schema

def concat_sessions(paths, area):
    allDF = pd.DataFrame(columns=['meanRates', 'troughToPeak', 'repolTime', 'CV', 'LV'])
    
    for path in paths:
        spike_times, spike_times_schema, unit_info, trial_info, session_info, spike_waves, spike_waves_schema = load_data(path)
        area_spike_times, area_spike_waves = select_area(unit_info, spike_times, spike_waves, area)
        
        validTrials, validNeurons, meanRates, ISIs, meanAlignWaves, smpRate, rates = preProcessing(area_spike_times, trial_info, session_info, area_spike_waves, spike_waves_schema)
        featuresDF = featExtract(meanRates, ISIs, meanAlignWaves, smpRate, rates)
        
        allDF = pd.concat([allDF, featuresDF])
    
    return allDF
        
        

def main(): 
    #path = '/Volumes/common/scott/laminarPharm/mat/Lucky-DMS_ACSF_PFC-10162020-001.mat'
    paths = ['/Volumes/common/scott/laminarPharm/mat/Lucky-DMS_ACSF_PFC-09192020-001.mat',
              '/Volumes/common/scott/laminarPharm/mat/Lucky-DMS_SALINE_PFC-07052021-001.mat',
              '/Volumes/common/scott/laminarPharm/mat/Lucky-DMS_ACSF_PFC-10162020-001.mat']
    #spike_times, spike_times_schema, unit_info, trial_info, session_info, spike_waves, spike_waves_schema = load_data(path)
    area = 'V4'
    #area_spike_times, area_spike_waves = select_area(unit_info, spike_times, spike_waves, area)
    
    allDF = concat_sessions(paths, area)
    
    all_params = ['meanRates', 'troughToPeak', 'repolTime', 'CV', 'LV']
    
    cluster_stats = allDF[all_params].to_numpy()
    scaler = StandardScaler() 
    cluster_stats_norm = scaler.fit_transform(cluster_stats)
    gmm_min, min_labels = GMM(cluster_stats_norm, 40)
    
    allDF['cluster_labels'] = min_labels
    sns.pairplot(allDF, hue = "cluster_labels", kind='scatter', 
                            diag_kind='kde')

if __name__ == "__main__":
    main()
    
    