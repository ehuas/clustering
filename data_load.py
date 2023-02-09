#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 13:25:19 2022

@author: huange
"""
from spynal.matIO import loadmat
from spynal import utils
import numpy as np
import pandas as pd
from preProcessing import preProcessing, featExtract
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from clustering import GMM, pairplot, feat_reduction
import os

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
    allAlignWaves = np.empty((470, 0))
    
    for path in paths:
        spike_times, spike_times_schema, unit_info, trial_info, session_info, spike_waves, spike_waves_schema = load_data(path)
        area_spike_times, area_spike_waves = select_area(unit_info, spike_times, spike_waves, area)
        
        validTrials, validNeurons, meanRates, ISIs, meanAlignWaves, smpRate, rates, alignWaves = preProcessing(area_spike_times, trial_info, session_info, area_spike_waves, spike_waves_schema)
        #validTrials, validNeurons, meanRates, ISIs, meanAlignWaves, smpRate, rates, alignWaves = preProcessing(spike_times, trial_info, session_info, spike_waves, spike_waves_schema)
        print(smpRate)
        if validTrials.size < 2 or validNeurons.size < 2:
            pass
        else:
            featuresDF = featExtract(meanRates, ISIs, meanAlignWaves, smpRate, rates)
            allDF = pd.concat([allDF, featuresDF], ignore_index=True)
            allAlignWaves = np.concatenate((allAlignWaves, meanAlignWaves), axis = 1)
    
    return allDF, meanAlignWaves, allAlignWaves
        
def save_df(df):
    os.makedirs('/home/ehua/clustering', exist_ok=True)
    df.to_csv('/home/ehua/clustering/dlPFC_df.csv')
        
def main(): 
    #path = '/Volumes/common/scott/laminarPharm/mat/Lucky-DMS_ACSF_PFC-10162020-001.mat'
    directory = '/mnt/common/scott/laminarPharm/mat'
    paths = []
    for filename in os.listdir(directory):
        if filename == 'laminarPharm_databases.mat':
            pass
        else:
            f = os.path.join(directory, filename)
            paths.append(f)
    area = 'dlPFC'
    
    allDF, meanAlignWaves, allAlignWaves = concat_sessions(paths, area)
    save_df(allDF)
    allAlignWavesDF = pd.DataFrame(allAlignWaves)
    allAlignWavesDF.to_csv('/home/ehua/clustering/allAlignWaves_dlPFC.csv')
    
    # all_params = ['meanRates', 'troughToPeak', 'repolTime', 'CV', 'LV']
   
    # cluster_stats = allDF[all_params].to_numpy()
    # scaler = StandardScaler() 
    # cluster_stats_norm = scaler.fit_transform(cluster_stats)
    
    # pairplot(allDF, cluster_stats_norm)
    
    # feat_reduction(allDF)
    


if __name__ == "__main__":
    main()
    
    