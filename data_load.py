#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 13:25:19 2022

@author: huange
"""
from spynal.matIO import loadmat
import numpy as np

def select_area(unit_info, spike_times, spike_waves, area_name):
    areas = unit_info['area'].to_numpy()
    area_idx = np.where(areas == area_name)[0]
    
    spike_times = spike_times[:, area_idx]
    spike_waves = spike_waves[:, area_idx]
    return spike_times, spike_waves

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

def concat_sessions(paths):
    for path in paths:
        spike_times, spike_times_schema, unit_info, trial_info, session_info, spike_waves, spike_waves_schema = load_data(path)
        

def main(): 
    path = '/Volumes/common/scott/laminarPharm/mat/Lucky-DMS_ACSF_PFC-10162020-001.mat'
    #path = '/Volumes/common/scott/laminarPharm/mat/Lucky-DMS_ACSF_PFC-09192020-001.mat'
    #path = '/Volumes/common/scott/laminarPharm/mat/Lucky-DMS_SALINE_PFC-07052021-001.mat'
    spike_times, spike_times_schema, unit_info, trial_info, session_info, spike_waves, spike_waves_schema = load_data(path)
    area = 'V4'
    area_spike_times, area_spike_waves = select_area(unit_info, spike_times, spike_waves, area)
    
if __name__ == "__main__":
    main()
    
    