#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 17:13:33 2022

@author: huange
"""
from spynal import matIO, spikes, utils
from spynal.matIO import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

spike_times, spike_times_schema, unit_info, trial_info, session_info, spike_waves, spike_waves_schema = \
    loadmat('/Volumes/common/scott/laminarPharm/mat/Lucky-DMS_ACSF_PFC-10162020-001.mat',
        variables=['spikeTimes','spikeTimesSchema','unitInfo','trialInfo', 'sessionInfo', 'spikeWaves', 'spikeWavesSchema'],
        typemap={'unitInfo':'DataFrame', 'trialInfo':'DataFrame'})\
            
for i in range(spike_times.shape[0]):
   for j in range(spike_times.shape[1]):
       spike_times[i,j] = np.atleast_1d(spike_times[i,j])
       
#%% functions

def waveformFeat(spike_waves):
    troughToPeak = spikes.waveform_stats(spike_waves, stat='width')
    repolTime = spikes.waveform_stats(spike_waves, stat='repolarization')
    
    return troughToPeak, repolTime

def rateFeat(spike_times):
    meanRates = spikes.rate(spike_times)
    CV = spikes.rate_stats(spike_times, stat='CV')
    return meanRates, CV
    
def isiFeat(spike_times):
    LV = spikes.isi_stats(spike_times, stat='LV')
    return LV
    

def featExtract(spike_times, spike_times_schema, unit_info, trial_info, session_info, spike_waves, spike_waves_schema):
    

def main():     
    

if __name__ == "__main__":
    main()
