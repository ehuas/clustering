#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 22:52:30 2022

@author: huange
"""
from spynal import matIO, spikes
from spynal.matIO import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

spike_times, spike_times_schema, unit_info, trial_info, session_info = \
    loadmat('/Volumes/common/scott/laminarPharm/mat/Lucky-DMS_ACSF_PFC-10162020-001.mat',
        variables=['spikeTimes','spikeTimesSchema','unitInfo','trialInfo', 'sessionInfo'],
        typemap={'unitInfo':'DataFrame', 'trialInfo':'DataFrame'})\
            
for i in range(spike_times.shape[0]):
   for j in range(spike_times.shape[1]):
       spike_times[i,j] = np.atleast_1d(spike_times[i,j])
       
#%% 

def trialSelection(trial_info, session_info):
    '''
    Selects valid trials from data. 
        
        Input: trialInfo (n_trials, n_variables) DataFrame for single session
        Output: (n_trials,) bool vector indicating which units to keep
    
    Valid trials are defined as: 
        (a) correct and 
        (b) ≤ 5 trials before drug injection onset

    '''
    n_trials, n_variables = np.shape(trial_info)
    bool_keep = np.zeros((n_trials,))
    drugStartTrial = session_info['drugStartTrial']
    
    for trial in range(n_trials): #index backwards to see where drug_onset = False
        if trial <= drugStartTrial - 5:
            if trial_info.iloc[trial]['correct']:
                bool_keep[trial] = 1
    return bool_keep
    
def neuronSelection(spike_times):
    '''
    Selects valid neurons from data. 
        
        Input: spikeTimes (n_trials, n_units) object array for single session
        Output: (n_units,) bool vector indicating which units to
    
    Valid neurons are defined as: 
        (a) having an overall mean spike rate > 1 spike/s (weeding out silent cells) and 
        (b) having < 0.1% ISIs within 1 ms (weeding out poorly isolated single neurons)

    '''
    n_trials, n_units = np.shape(spike_times)
    bool_keep = np.zeros((n_units,))
    
    for neuron in range(n_units):
        rates, timepts = spikes.rate(spike_times[:, neuron], method='bin', lims = [-1, 2])
        ISIs = spikes.isi(spike_times[:, neuron])
        ms_ISI = np.multiply(ISIs, 1000) #converts ISI units from s to ms
        allShorts = 0
        numISIs = 0
        for trial in range(len(ms_ISI)):
            trialShorts = np.where(ms_ISI[trial] <= 1, 1, 0)
            allShorts += np.sum(trialShorts)
            numISIs += len(trialShorts)
        if np.sum(np.sum(rates, axis = 0), axis = 0) / np.size(rates) > 1:
            if allShorts/numISIs < 0.1:
                bool_keep[neuron] = 1
    return bool_keep

def preProcessing(spike_times, trial_info, session_info):
    validTrials = trialSelection(trial_info, session_info)
    validNeurons = neuronSelection(spike_times)
    
    return validTrials, validNeurons

def main():     
    
    validTrials, validNeurons = preProcessing(spike_times, trial_info, session_info)
    print('all valid trials ', validTrials)
    print('all valid neurons ', validNeurons)
    print('length of all valid neurons ', len(validNeurons))

if __name__ == "__main__":
    main()

