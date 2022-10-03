#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 22:52:30 2022

@author: huange
"""
#%% test



#%% actual code
from spynal import matIO, spikes
from spynal.matIO import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    bool_keep = np.zeros((n_trials,))
    
    for neuron in range(n_units):
        rates, timepts = spikes.rate(spike_times[:, neuron], method='bin', lims = [-1, 2])
        ISIs = spikes.isi(spike_times[:, neuron])
        ms_ISI = np.multiply(ISIs, 1000) #converts ISI units from s to ms
        #rows = np.shape(ms_ISI)
        #shortISIs = np.zeros(rows,)
        allShorts = 0
        numpts = 0
        for trial in range(len(ms_ISI)):
            trialShorts = np.where(ms_ISI[trial] <= 1, 1, 0)
            #shortISIs[trial] = trialShorts
            allShorts += np.sum(trialShorts)
            numpts += len(trialShorts)
        if np.sum(np.sum(rates, axis = 0), axis = 0) / np.size(rates) > 1:
            if allShorts/numpts < 0.1:
                bool_keep[neuron] = 1
    return bool_keep

def preProcessing(spike_times, trial_info, session_info):
    validTrials = trialSelection(trial_info, session_info)
    validNeurons = neuronSelection(spike_times)
    
    return validTrials, validNeurons

def main():     
    spike_times, spike_times_schema, unit_info, trial_info, session_info = \
        loadmat('/Volumes/common/scott/laminarPharm/mat/Lucky-DMS_ACSF_PFC-10162020-001.mat',
            variables=['spikeTimes','spikeTimesSchema','unitInfo','trialInfo', 'sessionInfo'],
            typemap={'unitInfo':'DataFrame', 'trialInfo':'DataFrame'})\
            
    for i in range(spike_times.shape[0]):
        for j in range(spike_times.shape[1]):
            spike_times[i,j] = np.atleast_1d(spike_times[i,j])
    
    validTrials, validNeurons = preProcessing(spike_times, trial_info, session_info)
    print('all valid trials ', validTrials)
    print('all valid neurons ', validNeurons)

if __name__ == "__main__":
    main()

#%%
def troughToPeak(waveForm, samplerate):
    trough = np.argmin(waveForm)
    peak = trough + np.argmax(waveForm[trough:])
    return (peak - trough)/samplerate

def repolTime(waveForm, samplerate):
    peak = np.argmin(waveForm) + np.argmax(waveForm[np.argmin(waveForm):])
    infls = np.where(np.diff(np.sign(np.gradient(np.gradient(waveForm)))))[0]
    repols = infls[infls > peak]
    repol = peak
    if len(repols) > 0:
        repol = repols[0]
    return (repol-peak)/samplerate

def Lv(isiList):
    transformedISIs = []
    for trialIsis in isiList:
         transformedISIs = transformedISIs + [((isi_prev - isi_next)/(isi_prev + isi_next))**2
                       for (isi_prev, isi_next) in zip(trialIsis[:-1], trialIsis[1:])]
    return 3/(len(transformedISIs))*sum(transformedISIs)

def Cv(isiList):
    flattened_isis = np.array(list(itertools.chain(*isiList)))
    return np.std(flattened_isis)/np.mean(flattened_isis)

def get_fr_stats(spike_times):
    frs = []
    Lvs = []
    Cvs = []
    for unit_spikes in spike_times:
        unit_frs = 0
        unit_isis = []
        for trial_spikes in unit_spikes:
            if trial_spikes.ndim == 0:
                trial_spikes = np.expand_dims(trial_spikes, axis = 0)
            if trial_spikes.ndim == 2:
                if trial_spikes.shape[0] == 0: continue
                trial_spikes = np.squeeze(trial_spikes, axis = 0)
            unit_frs = unit_frs + trial_spikes[(-1 < trial_spikes) & (trial_spikes < 2)].size
            unit_isis.append(np.diff(trial_spikes))
        unit_frs = unit_frs/(3*unit_spikes.size)
        Lvs.append(Lv(unit_isis))
        Cvs.append(Cv(unit_isis))
        frs.append(unit_frs)
    return np.array(frs), np.array(Lvs), np.array(Cvs)

