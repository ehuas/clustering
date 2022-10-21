#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 22:52:30 2022

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
       
#%% 

def trialSelection(trial_info, session_info):
    '''
    Selects valid trials from data. 
        
        Input: trialInfo (n_trials, n_variables) DataFrame for single session
        Output: (valid_trials,) vector of indices of units to keep
    
    Valid trials are defined as: 
        (a) correct and 
        (b) ≤ 5 trials before drug injection onset

    '''
    drugStartTrial = session_info['drugStartTrial']
    beforeDrugTrials = trial_info[:drugStartTrial-5]
    trials_keep = np.where(beforeDrugTrials['correct'])[0]

    return trials_keep
    
def neuronSelection(spike_times, trials_keep):
    '''
    Selects valid neurons from data. 
        
        Input: spikeTimes (n_trials, n_units) object array for single session
        Output: (n_units,) bool vector indicating which units to keep
    
    Valid neurons are defined as: 
        (a) having an overall mean spike rate > 1 spike/s (weeding out silent cells) and 
        (b) having < 0.1% ISIs within 1 ms (weeding out poorly isolated single neurons)

    '''
    valid_spikes = spike_times[trials_keep, :] #keeps only the valid trials 
    
    rates, timepts = spikes.rate(valid_spikes, method='bin', lims = [-1, 2])
    meanRates = np.mean(np.mean(rates, axis = 2), axis = 0) #takes mean over all trials & timepts
    
    neurons_keep = np.where(meanRates > 1)[0] #find all indices where mean rate > 1
    valid_spikes = valid_spikes[:, neurons_keep]
    n_units = np.shape(neurons_keep)
    
    allISIs = spikes.isi(valid_spikes)
    ms_ISI = np.multiply(allISIs, 1000)
    flatAll = utils.concatenate_object_array(ms_ISI, 0) #every ISI per unit
    
    for neuron in range(len(n_units)):
        shorts = np.where(flatAll[:, neuron][0] <= 1)[0] #all ISIs less than 1 ms for neuron
        num_shorts = np.size(shorts)
        total_isi = np.size(flatAll[:, neuron][0])
        if num_shorts/total_isi >= 0.1:
            neurons_keep = np.delete(neurons_keep, neuron) #don't keep the neuron           
    
    return neurons_keep, meanRates, ms_ISI

def waveAlign(spike_waves, spike_waves_schema, trials_keep, neurons_keep, trial_subset_indices):
    valid_waves = spike_waves[trials_keep, neurons_keep]
    
    n_trials, n_units = np.shape(valid_waves)
    x = np.arange(0, spike_waves_schema.smpRate)
    xinterp = np.arange(0, spike_waves_schema.smpRate, 0.1) #keep length, divide step by 10
    valid_waves_interp = utils.interp1(x, valid_waves, xinterp) #interpolate 
    
    for neuron in range(len(n_units)):
        spikesAll = utils.concatenate_object_array(valid_waves_interp[:, neuron], axis = 0, elem_axis = 1)
        meanWave = np.mean(spikesAll, axis=1) #get mean waveform over all spikes
        meanTroughIdx = np.argmin(meanWave) #get mean trough idx
        
        # not sure if i can simplify this since i'm modifying the old data structure
        #BUILD NEW DATA STRUCTURE HERE?
        for trial in range(len(n_trials)):
            spikes = valid_waves_interp[trial, neuron]
            for spike in spikes:
                n_timepts, = np.shape(spike)
                spikeTroughIdx = np.argmin(spike)
                diff = spikeTroughIdx - meanTroughIdx
                newSpike = np.zeros(np.shape(spike))
                if diff >= 0: # if the spike's trough is shifted ahead of mean trough
                    newSpike[0:n_timepts-diff] = spike[diff:] #move it back
                else:
                    newSpike[abs(diff):] = spike[diff:]
                spikes[spike] = newSpike
        
    union = np.intersect1d(trials_keep, trial_subset_indices)
    subset = spike_waves[union, neurons_keep] #do we need to check if the trial indices are valid trials?
    subset_interp = utils.interp1(x, subset, xinterp)
    meanAlignWaves = np.zeros((n_timepts, n_units))
    
    for neuron in range(len(n_units)):
        alignedSpikesAll = utils.concatenate_object_array(subset_interp[:, neuron], axis = 0, elem_axis = 1)
        meanAlignWave = np.mean(alignedSpikesAll, axis=1)
        meanAlignWaves[:, neuron] = meanAlignWave
        
    return meanAlignWaves

def preProcessing(spike_times, trial_info, session_info, spike_waves, spike_waves_schema):
    validTrials = trialSelection(trial_info, session_info)
    validNeurons, meanRates, ISIs = neuronSelection(spike_times, validTrials)
    meanAlignWave = waveAlign(spike_waves, spike_waves_schema)
    
    return validTrials, validNeurons, meanRates, ISIs, meanAlignWave

def main():     
    validTrials, validNeurons, meanRates, ISIs, meanAlignWave = preProcessing(spike_times, trial_info, session_info, spike_waves, spike_waves_schema)
    print('all valid trials ', validTrials)
    print('all valid neurons ', validNeurons)
    print('length of all valid neurons ', len(validNeurons))

if __name__ == "__main__":
    main()

