#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 22:52:30 2022

@author: huange
"""
from spynal import matIO, spikes, utils
from spynal.matIO import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

spike_times, spike_times_schema, unit_info, trial_info, session_info, spike_waves, spike_waves_schema = \
    loadmat('/Volumes/common/scott/laminarPharm/mat/Lucky-DMS_ACSF_PFC-10162020-001.mat',
        variables=['spikeTimes','spikeTimesSchema','unitInfo','trialInfo', 'sessionInfo', 'spikeWaves', 'spikeWavesSchema'],
        typemap={'unitInfo':'DataFrame', 'trialInfo':'DataFrame'})\
#%%
for i in range(spike_times.shape[0]):
   for j in range(spike_times.shape[1]):
       spike = spike_times[i,j]
       trunc_spike = trunc_spike = (-1 < spike) & (spike < 2)
       if spike[trunc_spike]:
           spike_times[i,j] = np.atleast_1d(spike[trunc_spike])
       else:
           spike_times[i,j] = []

for i in range(spike_waves.shape[0]):
   for j in range(spike_waves.shape[1]):
       wave = spike_waves[i,j]
       trunc_wave = (-1 < wave) & (wave < 2)
       spike_waves[i,j] = np.atleast_1d(wave[trunc_wave])
       if len(np.shape(spike_waves[i,j])) == 1:
           spike_waves[i,j] = np.expand_dims(spike_waves[i,j], 1)
       
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
    n_units = np.shape(neurons_keep)[0]
    
    allISIs = spikes.isi(valid_spikes)
    ms_ISI = np.multiply(allISIs, 1000)
    flatAll = utils.concatenate_object_array(ms_ISI, 0) #every ISI per unit
    
    for neuron in range(n_units):
        shorts = np.where(flatAll[0, neuron] <= 1)[0] #all ISIs less than 1 ms for neuron
        if np.size(shorts) >= 1:
            print('flag')
        num_shorts = np.size(shorts)
        total_isi = np.size(flatAll[:, neuron][0])
        if num_shorts/total_isi >= 0.1:
            neurons_keep = np.delete(neurons_keep, neuron) #don't keep the neuron           
    
    newrates, newpts = spikes.rate(valid_spikes, method='bin', lims = [-1, 2])
    newmeanRates = np.mean(np.mean(newrates, axis = 2), axis = 0)
    newmeanRates = np.expand_dims(newmeanRates, axis=0)
    
    return neurons_keep, newmeanRates, ms_ISI

def waveAlign(spike_waves, spike_waves_schema, trials_keep, neurons_keep, trial_subset_indices = None):
    '''
    Gets the mean-aligned waveform from data. 
        
        Input: spikeWaves (n_trials, n_units) object array for single session
               spikeWavesSchema: 
               trials_keep: all valid trials
               neurons_keep: all valid units
               trial_subset_indices: optional subset of (valid) trials of interest
               
        Output: meanAlignWaves (n_trials, n_units): aligned mean waveforms for each unit
                smpRate: 10x interpolated sampling rate
                
    '''
    
    #TODO: truncate for -1 to 2 ms
    if trial_subset_indices: #if we pass in some subset
        valid_waves = spike_waves[np.ix_(trial_subset_indices, neurons_keep)]
    else: 
        valid_waves = spike_waves[np.ix_(trials_keep, neurons_keep)]
    
    n_trials, n_units = np.shape(valid_waves)
    
    timepts = (np.shape(valid_waves[0, 0])[0]-1)*10
    meanAlignWaves = np.zeros((timepts, n_units))
    
    for neuron in range(n_units):
        spikesAll = utils.concatenate_object_array(valid_waves[:, neuron], axis = 0, elem_axis = 1) #unexpected elem_axis
        n_timepts, n_spikes = np.shape(spikesAll) #get # of time pts
        x = np.arange(0, n_timepts)
        xinterp = np.arange(0, n_timepts-1, 0.1) #keep length, divide step by 10
        waves_interp = utils.interp1(x, spikesAll, xinterp, axis = 0)
        meanWave = np.mean(waves_interp, axis=1) #get mean waveform over all spikes
        meanTroughIdx = np.argmin(meanWave) #get mean trough idx
        
        for spike_idx in range(n_spikes):
            spike = waves_interp[:, spike_idx]
            spikeTroughIdx = np.argmin(spike)
            diff = spikeTroughIdx - meanTroughIdx
            newSpike = np.zeros(np.shape(spike))
            if diff > 0: # if the spike's trough is shifted ahead of mean trough
                newSpike[:-diff] = spike[diff:] #move it back
            elif diff < 0:
                newSpike[abs(diff):] = spike[:diff]
            else:
                newSpike = spike
            waves_interp[:, spike_idx] = newSpike
            
        # new waves_interp with aligned spikes
        meanAlignWave = np.mean(waves_interp, axis=1) #take mean of all spikes
        meanAlignWaves[:, neuron] = meanAlignWave
        
        smpRate = spike_waves_schema['smpRate']*10
        
    return meanAlignWaves, smpRate

def LV(ISIs):
    n_trials, n_units = np.shape(ISIs)
    allLV = np.zeros((1, n_units))
    
    for neuron in range(n_units):
        neuronLV = np.zeros((n_trials,))
        for trial in range(n_trials): 
            if len(ISIs[trial, neuron]) <= 1: #if there are no ISIs to compare against each other
                neuronLV[trial] = float('NaN')
            else: 
                LV = spikes.isi_stats(ISIs[trial, neuron], stat='LV')
                neuronLV[trial] = LV
        meanLV = np.nanmean(neuronLV, axis=0)
        allLV[:, neuron] = meanLV
    return allLV
    
def preProcessing(spike_times, trial_info, session_info, spike_waves, spike_waves_schema):
    validTrials = trialSelection(trial_info, session_info)
    validNeurons, meanRates, ISIs = neuronSelection(spike_times, validTrials)
    meanAlignWaves, smpRate = waveAlign(spike_waves, spike_waves_schema, validTrials, validNeurons)
    
    return validTrials, validNeurons, meanRates, ISIs, meanAlignWaves, smpRate

def featExtract(meanRates, ISIs, meanAlignWaves, smpRate):    
    '''
    Extracts features of interest from data. 
        
        Input: meanRates (n_units): mean spike rates for each unit 
               ISIs (n_trials, n_units): ISIs for each trial and unit
               meanAlignWaves (n_timepts, n_units): mean aligned wave for each unit
               smpRate: 10x interpolated sampling rate
        Output: featuresDF: dataframe containing features of interest
                            meanRates, troughToPeak, repolTime, CV, LV

    '''
    
    troughToPeak = spikes.waveform_stats(meanAlignWaves, stat='width', smp_rate=smpRate) #axis is 0?
    repolTime = spikes.waveform_stats(meanAlignWaves, stat='repolarization', smp_rate=smpRate)
    
    CV = spikes.rate_stats(meanRates, stat='CV')
    allLV = LV(ISIs)
    
    features = {'meanRates': meanRates.tolist(), 'troughToPeak':troughToPeak.tolist(), 'repolTime': repolTime.tolist(), 'CV': CV, 'LV': allLV.tolist()}
    featuresDF = pd.DataFrame(data=features)
    
    return featuresDF

def main():     
    validTrials, validNeurons, meanRates, ISIs, meanAlignWaves, smpRate = preProcessing(spike_times, trial_info, session_info, spike_waves, spike_waves_schema)
    featuresDF = featExtract(meanRates, ISIs, meanAlignWaves, smpRate)

if __name__ == "__main__":
    main()

