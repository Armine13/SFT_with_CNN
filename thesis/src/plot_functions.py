#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:23:36 2017

@author: arvardaz
"""
#%%
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import numpy as np
import matplotlib.mlab as mlab
import pandas as pd
import seaborn as sns
#%%
      
if __name__ == '__main__':
    #%%
    data_rig_easy = np.load('results/rig_easy.npz')
    data_def_easy = np.load('results/def_easy_.npz')
    data_def_hard = np.load('results/def_hard.npz')
    data_rig_hard = np.load('results/rig_hard.npz')
    data_rig_gt = np.load('results/rig_gt.npz')
#    outdir = '/home/arvardaz/SFT_with_CNN/src/figures/'
    outdir = '/home/arvardaz/Dropbox/out/'
#    edges = np.genfromtxt("edges.csv", dtype=np.int32)
#    synth_gt_dist = np.genfromtxt("dist_norm.csv")
    
    obj_size = 23.4
    
#    plot_n = 7
#    def rmse(a, b):
#        return np.sqrt(np.mean(np.square(a-b)))
#        
#    def rel_error(a, b):
#        return np.abs(a-b)/a
    
#    n = data['pred'].shape[0]
#    n = 10
#    cumul_loss = 0
#    cumul_iso_loss = 0
#    for i in range(n):
#        pred = data['pred'][i,:].reshape((1002,3))
#        gt = data['gt'][i].reshape((1002,3))
#        gt_fl = data['gt_fl'][i]
#        pred_fl = data['pred_fl'][i]
#        im = data['im'][i]
        
        
#    #%% plot rmse vs depth
#    depth =np.mean(data['gt'][:,2::3],axis=1)
#    
#    arr1inds = depth.argsort()
#    sorted_arr1 = depth[arr1inds[::-1]]
#    sorted_arr2 = data['fl_re'][arr1inds[::-1]]
#
#    fig = plt.figure()
#    
#    plt.plot(sorted_arr1, sorted_arr2,'ro')
#    plt.xlabel('Depth')
#    plt.ylabel('RMSE')
#    plt.grid('on')
#    plt.show()
#       
    
#%% histogram of rmse rig
    sns.set(color_codes=True)
    fig = plt.figure()
    x1 = data_rig_easy['rmse']*1.0/obj_size*100
    x2 = data_rig_hard['rmse']*1.0/obj_size*100
    x3 = data_rig_gt['rmse']*1.0/obj_size*100
#    x_rig_gt = data2['rmse']*1.0/obj_size*100
    
    ax = sns.kdeplot(x1, shade=True, label="Rigid: synth. basic");
    ax = sns.kdeplot(x2, shade=True,label="Rigid: synth. full");
    ax = sns.kdeplot(x3, shade=True,label="Rigid: real");
    ax.set(xlabel=r'$RMSE\, (\%)$',ylabel='Probability')
#    plt.show()
#    plt.savefig('destination_path.eps', format='eps', dpi=1000)
    plt.savefig(outdir+'hist_rmse_rig.png', dpi=1000)
        

#%% histogram of rmse def
    sns.set(color_codes=True)
    fig = plt.figure()
    x1 = data_def_easy['rmse']*1.0/obj_size*100
    x2 = data_def_hard['rmse']*1.0/obj_size*100
#    x_rig_gt = data2['rmse']*1.0/obj_size*100
    
    ax = sns.kdeplot(x1, shade=True, label="Deformable: synth. basic");
    ax = sns.kdeplot(x2, shade=True,label="Deformable: synth. full");
    ax.set(xlabel=r'$RMSE\, (\%)$',ylabel='Probability')
#    plt.show()
    plt.savefig(outdir+'rmse_bar_chart_defs.png', dpi=1000)
    
    
    #%%
#    histograms fl all datasets
    sns.set(color_codes=True)
    fig = plt.figure()
    x1 = data_rig_easy['fl_re']*100
    x2 = data_rig_hard['fl_re']*100
    x3 = data_rig_gt['fl_re']*100
    x4 = data_def_easy['fl_re']*100
    x5 = data_def_hard['fl_re']*100
#    x_rig_gt = data2['rmse']*1.0/obj_size*100
    
    ax = sns.kdeplot(x1, shade=True, label="Rigid: synth. basic");
    ax = sns.kdeplot(x2, shade=True,label="Rigid: synth. full");
    ax = sns.kdeplot(x3, shade=True,label="Rigid: real");
    ax = sns.kdeplot(x4, shade=True,label="Deformable: synth. basic");
    ax = sns.kdeplot(x5, shade=True,label="Deformable: synth. full");
    ax.set(xlabel= r'$FL_{re}\, (\%)$',ylabel='Probability')
#    plt.show()
    plt.savefig(outdir+'hist_fl_re.png', dpi=1000)
        
#    #%%    
##    plt.clf()
#    fig = plt.figure()
#    x = data['fl_re']
#    x2 = data2['fl_re']
#    ax = sns.kdeplot(x, shade=True, label="rigid:basic")
#    ax = sns.kdeplot(x2, shade=True,label="rigid:full")
#    ax.set(xlabel='relative error',ylabel='Probability')
##    plt.savefig(outdir+'rmse_bar_chart.png')
#    plt.show()


    #%% rmse
    sns.set(style="whitegrid")
    rmse_rig_easy = data_rig_easy['rmse']*1.0/obj_size*100
    rmse_rig_hard = data_rig_hard['rmse']*1.0/obj_size*100
    rmse_rig_gt = data_rig_gt['rmse']*1.0/obj_size*100
    rmse_def_easy = data_def_easy['rmse']*1.0/obj_size*100
    rmse_def_hard = data_def_hard['rmse']*1.0/obj_size*100
    
    df_rig = pd.DataFrame({r'$RMSE\, (\%)$': rmse_rig_easy, 'dataset': 'synth. basic','model':'rigid'})
    df2 = pd.DataFrame({r'$RMSE\, (\%)$': rmse_rig_hard, 'dataset': 'synth. full','model':'rigid'})
    df_gt = pd.DataFrame({r'$RMSE\, (\%)$': rmse_rig_gt, 'dataset': 'real','model':'rigid'})
    df_rig = df_rig.append(df2, ignore_index=True)
    df_rig = df_rig.append(df_gt,ignore_index=True)
    
    df_def = pd.DataFrame({r'$RMSE\, (\%)$': rmse_def_easy, 'dataset': 'synth. basic','model':'deformable'})
    df2 = pd.DataFrame({r'$RMSE\, (\%)$': rmse_def_hard, 'dataset': 'synth. full','model':'deformable'})
    df_def = df_def.append(df2, ignore_index=True)
    df_all = df_rig.append(df_def)
    
    fig = plt.figure()
    ax = sns.barplot(x = "model",y = r'$RMSE\, (\%)$',hue='dataset',data=df_all)
    plt.savefig(outdir+'rmse_bar_chart.png', dpi=1000)
    #%% fl
    
    
    flre_rig_easy = data_rig_easy['fl_re']*100
    flre_rig_hard = data_rig_hard['fl_re']*100
    flre_rig_gt = data_rig_gt['fl_re']*100
    flre_def_easy = data_def_easy['fl_re']*100
    flre_def_hard = data_def_hard['fl_re']*100
        
    sns.set(style="whitegrid")
    df_rig = pd.DataFrame({r'$FL_{re}\, (\%)$': flre_rig_easy, 'dataset': 'synth. basic','model':'rigid'})
    df2 = pd.DataFrame({r'$FL_{re}\, (\%)$': flre_rig_hard, 'dataset': 'synth. full','model':'rigid'})
    df_gt = pd.DataFrame({r'$FL_{re}\, (\%)$': flre_rig_gt, 'dataset': 'real','model':'rigid'})
    df_rig = df_rig.append(df2, ignore_index=True)
    df_rig = df_rig.append(df_gt,ignore_index=True)
    
    df_def = pd.DataFrame({r'$FL_{re}\, (\%)$': flre_def_easy, 'dataset': 'synth. basic','model':'deformable'})
    df1 = pd.DataFrame({r'$FL_{re}\, (\%)$': flre_def_hard, 'dataset': 'synth. full','model':'deformable'})
    df_def = df_def.append(df1, ignore_index=True)
    df_all = df_rig.append(df_def)
    
    fig = plt.figure()
    ax = sns.barplot(x = "model",y = r'$FL_{re}\, (\%)$',hue='dataset',data=df_all)    
    plt.savefig(outdir+'flre_bar_chart.png', dpi=1000)
    