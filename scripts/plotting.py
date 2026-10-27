# IED script for plotting the behavioural and modelling data

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def group_level_plots(traj, costs, p_switch, p_switch_diff):

    sns.set_style("ticks")      # cleaner than whitegrid
    sns.set_context("paper", font_scale=1.3)

    # mouse data with are GH, ISO, and DREADDs groups
    palette_mapping = {'GH': '#7C6761', 'ISO': '#6668AC', 'GH DREADDs': '#AFA494', 'ISO DREADDs': '#DBD6D2'}  # Adjust mapping based on conditions

    # line plot
    ax = sns.pointplot(data=traj, 
                       x = 'block', 
                       y = 'errors', 
                       hue='condition', 
                       palette=palette_mapping, 
                       errorbar = 'se',
                       linewidth = 4)
    ax.set_ylim(0,15)
    ax.set_title('Errors per stage')
    sns.despine(ax=ax)
    plt.tight_layout()
    #plt.savefig('plots/error_trajectory' + group[0] + '.svg')
    plt.show()

    # shift cost distributions
    plt.figure(figsize=(3,5))
    ax = sns.boxplot(costs, 
                     x = 'condition', 
                     y = 'shift_cost', 
                     hue='condition', 
                     palette=palette_mapping, 
                     showfliers = False,
                     linewidth = 2, 
                     showmeans = True)
    
    ax = sns.stripplot(costs, 
                       x = 'condition', 
                       y = 'shift_cost',  
                       hue='condition',
                       palette=palette_mapping, 
                       edgecolor='k', 
                       linewidth=2)
    ax.set_title('Shift costs for each group')
    ax.set_xlabel('Condition')
    ax.set_ylabel('Shift cost (EDS - IDS4)')
    ax.set_ylim(-7.5,20)
    sns.despine(ax=ax)
    plt.xticks(rotation=45)
    plt.title("Shift cost")
    plt.tight_layout()
    #plt.savefig('plots/shift_costs_controls.svg')
    plt.show

    # p_switch distributions
    plt.figure(figsize=(3,5))
    ax = sns.boxplot(p_switch, 
                        x = 'condition', 
                        y = 'p_switch', 
                        hue='condition', 
                        palette=palette_mapping, 
                        showfliers = False,
                        linewidth = 2,
                        showmeans = True)
    
    ax = sns.stripplot(p_switch, 
                        x = 'condition', 
                        y = 'p_switch',  
                        hue='condition',
                        palette=palette_mapping, 
                        edgecolor='k', 
                        linewidth=2)
    ax.set_title('P(switch) after 1st win (IDS stages)')
    ax.set_xlabel('Condition')
    ax.set_ylabel('p(switch)')
    sns.despine(ax=ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.savefig('plots/shift_costs_controls.svg')
    plt.show


    # p switch diff distributions
    plt.figure(figsize=(3,5))
    ax = sns.boxplot(p_switch_diff, 
                        x = 'condition', 
                        y = 'diff', 
                        hue='condition', 
                        palette=palette_mapping, 
                        showfliers = False,
                        linewidth = 2,
                        showmeans = True)
    
    ax = sns.stripplot(p_switch_diff, 
                        x = 'condition', 
                        y = 'diff',  
                        hue='condition',
                        palette=palette_mapping, 
                        edgecolor='k', 
                        linewidth=2)
    ax.set_title('P(switch) difference (IDS4 - IDS1)')
    ax.set_xlabel('Condition')
    ax.set_ylabel('p(switch) difference')
    sns.despine(ax=ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.savefig('plots/shift_costs_controls.svg')
    plt.show

def individual_plots(traj, costs):

    n = len(traj['mouse'].unique()) 

    fig, axs = plt.subplots(n,2)

    fig.set_figheight(10*n)
    fig.set_figwidth(2*n)

    # loop over mice
    i=0
    for mouse in traj['mouse'].unique():

        # get the individual subj we want to plot
        tmp_df = traj[traj['mouse'] == mouse]

        # let's plot errors over blocks
        sns.lineplot(tmp_df,x='block',y='errors',ax = axs[i,0], linewidth = 10)
        condition = traj[traj['mouse'] == mouse]['condition'].unique() # get the condition for the plot title
        axs[i,0].set_title(str(mouse + ', ' + condition))

        # plot shift cost of current mouse on dist of all shift costs
        costs = costs[['mouse', 'shift_cost']]
        sns.histplot(costs,ax=axs[i,1])
        axs[i,1].set_xlim(-10,20)
        subj_shift_cost = costs[costs['mouse'] == mouse]['shift_cost'].tolist()
        axs[i,1].axvline(x=subj_shift_cost, color='red',linewidth=10) 

        i += 1

    plt.tight_layout()
    plt.show