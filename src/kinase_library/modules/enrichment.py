"""
##########################################
# The Kinase Library - Kinase Enrichment #
##########################################
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy.cluster import hierarchy
from adjustText import adjust_text
from natsort import natsorted, natsort_keygen
from tqdm import tqdm

from ..utils import _global_vars, exceptions
from . import data

pd.set_option('future.no_silent_downcasting', True)

# mpl.use("Qt5Agg")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

#%%
"""
###########################
# Differential Phosphorylation #
###########################
"""

def dp_regulated_sites(dp_data, lfc_col, lfc_thresh=0,
                       pval_col=None, pval_thresh=0.05,
                       percent_rank=None, percent_thresh=20,
                       drop_na=True, suppress_warnings=False):
    """
    Splitting data into up/down/un-regulated sites based on lfc and p-value.

    Parameters
    ----------
    dp_data : pd.DataFrame
        Differential Phosphorylation data. Must have lfc column.
    lfc_col : str
        lfc column name.
    fc_thresh : float
        lfc threshold for up/down/un-regulated sites.
    pval_col : str, optional
        P-value column name. The default is None.
    pval_thresh : float, optional
        P-value threshold for up/down/un-regulated sites. The default is 0.05.
    drop_na : bool, optional
        Drop entries with NA in their lfc or p-value columns. The default is True.
    suppress_warnings : bool, optional
        Do not print warnings. The default is False.

    Returns
    -------
    dp_data : pd.DataFrame
        Valid Differential Phosphorylation data after dropping missing values. Must have lfc column.
    reg_sites_dict : dictionary
        Dictionary with the 3 types of regulated sites: upregulated, downregulated, and unregulated.
    dropped_enteries : pd.DataFrame
        Dropped enteries due to missing differential phosphorylation data.
    """

    if lfc_thresh < 0:
        raise ValueError('Log2FC threshold cannot be negative.')
    if not 0<=pval_thresh<=1:
        raise ValueError('P-values threshold must be between 0-1.')
    if not 0<=percent_thresh<=100:
        raise ValueError('Percent threshold must be between 0-100.')
        
    all_dp_data = dp_data.copy()
    if drop_na:
        dp_data = all_dp_data[~all_dp_data[lfc_col].isna()]
        if pval_col is not None:
            dp_data = dp_data[~dp_data[pval_col].isna()]
        if (len(all_dp_data)-len(dp_data) > 0) and not suppress_warnings:
            if pval_col is not None:
                print(str(len(all_dp_data)-len(dp_data)) + ' entries were dropped due to empty value in the logFC or p-value columns.')
            else:
                print(str(len(all_dp_data)-len(dp_data)) + ' entries were dropped due to empty value in the logFC column.')
    
    dropped_enteries = pd.concat([all_dp_data, dp_data])
    dropped_enteries = dropped_enteries.loc[dropped_enteries.astype(str).drop_duplicates(keep=False).index] #In order to deal with lists in the DataFrame
    
    if len(dropped_enteries)>0 and not suppress_warnings:
        print('Use the \'dp_dropped_enteries\' attribute to view dropped enteries due to invalid differential phosphorylation values.')
    
    if percent_rank is not None:
        if percent_rank == 'logFC':
            sorted_dp_data = dp_data.sort_values(lfc_col, ascending=False)
        elif percent_rank == 'pvalue':
            sort_values = -np.sign(dp_data[lfc_col])*np.log10(dp_data[pval_col])
            sorted_indices = sort_values.argsort()[::-1]
            sorted_dp_data = dp_data.iloc[sorted_indices]
        else:
            raise ValueError('percent_rank must be either \'logFC\' or \'pvalue\'.')
        reg_sites_dict = dict(zip(['upreg','unreg','downreg'], np.split(sorted_dp_data, [int(percent_thresh/100*len(dp_data)), int((1-percent_thresh/100)*len(dp_data))])))
    
    else:
        reg_sites_dict = {}
        if lfc_thresh == 0: # In case of lfc_thresh of zero, treat lfc of 0 as unregulated
            if pval_col is not None:
                reg_sites_dict['upreg'] = dp_data[(dp_data[lfc_col]>lfc_thresh) & (dp_data[pval_col]<=pval_thresh)]
                reg_sites_dict['downreg'] = dp_data[(dp_data[lfc_col]<-lfc_thresh) & (dp_data[pval_col]<=pval_thresh)]
                reg_sites_dict['unreg'] = dp_data[(dp_data[lfc_col].abs()==lfc_thresh) | (dp_data[pval_col]>pval_thresh)]
            else:
                reg_sites_dict['upreg'] = dp_data[dp_data[lfc_col]>lfc_thresh]
                reg_sites_dict['downreg'] = dp_data[dp_data[lfc_col]<-lfc_thresh]
                reg_sites_dict['unreg'] = dp_data[dp_data[lfc_col].abs()==lfc_thresh]
        else:
            if pval_col is not None:
                reg_sites_dict['upreg'] = dp_data[(dp_data[lfc_col]>=lfc_thresh) & (dp_data[pval_col]<=pval_thresh)]
                reg_sites_dict['downreg'] = dp_data[(dp_data[lfc_col]<=-lfc_thresh) & (dp_data[pval_col]<=pval_thresh)]
                reg_sites_dict['unreg'] = dp_data[(dp_data[lfc_col].abs()<lfc_thresh) | (dp_data[pval_col]>pval_thresh)]
            else:
                reg_sites_dict['upreg'] = dp_data[dp_data[lfc_col]>=lfc_thresh]
                reg_sites_dict['downreg'] = dp_data[dp_data[lfc_col]<=-lfc_thresh]
                reg_sites_dict['unreg'] = dp_data[dp_data[lfc_col].abs()<lfc_thresh]
    
    return(dp_data, reg_sites_dict, dropped_enteries)

#%%
"""
#######################
# Enrichment Analysis #
#######################
"""

def combine_binary_enrichment_results(enrichment_results_dict, data_type='kl_object', lff_col_name=None, pval_col_name=None):
    """
    Function to combine multiple binary enrichment results into lff and pval dataframes for plotting bubblemap.

    Parameters
    ----------
    enrichment_results : dict
        Dictionary of either kl.EnrichmentResults objects or pd.DataFrame enrichment results tables, and conditions as keys.
    data_type : str, optional
        Type of enrichment results data: 'kl_object' (kl.EnrichmentResults) or 'data_frame' (pd.DataFrame)
    lff_col_name : str, optional
        Log frequency factor column name. The default is None.
        If None, will be set to 'log2_freq_factor'.
    pval_col_name : str, optional
        Adjusted p-value column name. The default is None.
        If None, will be set to 'fisher_adj_pval'.
    
    Raises
    ------
    ValueError
        If not all enrichment results have the same list of enriched kinases.

    Returns
    -------
    lff_data : pd.DataFrame
        Dataframe with log frequency factor enrichment data of all conditions.
    pval_data : pd.DataFrame
        Dataframe with adjusted p-value enrichment data of all conditions.
    """

    if data_type not in ['kl_object','data_frame']:
        raise ValueError('data_type must be either \'kl_object\' or \'data_frame\'.')
    
    enrichment_results = list(enrichment_results_dict.values())
    conds_list = list(enrichment_results_dict.keys())
    
    if data_type == 'data_frame':
        enrichment_results_tables = enrichment_results
    else:
        enrichment_results_tables = [res.enrichment_results for res in enrichment_results]
        
    if lff_col_name is None:
        lff_col_name = 'log2_freq_factor'
    if pval_col_name is None:
        pval_col_name = 'fisher_adj_pval'
    
    index_test = [x.index.to_list() == enrichment_results_tables[0].index.to_list() for x in enrichment_results_tables]
    if not np.all(index_test):
        raise ValueError('All enrichment results must have the same kinases enriched.')
    kinases = enrichment_results_tables[0].index.to_list()
    
    lff_data = pd.DataFrame(index=kinases, columns=conds_list)
    pval_data = pd.DataFrame(index=kinases, columns=conds_list)
    
    for res,cond in zip(enrichment_results_tables,conds_list):
        lff_data[cond] = res[lff_col_name]
        pval_data[cond] = res[pval_col_name]
    
    return(lff_data,pval_data)
    
    
def combine_diff_phos_enrichment_results(enrichment_results_dict, enrichment_type='combined', data_type='kl_object',
                                        lff_col_name=None, pval_col_name=None, cont_kins_col_name=None):
    """
    Function to combine multiple DE enrichment results into lff and pval dataframes for plotting bubblemap.

    Parameters
    ----------
    enrichment_results : dict
        Dictionary of either kl.EnrichmentResults objects or pd.DataFrame enrichment results tables, and conditions as keys.
    conds_list : list
        List of conditions.
    enrichment_type : str, optional
        Enrichment type: 'combined', 'upregulated', 'downregulated'. The default is 'combined'.
    data_type : str, optional
        Type of enrichment results data: 'kl_object' (kl.EnrichmentResults) or 'data_frame' (pd.DataFrame)
    lff_col_name : str, optional
        Log frequency factor column name. The default is None.
        If None, will be set to 'log2_freq_factor'.
    pval_col_name : str, optional
        Adjusted p-value column name. The default is None.
        If None, will be set to 'fisher_adj_pval'.
    cont_kins_col_name : TYPE, optional
        Contradicting kinases column name. The default is None.
        If None, will be set to 'most_sig_direction'.

    Raises
    ------
    ValueError
        If not all enrichment results have the same list of enriched kinases.

    Returns
    -------
    lff_data : pd.DataFrame
        Dataframe with log frequency factor enrichment data of all conditions.
    pval_data : pd.DataFrame
        Dataframe with adjusted p-value enrichment data of all conditions.
    """

    exceptions.check_dp_enrichment_type(enrichment_type)
    if data_type not in ['kl_object','data_frame']:
        raise ValueError('data_type must be either \'kl_object\' or \'data_frame\'.')
    
    enrichment_results = list(enrichment_results_dict.values())
    conds_list = list(enrichment_results_dict.keys())
    
    if data_type == 'data_frame':
        enrichment_results_tables = enrichment_results
    else:
        if enrichment_type == 'upregulated':
            enrichment_type = 'upreg'
            enrichment_results_tables = [getattr(res,enrichment_type+'_enrichment_results').enrichment_results for res in enrichment_results]
        elif enrichment_type == 'downregulated':
            enrichment_type = 'downreg'
            enrichment_results_tables = [getattr(res,enrichment_type+'_enrichment_results').enrichment_results for res in enrichment_results]
        else:
            enrichment_results_tables = [res.combined_enrichment_results for res in enrichment_results]
    
    if lff_col_name is None:
        lff_col_name = 'most_sig_'*(enrichment_type=='combined') + 'log2_freq_factor'
    if pval_col_name is None:
        pval_col_name = 'most_sig_'*(enrichment_type=='combined') + 'fisher_adj_pval'
    
    index_test = [x.index.to_list() == enrichment_results_tables[0].index.to_list() for x in enrichment_results_tables]
    if not np.all(index_test):
        raise ValueError('All enrichment results must have the same kinases enriched.')
    kinases = enrichment_results_tables[0].index.to_list()
    
    lff_data = pd.DataFrame(index=kinases, columns=conds_list)
    pval_data = pd.DataFrame(index=kinases, columns=conds_list)
    if enrichment_type == 'combined':
        cont_kins_data = pd.DataFrame(False, index=kinases, columns=conds_list)
    
    for res,cond in zip(enrichment_results_tables,conds_list):
        lff_data[cond] = res[lff_col_name]
        pval_data[cond] = res[pval_col_name]
        if enrichment_type == 'combined':
            cont_kins_data.loc[enrichment_results_dict[cond].contradicting_kins(),cond] = True

    if enrichment_type == 'combined':
        return(lff_data,pval_data,cont_kins_data)
    else:
        return(lff_data,pval_data)


def create_kin_sub_sets(data_values, threshold, comp_direction):
    
    if comp_direction not in ['higher','lower']:
        raise ValueError('\'comp_direction\' must be either \'higher\' or \'lower\'.')
    
    if comp_direction == 'higher':
        pred_kins = (data_values>=threshold)
    elif comp_direction == 'lower':
        pred_kins = (data_values<=threshold)
    
    kins_subs_dict = {}
    for kin in tqdm(pred_kins.columns):
        kins_subs_dict[kin] = pred_kins.loc[pred_kins[kin],kin].index.to_list()
    
    return(kins_subs_dict)


def combine_mea_enrichment_results(enrichment_results_dict, data_type='kl_object',
                                   lff_col_name='NES', pval_col_name=None, adj_pval=True):
    """
    Function to combine multiple MEA enrichment results into lff and pval dataframes for plotting bubblemap.

    Parameters
    ----------
    enrichment_results : dict
        Dictionary of either kl.EnrichmentResults objects or pd.DataFrame enrichment results tables, and conditions as keys.
    conds_list : list
        List of conditions.
    data_type : str, optional
        Type of enrichment results data: 'kl_object' (kl.EnrichmentResults) or 'data_frame' (pd.DataFrame)
    lff_col_name : str, optional
        Log frequency factor column name. The default is None.
        If None, will be set to 'log2_freq_factor'.
    pval_col_name : str, optional
        Adjusted p-value column name. The default is None.
        If None, will be set to 'fisher_adj_pval'.

    Raises
    ------
    ValueError
        If not all enrichment results have the same list of enriched kinases.

    Returns
    -------
    lff_data : pd.DataFrame
        Dataframe with log frequency factor enrichment data of all conditions.
    pval_data : pd.DataFrame
        Dataframe with adjusted p-value enrichment data of all conditions.
    """
    
    if data_type not in ['kl_object','data_frame']:
        raise ValueError('data_type must be either \'kl_object\' or \'data_frame\'.')
    
    enrichment_results = list(enrichment_results_dict.values())
    conds_list = list(enrichment_results_dict.keys())
    
    if data_type == 'data_frame':
        enrichment_results_tables = enrichment_results
    else:
        enrichment_results_tables = [res.enrichment_results for res in enrichment_results]
    
    index_test = [x.index.to_list() == enrichment_results_tables[0].index.to_list() for x in enrichment_results_tables]
    if not np.all(index_test):
        raise ValueError('All enrichment results must have the same kinases enriched.')
    kinases = enrichment_results_tables[0].index.to_list()
    
    if pval_col_name is None:
        if adj_pval:
            pval_col_name = 'FDR'
        else:
            pval_col_name = 'pvalue'
    
    lff_data = pd.DataFrame(index=kinases, columns=conds_list)
    pval_data = pd.DataFrame(index=kinases, columns=conds_list)
    
    for res,cond in zip(enrichment_results_tables,conds_list):
        lff_data[cond] = res[lff_col_name]
        pval_data[cond] = res[pval_col_name]
    
    return(lff_data,pval_data)


#%%
"""
############
# Plotting #
############
"""

def plot_volcano(enrichment_data, sig_lff=0, sig_pval=0.1,
                 lff_col='log2_freq_factor', pval_col='fisher_adj_pval',
                 highlight_kins=None, kinases=None, ignore_depleted=False,
                 label_kins=None, adjust_labels=True, labels_fontsize=7,
                 symmetric_xaxis=True, grid=True, max_window=False,
                 title=None, xlabel=None, ylabel=None,
                 plot=True, save_fig=False, return_fig=False,
                 ax=None, **plot_kwargs):
    """
    Plot volcano plot of the Kinase Library enrichment results.

    Parameters
    ----------
    enrichment_data : pd.DataFrame
        Kinase Library enrichment results with kinases as the index; must include the lff_col and pval_col.
    sig_lff : float, optional
        Significance threshold for logFF in the enrichment results. The default is 0.
    sig_pval : float, optional
        Significance threshold for and adjusted p-value in the enrichment results. The default is 0.1.
    lff_col : str, optional
        Log frequency factor column name used for volcano plot. The default is 'log2_freq_factor'.
    pval_col : str, optional
        P-value column name used for volcano plot. The default is 'fisher_adj_pval'.
    highlight_kins : list, optional
        List of kinases to be marked in yellow on the kinase enrichment volcano plot.
    kinases : list, optional
        If provided, kinases to plot in the volcano plot. The default is None.
    label_kins : list, optional
        List of kinases to label on volcano plot. The default is None.
        If none, all significant kinases will be labelled plus any non-significant kinases marked for highlighting.
    adjust_labels : bool, optional
        If True, labels will be adjusted in space. The default is True.
    labels_fontsize : int, optional
        Font size used for the volcano's kinase labels. The default is 7.
    symmetric_xaxis : bool, optional
        If True, x-axis will be made symmetric to its maximum absolute value. The default is True.
    grid : bool, optional
        Plot grid in the volcano plot. The default is True.
    max_window : bool, optional
        Maximize the plotting window. The default is False.
        Must be False if an axis is provided to the function.
    title : str or bool, optional
        Title for the volcano plot. The default is False.
    xlabel : str or bool, optional
        x-axis label for the volcano plot. The default is 'log$_2$(Frequency Factor)'.
    ylabel : str or bool, optional
        y-axis label for the volcano plot. The default is '-log$_{10}$(Adjusted p-value)'.
    plot : bool, optional
        Plot the generated enrichment volcano plot. The default is True.
        Will be automatically changed to False if an axis is provided.
    save_fig : str, optional
        Path to file for saving the volcano plot. The default is False.
        Must be False if an axis is provided.
    return_fig : bool, optional
        If true, the volcano plot will be returned as a plt.figure object. The default is False.
    ax : plt.axes, optional
        Axes provided to plot the kinase enrichment volcano onto. The default is None.
    **plot_kwargs: optional
        Optional keyword arguments passed into the plot_volcano function, otherwise defaults will be set as seen below.

    Returns
    -------
    If return_fig, the Kinase Library enrichment volcano plot.
    """

    plot_kwargs['marker'] = plot_kwargs.get('marker','.')
    plot_kwargs['markersize'] = plot_kwargs.get('markersize',7)
    plot_kwargs['markeredgewidth'] = plot_kwargs.get('markeredgewidth',0)
    plot_kwargs['markeredgecolor'] = plot_kwargs.get('markeredgecolor','k')
    plot_kwargs['alpha'] = plot_kwargs.get('alpha',1)

    if kinases is not None:
        if len(kinases) != len(set(kinases)):
            kinases = list(dict.fromkeys(kinases))
        enrichment_data = enrichment_data.loc[kinases]
        if highlight_kins is not None:
            highlight_kins = [x for x in highlight_kins if x in kinases]
    
    enrichment_data[pval_col] = enrichment_data[pval_col].astype(float)
    
    if ignore_depleted:
        if 'most_sig_direction' in enrichment_data.columns:
            signed_log2_ff = (enrichment_data['most_sig_direction'].replace({'-': -1, '+': 1}))*enrichment_data['most_sig_log2_freq_factor']
            enrichment_data.loc[(signed_log2_ff < 0), 'most_sig_log2_freq_factor'] = 0
            enrichment_data.loc[(signed_log2_ff < 0), 'most_sig_fisher_pval'] = 1
            enrichment_data.loc[(signed_log2_ff < 0), 'most_sig_fisher_adj_pval'] = 1
        else:
            print('Warning: \'most_sig_direction\' column does not exist, if ignore_depleted is True - all kinases with lff<0 will be ignored.')
            depleted_kins = (enrichment_data[lff_col] < 0)
            enrichment_data.loc[depleted_kins, lff_col] = 0
            enrichment_data.loc[depleted_kins, pval_col] = 1
    
    sig_enriched_data = enrichment_data[(enrichment_data[lff_col] >= sig_lff) & (enrichment_data[pval_col] <= sig_pval)]
    sig_depleted_data = enrichment_data[(enrichment_data[lff_col] <= -sig_lff) & (enrichment_data[pval_col] <= sig_pval)]
    non_sig_data = enrichment_data[~((np.abs(enrichment_data[lff_col]) >= sig_lff) & (enrichment_data[pval_col] <= sig_pval))]
    
    if ax is None:
        existing_ax = False
        fig,ax = plt.subplots()
    else:
        existing_ax = True
        plot = False
        if max_window or save_fig or return_fig:
            raise ValueError('When Axes provided, \'max_window\', \'save_fig\', and \'return_fig\' must be False.')

    ax.plot(non_sig_data[lff_col],-np.log10(non_sig_data[pval_col]), markerfacecolor='black', linestyle='None', **plot_kwargs)
    ax.plot(sig_enriched_data[lff_col],-np.log10(sig_enriched_data[pval_col]), markerfacecolor='red', linestyle='None', **plot_kwargs)
    ax.plot(sig_depleted_data[lff_col],-np.log10(sig_depleted_data[pval_col]), markerfacecolor='blue', linestyle='None', **plot_kwargs)
    
    if highlight_kins is not None:
        if isinstance(highlight_kins, str):
            highlight_kins = [highlight_kins]
        if not (set(highlight_kins) <= set(enrichment_data.index)):
            missing_kinases = list(set(highlight_kins) - set(enrichment_data.index))
            raise ValueError(f'Some kinases to highlight are not in the enrichment results ({missing_kinases}).')
        highlight_data = enrichment_data.loc[highlight_kins]
        ax.plot(highlight_data[lff_col],-np.log10(highlight_data[pval_col]), markerfacecolor='yellow', linestyle='None', **plot_kwargs)
    
    if symmetric_xaxis:
        ax.set_xlim(-abs(max(ax.get_xlim(), key=abs)), abs(max(ax.get_xlim(), key=abs)))
    
    ax.axhline(-np.log10(sig_pval), ls='--', lw=1, color='k')
    if sig_lff>0:
        ax.axvline(sig_lff, ls='--', lw=1, color='k')
        ax.axvline(-sig_lff, ls='--', lw=1, color='k')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    try:
        xvalues = int(np.floor(enrichment_data[lff_col].abs().max()))
    except:
        xvalues = 1
    ax.set_xticks(range(-xvalues,xvalues+1))
    if grid:
        ax.grid('major')
        
    if max_window:
        figManager = fig.canvas.manager
        figManager.window.showMaximized()
        fig.tight_layout()
    
    if label_kins is None:
        label_kins = list(sig_enriched_data.index) + list(sig_depleted_data.index)
        if highlight_kins:
            label_kins = label_kins + list(highlight_data.index)
    elif not (set(label_kins) <= set(enrichment_data.index)):
        missing_kinases = list(set(label_kins) - set(enrichment_data.index))
        raise ValueError(f'Some kinases to label are not in the enrichment results ({missing_kinases}).')
    label_kins = list(set(label_kins))

    if label_kins:
        labels = []
        for _,kin_data in enrichment_data.loc[label_kins].iterrows():
            if adjust_labels:
                labels.append(ax.text(kin_data[lff_col],-np.log10(kin_data[pval_col]), kin_data.name, fontsize=labels_fontsize))
            else:
                ax.annotate(kin_data.name, (kin_data[lff_col],-np.log10(kin_data[pval_col])), xytext=(2,2), textcoords='offset points', fontsize=labels_fontsize)
        if adjust_labels:
            adjust_text(labels, ax=ax, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
    
    if save_fig:
        fig.savefig(save_fig, dpi=1000)
        
    if not plot and not existing_ax:
        plt.close(fig)
            
    if return_fig:
        return fig


def plot_3x3_volcanos(dp_data, kin_type, kl_method, kl_thresh, dp_lfc_col, dp_lfc_thresh=[0,0.5,1],
                      dp_pval_col=None, dp_pval_thresh=[0.1,0.1,0.1], drop_dp_na=True, kinases=None,
                      seq_col=None, ke_sig_lff=0, ke_sig_pval=0.1,
                      plot_cont_kins=True, highlight_kins=None, ignore_depleted=True,
                      label_kins=None, adjust_labels=True, labels_fontsize=7, title=None,
                      plot=True, save_fig=False, return_fig=False,
                      suppress_warnings=True,
                      scoring_kwargs={},
                      diff_phos_kwargs={},
                      enrichment_kwargs={},
                      plotting_kwargs={}):
    """
    Returns a 3x3 figure containing downregulated, upregulated, and combined volcano plots of the Kinase Library differential phosphorylation enrichment results for three logFC thresholds.

    Parameters
    ----------
    dp_data : pd.DataFrame
        DataFrame containing differential phosphorylation data (must include sequence and logFC columns).
    kin_type : str
        Kinase type ('ser_thr' or 'tyrosine').
    kl_method : str
        Kinase Library scoring method ('score', 'score_rank', 'percentile', 'percentile_rank').
    kl_thresh : int
        The threshold to be used for the specified kl_method.
    dp_lfc_col : str
        LogFC column name for Kinase Library enrichment analysis.
    dp_lfc_thresh : list, optional
        List of three logFC cuttoffs used to define up, down, and unregulated sites.
    dp_pval_col : str, optional
        P-value column name used to define a site's significance.
    dp_pval_thresh : list, optional
        List of three significance threshold corresponding to the p-value column. The default is [0.1]*3.
    drop_dp_na : bool, optional
        Drop dp_data rows with NaN values in the logFC column. The default is True.
    kinases : list, optional
        If provided, kinase enrichment will only be calculated for the specified kinase list, otherwise, all kinases of the specified kin_type will be included. The default is None.
    seq_col : str, optional
        Substrates column name in the differential phosphorylation data. The default is None (will be set as _global_vars.default_seq_col).
    ke_sig_lff : float, optional
        Significance threshold for logFF in the enrichment results. The default is 0.
    ke_sig_pval : float, optional
        Significance threshold for and adjusted p-value in the enrichment results. The default is 0.1.
    plot_cont_kins : bool, optional
        If False, kinases enriched in both upregulated and downregulated sites will be excluded from the volcano.
        If True, they will be highlighted in yellow.
    highlight_kins : list, optional
        List of kinases to be marked in yellow on the kinase enrichment volcano plots.
    ignore_depleted : bool, optional
            Ignore kinases that their FF is negative (depleted). The default is True.
    label_kins : list, optional
        List of kinases to label on volcano plots. The default is None.
        If none, all significant kinases will be labelled plus any non-significant kinases marked for highlighting.
    adjust_labels : bool, optional
        If True, labels will be adjusted to avoid other markers and text on volcano plots. The default is True.
    title : str, optional
        Title for the figure. The default is False.
    plot : bool, optional
        Whether or not to plot the produced enrichment figure. The default is True.
        Will be automatically changed to False if an axis is provided.
    save_fig : str, optional
        Path to file for saving the figure. The default is False.
        Must be False if an axis is provided.
    return_fig : bool, optional
        If true, the figure will be returned as a plt.figure object. The default is False.
    suppress_warnings : bool, optional
        Do not print warnings. The default is False.
    scoring_kwargs : dict, optional
        Optional keyword arguments to be passed to the scoring function.
    diff_phos_kwargs : dict, optional
        Optional keyword arguments to be passed to the PhosphoProteomics initialization function.
    enrichment_kwargs : dict, optional
        Optional keyword arguments to be passed to the kinase_enrichment function.
    plotting_kwargs : dict, optional
        Optional keyword arguments to be passed to the plot_volcano function.
        
    Returns
    -------
    If return_fig, the 3x3 figure containing downregulated, upregulated, and combined kinase enrichment volcano plots.
    """
    
    if len(dp_lfc_thresh) != 3:
        raise ValueError('\'dp_lfc_thresh\' must contain exactly three values.')
    if dp_pval_col is not None and len(dp_pval_thresh) != 3:
        raise ValueError('\'dp_pval_thresh\' must contain exactly three values.')
    
    if seq_col is None:
        seq_col = _global_vars.default_seq_col
    
    exceptions.check_kl_method(kl_method)
    print('Calculating scores for all sites')
    dp_data_pps = pps.PhosphoProteomics(data=dp_data, seq_col=seq_col, **diff_phos_kwargs)
    if kl_method in ['score','score_rank']:
        scores = dp_data_pps.score(kin_type=kin_type, kinases=kinases, values_only=True, **scoring_kwargs)
    elif kl_method in ['percentile','percentile_rank']:
        percentiles = dp_data_pps.percentile(kin_type=kin_type, kinases=kinases, values_only=True, **scoring_kwargs)
        
    fig = plt.figure(constrained_layout=True)
    figManager = fig.canvas.manager
    figManager.window.showMaximized()
    subfigs = fig.subfigures(nrows=3, ncols=1)
        
    for i,(lfc,pval) in enumerate(zip(dp_lfc_thresh,dp_pval_thresh)):
        
        subfigs[i].suptitle(r'$\bf{' + f'DE\ logFC\ threshold:\ {lfc}' + f'\ /\ DE\ p-value\ threshold:\ {pval}'*(dp_pval_col is not None) + '}$')
        ax = subfigs[i].subplots(nrows=1, ncols=3)
        
        print(f'\nLogFC threshold: {lfc}' + f' / p-value threshold: {pval}'*(dp_pval_col is not None))
        diff_phos_data = DiffPhosData(dp_data=dp_data, kin_type=kin_type,
                                    lfc_col=dp_lfc_col, lfc_thresh=lfc,
                                    pval_col=dp_pval_col, pval_thresh=pval,
                                    seq_col=seq_col, drop_dp_na=drop_dp_na,
                                    **diff_phos_kwargs)
        if kl_method in ['score','score_rank']:
            diff_phos_data.submit_scores(kin_type=kin_type, scores=scores, suppress_messages=suppress_warnings)
        elif kl_method in ['percentile','percentile_rank']:
            diff_phos_data.submit_percentiles(kin_type=kin_type, percentiles=percentiles, suppress_messages=suppress_warnings)
        
        enrich_results = diff_phos_data.kinase_enrichment(kl_method=kl_method, kl_thresh=kl_thresh,
                                                       **enrichment_kwargs)
            
        enrich_results.plot_down_up_comb_volcanos(sig_lff=ke_sig_lff, sig_pval=ke_sig_pval, kinases=kinases,
                                                  plot_cont_kins=plot_cont_kins, highlight_kins=highlight_kins, ignore_depleted=ignore_depleted,
                                                  label_kins=label_kins, adjust_labels=adjust_labels, labels_fontsize=labels_fontsize, ax=ax,
                                                  **plotting_kwargs)
    
    fig.suptitle(title)
    
    if save_fig:
        fig.savefig(save_fig, dpi=1000)
        
    if not plot:
        plt.close(fig)
            
    if return_fig:
        return fig
    
    
def plot_bubblemap(lff_data, pval_data, cont_kins=None, sig_lff=0, sig_pval=0.1, kinases=None,
                   plot_cont_kins=True, highlight_cont_kins=True, sort_kins_by='family',
                   cond_order=None, only_sig_kins=False, only_sig_conds = False,
                   kin_clust=False, condition_clust=False, cluster_by=None,
                   cluster_by_matrix=None, cluster_method='average',
                   color_kins_by='family', kin_categories_colors=None, cond_colors=None,
                   title=None, family_legned=True, pval_legend=True, lff_cbar=True,
                   pval_legend_spacing=None, save_fig=False, max_window=True,
                   lff_clim=(-2,2), max_pval_size=4, bubblesize_range=(10,100),
                   num_panels=6, vertical=True, constrained_layout=True,
                   xaxis_label='Condition', yaxis_label='Kinase',
                   xlabel=True, xlabels_size=8, ylabel=True, ylabels_size=10):
    """
    Function to display a bubblemap with Kinase Library enrichment results inputted as log frequency factor and p-value matrices.

    Parameters
    ----------
    lff_data : pd.DataFrame
        Matrix containing Kinase Library enrichment log frequency factor data with kinases as index and conditions as columns. 
    pval_data : pd.DataFrame
        Matrix containing Kinase Library enrichment p_value data. Index (kinases) and columns (conditions) must be identical to lff_data.
    cont_kins : pd.DataFrame, optional
        Matrix containing boolean values specifying contradicting kinases in each condition. Index (kinases) and columns (conditions) must be identical to lff_data and pval_data. The default is None.
    sig_lff : float, optional
        The minimum log frequency factor value to be displayed on the bubblemap. The default is 0.
    sig_pval : float, optional
        The maximum p_value that will be displayed on the bubblemap. The default is 0.1.
    kinases : list, optional
        If provided, kinases to plot in the bubblemap. The default is None.
    plot_cont_kins : boolean, optional
        Plot contradicting kinases. The default is True.
    highlight_cont_kins : boolean, optional
        Highlight contradicting kinases. The default is True.
    sort_kins_by : str, optional
        String specifying what to sort the kinases on the figure's x-axes by. If provided, must be 'family' (default) or 'name'. 
    cond_order : list of str, optional
        Order in which the conditions will appear on the bubblemap. This list may be a subset of conditions but may not contain extra strings. The default is None, preserving the condition order from the lff_data matrix.
    only_sig_kins : bool, optional
        If True, only kinases that have a significant result for at least one condition will be displayed. The default is False.
    only_sig_conds : bool, optional
        If True, only conditions that have a significant result for at least one kinase will be displayed. The default is False.
    kin_clust : bool, optional
        If True, kinases will be clustered based on the cluster_by metric. The default is False.
    condition_clust : bool, optional
        If True, conditions will be clustered based on the cluster_by metric. The default is False.
    cluster_by : str, optional
        Metric specifying which matrix to cluster kinases or conditions by. Options are 'lff', 'pval', 'both', or 'custom'. The default is None.
    cluster_by_matrix : pd.DataFrame, optional
        If cluster_by is set to 'custom', a cluster_by_matrix with the same dimensions of the original data must be provided. The default is None.
    cluster_method : str, optional
        Method to be passed into scipy.cluster.hierarchy.linkage to calculate the distance between clusters. Options include 'average', 'single', 'complete'. The default is 'average'.
    color_kins_by : str, optional
        Grouping by which to color code the kinase names. Options are 'type', 'family' (default), or None. If None, all kinase names will appear in black.
    kin_categories_colors : dict, optional
        If specified, this dictionary of category-color pairs will be used to color code the kinases. The default is None.
    cond_colors : dict, optional
        Dictionary of colors for each condition. The default is None.
    title : str, optional
        String to be used as the figure's title. The default is None.
    family_legned : bool, optional
        If True, legend showing either kinase family or kinase type colors, depending on color_kins_by. The default is True.
    lff_cbar : bool, optional
        If True, legend displaying reflecting the log frequency factor colors from the bubblemap. The default is True.
    pval_legend : bool, optional
        If True, legend displaying four equivalently spaced p-values from the dataset (including min and max) is included. The default is True.
    pval_legend_spacing : float, optional
        Optional parameter to add spacing between elements in the p-value legend if the bubbles are large or overlapping. The default is None.
    save_fig : str, optional
        Path to file for saving the volcano plot. The default is False.
        Must be False if an axis is provided.
    max_window : bool, optional
        Maximize the plotting window. The default is True.
        Must be False if an axis is provided to the function.
    lff_clim : tuple of float, optional
        Color limit for the log frequency factors on the bubblemap. If not specified, the default is (-2,2).
    max_pval_size : float, optional
        Maximum p-value the data can be scaled to, prevents extremely high p-values from dwarfing smaller, still significant bubbles. The default is 4, meaning all bubbles of equal or greater significant than 1x10^-4 will appear as the same size.
    bubblesize_range : tuple of float,, optional
        This parameter allows the user to manipulate the allowed bubblesizes in the figure. The default is (10,100).
    num_panels : int, optional
        Integer used to determine the number of panels that kinases are split between. Useful when subsetting kinases or with the only_sig_kins option, defaults to 6.
    vertical : bool, optional
        If True, panels will be displayed vertically instead or horizontally. The default is True.
    constrained_layout : bool, optional
        Subplots, colorbars, and legends are adjusted to fit within the figure window. The default is True.
    xaxis_label : str, optional
        Label for the x-axis (in vertical layout). The default is 'Condition'.
    yaxis_label : str, optional
        Label for the y-axis (in vertical layout). The default is 'Kinase'.
    xlabel : bool, optional
        If True, the x-axes will each be labelled with 'Kinase'. The default is True.
    xlabels_size : float, optional
        If provided, float will be used to determine kinase label sizes. The default is 8.
    ylabel : bool, optional
        If True, the y-axes will each be labelled with 'Condition'. The default is True.
    ylabels_size : float, optional
        If provided, float will be used to determine condition label sizes. The default is 10.

    Raises
    ------
    ValueError
        Mismatched input matrix dimensions, invalid sort_kins_by, incomplete color dictionary, cluster_by not specified, cluster_by_matrix not provided for custom clustering, or invalid max_pval_size.

    Returns
    -------
    None.
    """
    
    if not ((lff_data.index.equals(pval_data.index)) and (lff_data.columns.equals(pval_data.columns))):
        raise ValueError('lff_data and pval_data must have the same columns and indices.')
    
    if (pval_data == 0).any().any():
        print('Warning: zeros were detected in pval_data. All zeros were replaced with the non-zero minimal value in pval_data.')
        pval_data = pval_data.replace(0, pval_data[pval_data>0].min().min())
    
    if cont_kins is None:
        cont_kins = pd.DataFrame(False, index=lff_data.index, columns=lff_data.columns)
    elif not ((lff_data.index.equals(cont_kins.index)) and (lff_data.columns.equals(cont_kins.columns))):
        raise ValueError('cont_kins must have the same columns and indices as lff_data and pval_data.')
    
    if kinases is None:
        kinases = lff_data.index.to_list()
    else:
        if len(kinases) != len(set(kinases)):
            kinases = list(dict.fromkeys(kinases))
    lff_data = lff_data.loc[kinases]
    pval_data = pval_data.loc[kinases]
    
    if not sort_kins_by:
        kins_order = kinases
    elif sort_kins_by == 'family':
        kinase_info = data.get_kinase_family(kinases).reset_index().sort_values(by=['FAMILY','MATRIX_NAME'], key=natsort_keygen()).set_index('MATRIX_NAME')
        kins_order = kinase_info.index
    elif sort_kins_by == 'name':
        kins_order = natsorted(kinases)
    else:
        raise ValueError('sort_kins_by must be either \'family\', \'name\', or False.')
    
    if cond_order is None:
        cond_order = lff_data.columns
    
    if color_kins_by:
        exceptions.check_color_kins_method(color_kins_by)
        label_info_col = color_kins_by.upper()
        
        kinome_data = data.get_kinase_info(kinases)
        kin_categories_list = natsorted(kinome_data[label_info_col].unique())
        if kin_categories_colors is None:
            kin_categories_colors = {k:getattr(_global_vars,color_kins_by+'_colors')[k] for k in kin_categories_list}
        else:
            if set(list(kin_categories_colors.keys())) < set(kin_categories_list):
                raise ValueError('Some families are missing from kin_categories_colors dictionary ({})'.format(list(set(kin_categories_list)-set(list(kin_categories_colors.keys())))))
        kin_colors = {}
        for kin,label_type in zip(kinome_data.index,kinome_data[label_info_col]):
            kin_colors[kin] = kin_categories_colors[label_type]
    
    sorted_lff_data = lff_data.loc[kins_order,cond_order]
    sorted_pval_data = pval_data.loc[kins_order,cond_order]
    if highlight_cont_kins:
        sorted_highlight_data = cont_kins.loc[kins_order,cond_order]
    else:
        sorted_highlight_data = pd.DataFrame(False, index=sorted_lff_data.index, columns=sorted_lff_data.columns)

    sig_data = (abs(sorted_lff_data) >= sig_lff) & (sorted_pval_data <= sig_pval)
    sig_lff_data = sorted_lff_data.mask(~sig_data)
    sig_pval_data = sorted_pval_data.mask(~sig_data)
    if not plot_cont_kins:
        sig_lff_data = sig_lff_data.mask(cont_kins)
        sig_pval_data = sig_pval_data.mask(cont_kins)
    
    if only_sig_kins:
        sig_kins = list(sig_data.loc[sig_data.any(axis=1)].index)
        sig_lff_data = sig_lff_data.loc[sig_kins]
        sig_pval_data = sig_pval_data.loc[sig_kins]
        sorted_highlight_data = sorted_highlight_data.loc[sig_kins]
    if only_sig_conds:
        sig_conds = list(sig_data.loc[:,sig_data.any()].columns)
        sig_lff_data = sig_lff_data[sig_conds]
        sig_pval_data = sig_pval_data[sig_conds]
        sorted_highlight_data = sorted_highlight_data[sig_conds]

    minus_log10_pval_data = -np.log10(sig_pval_data.fillna(1))
    
    if kin_clust:
        if cluster_by is None:
            raise ValueError('If kin_clust is True, cluster_by must be specified.')
        else:
            exceptions.check_cluster_method(cluster_by)
            if cluster_by == 'lff':
                Z = hierarchy.linkage(sig_lff_data.fillna(0).values, method=cluster_method)
            elif cluster_by == 'pval':
                Z = hierarchy.linkage(minus_log10_pval_data.fillna(0).values, method=cluster_method)
            elif cluster_by == 'both':
                Z = hierarchy.linkage((np.sign(sig_lff_data.fillna(0))*minus_log10_pval_data.fillna(0)).values, method=cluster_method)
            elif cluster_by == 'custom':
                if cluster_by_matrix is None:
                    raise ValueError('cluster_by_matrix must be specified.')
                Z = hierarchy.linkage(cluster_by_matrix.values, method=cluster_method)
            sig_lff_data = sig_lff_data.iloc[hierarchy.leaves_list(Z)]
            minus_log10_pval_data = minus_log10_pval_data.iloc[hierarchy.leaves_list(Z)]
            sorted_highlight_data = sorted_highlight_data.iloc[hierarchy.leaves_list(Z)]
    
    if condition_clust:
        if cluster_by is None:
            raise ValueError('If condition_clust is True, cluster_by must be specified.')
        else:
            exceptions.check_cluster_method(cluster_by)
            if cluster_by == 'lff':
                Z = hierarchy.linkage(sig_lff_data.fillna(0).transpose().values, method=cluster_method)
            elif cluster_by == 'pval':
                Z = hierarchy.linkage(minus_log10_pval_data.fillna(0).transpose().values, method=cluster_method)
            elif cluster_by == 'both':
                Z = hierarchy.linkage((np.sign(sig_lff_data.fillna(0))*minus_log10_pval_data.fillna(0)).transpose().values, method=cluster_method)
            elif cluster_by == 'custom':
                if cluster_by_matrix is None:
                    raise ValueError('cluster_by_matrix must be specified.')
                Z = hierarchy.linkage(cluster_by_matrix.transpose().values, method=cluster_method)
            sig_lff_data = sig_lff_data.iloc[:,hierarchy.leaves_list(Z)]
            minus_log10_pval_data = minus_log10_pval_data.iloc[:,hierarchy.leaves_list(Z)]
            sorted_highlight_data = sorted_highlight_data.iloc[:,hierarchy.leaves_list(Z)]
    
    intensity_matrices = np.array_split(sig_lff_data, num_panels)
    size_matrices = np.array_split(minus_log10_pval_data, num_panels)
    highlight_matrices = np.array_split(sorted_highlight_data, num_panels)

    if lff_clim is None:
        lff_clim = (min(sig_lff_data.min().min(),0),max(sig_lff_data.max().max(),0))
    cnorm = mcol.TwoSlopeNorm(vmin=lff_clim[0], vcenter=0, vmax=lff_clim[1])
    
    if max_pval_size is None:
        max_pval_size = np.ceil(minus_log10_pval_data.max().max())
    if max_pval_size < -np.log10(sig_pval):
        raise ValueError('max_pval_size must be equal or greater than -log10(sig_pval).')
    pval_slim = (-np.log10(sig_pval), max_pval_size)
    
    if vertical:
        intensity_matrices = [mat.transpose() for mat in intensity_matrices]
        size_matrices = [mat.transpose() for mat in size_matrices]
        highlight_matrices = [mat.transpose() for mat in highlight_matrices]

    fig = plt.figure(constrained_layout=constrained_layout)
    subfigs = fig.subfigures(nrows=1, ncols=2, width_ratios=[10, 1])
    
    if not vertical:
        axes = subfigs[0].subplots(nrows=num_panels, ncols=1, squeeze=False).ravel()
    else:
        axes = subfigs[0].subplots(nrows=1, ncols=num_panels, squeeze=False).ravel()
    
    for idx in range(num_panels):
        intensity_matrix = intensity_matrices[idx].loc[:,::-1]
        size_matrix = size_matrices[idx].loc[:,::-1]
        highlight_matrix = highlight_matrices[idx].loc[:,::-1]

        ax = axes[idx]
        
        melt_lff = pd.melt(intensity_matrix, ignore_index=False, var_name='condition', value_name='lff').set_index('condition', append=True)[::-1]
        melt_lff.index.names = ['kinase','condition']
        melt_pval = pd.melt(size_matrix, ignore_index=False, var_name='condition', value_name='pval').set_index('condition', append=True)[::-1]
        melt_pval.index.names = ['kinase','condition']
        melt_highlight = pd.melt(highlight_matrix, ignore_index=False, var_name='condition', value_name='highlight').set_index('condition', append=True)[::-1]
        melt_highlight.index.names = ['kinase','condition']
        lff_pval_data = pd.concat([melt_lff, melt_pval, melt_highlight], axis=1).reset_index()
        sig_lff_pval_data = lff_pval_data[~lff_pval_data['lff'].isna() & ~lff_pval_data['pval'].isna()]

        sns.scatterplot(x='kinase', y='condition', data=lff_pval_data, legend=False, ax=ax, alpha=0, size=0)
        sns.scatterplot(x='kinase', y='condition', data=sig_lff_pval_data, hue='lff', hue_norm=lff_clim, palette='coolwarm', size='pval', sizes=bubblesize_range, size_norm=pval_slim, linewidth=1, edgecolor='black', legend=False, ax=ax)
        if not sig_lff_pval_data[sig_lff_pval_data['highlight']].empty:
            sns.scatterplot(x='kinase', y='condition', data=sig_lff_pval_data[sig_lff_pval_data['highlight']], hue='lff', hue_norm=lff_clim, palette='coolwarm', size='pval', sizes=bubblesize_range, size_norm=pval_slim, linewidth=1, edgecolor='yellow', legend=False, ax=ax)

        minx = ax.get_xticks()[0]
        maxx = ax.get_xticks()[-1]
        if len(ax.get_xticks()) == 1:
            eps = 0.5
        else:
            eps = ((maxx - minx) / (len(ax.get_xticks()) - 1)) / 2
        ax.set_xlim(maxx+eps, minx-eps)
        miny = ax.get_yticks()[0]
        maxy = ax.get_yticks()[-1]
        if len(ax.get_yticks()) == 1:
            eps = 0.5
        else:
            eps = ((maxy - miny) / (len(ax.get_yticks()) - 1)) / 2
        ax.set_ylim(maxy+eps, miny-eps)
        
        ax.grid(which='major', color='k', linestyle=':')
        ax.set_axisbelow(True)
        ax.set_aspect('equal', 'box')
        ax.tick_params(axis='x', which='major', labelsize=xlabels_size, labelrotation=90)
        ax.tick_params(axis='y', which='major', labelsize=ylabels_size)
        
    fig.canvas.draw()
    axes = subfigs[0].axes
    for ax in axes:
        if vertical:
            ax.set_xlabel(xaxis_label, fontsize=14, fontweight='bold')
            ax.set_ylabel(yaxis_label, fontsize=14, fontweight='bold')
        else:
            ax.set_xlabel(yaxis_label, fontsize=14, fontweight='bold')
            ax.set_ylabel(xaxis_label, fontsize=14, fontweight='bold')
        if xlabel:
            ax.xaxis.set_major_locator(mticker.FixedLocator(ax.get_xticks()))
            ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
        else:
            ax.set(xlabel=None)
        if ylabel:
            ax.yaxis.set_major_locator(mticker.FixedLocator(ax.get_yticks()))
            ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
        else:
            ax.set(ylabel=None)
        if color_kins_by:
            if vertical:
                for l in ax.get_yticklabels():
                    label_color = kin_colors[l.get_text()]
                    l.set_color(label_color)
            else:
                for l in ax.get_xticklabels():
                    label_color = kin_colors[l.get_text()]
                    l.set_color(label_color)
        if cond_colors:
            if vertical:
                for l in ax.get_xticklabels():
                    label_color = cond_colors[l.get_text()]
                    l.set_color(label_color)
            else:
                for l in ax.get_yticklabels():
                    label_color = cond_colors[l.get_text()]
                    l.set_color(label_color)
    
    axes = subfigs[1].subplots(nrows=3, ncols=1, squeeze=False).ravel()
    
    if family_legned and color_kins_by:
        patches = [mpatches.Patch(color=x[1], label=x[0]) for x in kin_categories_colors.items()]
        axes[0].legend(handles=patches, loc='center', title='Family', facecolor='white')
    axes[0].axis('off')
    
    if pval_legend:
        size_legend_data = pd.DataFrame({'x': 0, 'y': 0, 'sizes': np.linspace(pval_slim[0], pval_slim[1], 4, dtype=int)})
        sns.scatterplot(x='x', y='y', data=size_legend_data, size='sizes', sizes=bubblesize_range, size_norm=pval_slim,
                        alpha=0, ax=axes[1])
        handles,labels  =  axes[1].get_legend_handles_labels()
        for i in range(len(handles)):
            handles[i].set_markeredgecolor('black')
            handles[i].set_markeredgewidth(1)
            handles[i].set_markerfacecolor('white')
            handles[i].set_alpha(1)
        axes[1].legend(handles=handles, labels=[str(10**-x) for x in size_legend_data['sizes'][:3]] + ['<'+str(10**int(-(size_legend_data['sizes'][3])))],
                       title='Adj. p-value', loc='center', labelspacing=pval_legend_spacing, facecolor='white')
    axes[1].axis('off')
    
    if lff_cbar:
        axes[2].set(aspect=10)
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=cnorm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=axes[2])
        cbar.ax.set_title('log2(FF)')
    else:
        axes[2].axis('off')
    
    subfigs[0].suptitle(title, fontsize=16)
    if max_window:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
    
    if save_fig:
        fig.savefig(save_fig, dpi=1000)
