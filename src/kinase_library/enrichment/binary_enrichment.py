"""
##########################################
# The Kinase Library - Binary Enrichment #
##########################################
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.stats import multitest
from tqdm import tqdm
tqdm.pandas()

from ..utils import _global_vars, exceptions, utils
from ..modules import data, enrichment
from ..objects import core
from ..objects import phosphoproteomics as pps
from ..logger import logger

#%%

class EnrichmentData(object):
    """
    Class for binary kinase enrichment data using Fisher's exact test.

    Parameters
    ----------
    foreground : pd.DataFrame
        Dataframe with foreground substrates.
    background : pd.DataFrame
        Dataframe with background substrates.
    fg_seq_col : str, optional
        Substrates column name for the foreground data. The default is None (will be set as _global_vars.default_seq_col).
    bg_seq_col : str, optional
        Substrates column name for the background data. The default is None (will be set as _global_vars.default_seq_col).
    fg_pad : tuple, optional
        How many padding '_' to add from each side of the substrates from the foreground data. The default is False.
    bg_pad : tuple, optional
        How many padding '_' to add from each side of the substrates from the background data. The default is False.
    fg_pp : bool, optional
        Phospho-residues in the foreground (s/t/y, phospho-residues in the sequence). The default is False.
    bg_pp : bool, optional
        Phospho-residues in the background (s/t/y, phospho-residues in the sequence). The default is False.
    drop_invalid_subs : bool, optional
        Drop rows with invalid substrates. The default is True.
    new_seq_phos_res_cols : bool, optional
        Create a new sequence column or phosphorylated residue column even if they already exists. The default is True.
        Sequence column name: _global_vars.default_seq_col. Phosphorylated residue column name: 'phos_res'.
    suppress_warnings : bool, optional
        Do not print warnings. The default is False.
    """

    def __init__(self, foreground, background=None,
                 fg_seq_col=None, bg_seq_col=None,
                 fg_pad=False, bg_pad=False, fg_pp=False, bg_pp=False,
                 drop_invalid_subs=True,
                 new_seq_phos_res_cols=True,
                 suppress_warnings=False):
        
        if background is None:
            background = data.get_phosphoproteome()
        
        if isinstance(foreground, (pd.Series, list)):
            foreground = utils.list_series_to_df(foreground, col_name=fg_seq_col)
        if isinstance(background, (pd.Series, list)):
            background = utils.list_series_to_df(background, col_name=bg_seq_col)
        
        if fg_seq_col is None:
            fg_seq_col = _global_vars.default_seq_col
        if bg_seq_col is None:
            bg_seq_col = _global_vars.default_seq_col
        
        self.fg_data = foreground
        self.bg_data = background
        self.fg_pps = pps.PhosphoProteomics(foreground, seq_col=fg_seq_col, pad=fg_pad, pp=fg_pp, drop_invalid_subs=drop_invalid_subs, new_seq_phos_res_cols=new_seq_phos_res_cols, suppress_warnings=suppress_warnings)
        self.bg_pps = pps.PhosphoProteomics(background, seq_col=bg_seq_col, pad=bg_pad, pp=bg_pp, drop_invalid_subs=drop_invalid_subs, new_seq_phos_res_cols=new_seq_phos_res_cols, suppress_warnings=suppress_warnings)
        
    
    @staticmethod
    def _get_kinase_freq(scored_data, threshold, direction):
        """
        Returning the frequency of kinases based on scored data and threshold.

        Parameters
        ----------
        scored_data : pd.DataFrame
            Data frame containing scores, substrates as indices, and kinases as columns.
        threshold : float
            Prediction threshold value.
        direction : str
            Comparison direction: 'higher' or 'lower'.

        Returns
        -------
        Prediction frequency of every kinase in data.
        """
        
        if direction == 'higher':
            return((scored_data>=threshold).sum())
        elif direction == 'lower':
            return((scored_data<=threshold).sum())
        else:
            raise ValueError('\'direction\' must be either \'higher\' or \'lower\'')
    
    
    @staticmethod
    def _correct_contingency_table(cont_table, columns=None):
        """
        Applying Haldane correction (adding 0.5 to the cases with zero in one of the counts).
        Being used only for calculating log2(freq_factor), not for p-value.

        Parameters
        ----------
        cont_table : pd.DataFrame
            Dataframe of frequency values (each row is one contigency table).
            Must contain 4 columns, or a list of 4 columns must be provided.
        columns : list, optional
            List of columns to correct. Must contain 4 columns. The default is None.

        Returns
        -------
        corrected_cont_table : pd.DataFrame
            Corrected dataframe.
        """
        
        if columns is not None:
            if len(columns) != 4:
                print('Exactly 4 columns must be provided.')
        else:
            if cont_table.shape[1] != 4:
                print('Dataframe must have 4 columns (or 4 columns must be provided).')
            columns = cont_table.columns
        
        corrected_cont_table = cont_table[columns].astype(float)
        pd.options.mode.chained_assignment = None # Turning off pandas SettingWithCopyWarning
        corrected_cont_table[corrected_cont_table.min(axis=1) == 0] = corrected_cont_table[corrected_cont_table.min(axis=1) == 0] + 0.5 # Applying Haldane correction (adding 0.5 to the cases with zero in one of the counts) - being used only for calculating log2(freq_factor), not for p-value
        pd.options.mode.chained_assignment = 'warn' # Turning on pandas SettingWithCopyWarning
        
        return(corrected_cont_table)
    
    
    def submit_scores(self, kin_type, scores, data_type=['foreground','background'], suppress_messages=False):
        """
        Submitting scores for the foreground/background substrates.

        Parameters
        ----------
        kin_type : str
            Kinase type ('ser_thr' or 'tyrosine').
        scores : pd.DataFrame
            Dataframe with sites scores.
            Index must contain all the values in 'seq_col' with no duplicates.
            Columns must contain valid kinase names.
        data_type : str or list, optional
            Data type: foreground or background. The default is ['foreground','background'].
        suppress_messages : bool, optional
            Suppress messages. The default is False.

        Raises
        ------
        ValueError
            Raise error if data type is not valid.

        Returns
        -------
        None.
        """
        
        if isinstance(data_type, str):
            data_type = [data_type]
        
        for dt in data_type:
            exceptions.check_enrichment_data_type(dt)
            
            if dt in ['foreground','fg']:
                self.fg_pps.submit_scores(kin_type=kin_type, scores=scores, suppress_messages=suppress_messages)
            elif dt in ['background','bg']:
                self.bg_pps.submit_scores(kin_type=kin_type, scores=scores, suppress_messages=suppress_messages)
        
        
    def submit_percentiles(self, kin_type, percentiles, data_type=['foreground','background'], phosprot_name=None, suppress_messages=False):
        """
        Submitting percentiles for the foreground/background substrates.

        Parameters
        ----------
        kin_type : str
            Kinase type ('ser_thr' or 'tyrosine').
        percentiles : pd.DataFrame
            Dataframe with sites percentiles.
            Index must contain all the values in 'seq_col' with no duplicates.
            Columns must contain valid kinase names.
        data_type : str or list, optional
            Data type: foreground or background. The default is ['foreground','background'].
        phosprot_name : str, optional
            Name of phosphoproteome database.
        suppress_messages : bool, optional
            Suppress messages. the default is False.
        
        Raises
        ------
        ValueError
            Raise error if data type is not valid.
            
        Returns
        -------
        None.
        """
        
        if isinstance(data_type, str):
            data_type = [data_type]
        
        if phosprot_name is None:
            phosprot_name = _global_vars.phosprot_name
        self.phosprot_name = phosprot_name
            
        for dt in data_type:
            exceptions.check_enrichment_data_type(dt)
            
            if dt in ['foreground','fg']:
                self.fg_pps.submit_percentiles(kin_type=kin_type, percentiles=percentiles, phosprot_name=phosprot_name, suppress_messages=suppress_messages)
            elif dt in ['background','bg']:
                self.bg_pps.submit_percentiles(kin_type=kin_type, percentiles=percentiles, phosprot_name=phosprot_name, suppress_messages=suppress_messages)
        
        
    def kinase_enrichment(self, kin_type, kl_method, kl_thresh,
                          kinases=None, non_canonical=False,
                          enrichment_type='enriched', rescore=False):
        """
        Kinase enrichment analysis based on Fisher's exact test for foreground and background substarte lists.
    
        Parameters
        ----------
        kin_type : str
            Kinase type ('ser_thr' or 'tyrosine').
        kl_method : str
            Kinase Library scoring method ('score', 'score_rank', 'percentile', 'percentile_rank').
        kl_thresh : int
            The threshold to be used for the specified kl_method.
        kinases : list, optional
            If provided, kinase enrichment will only be calculated for the specified kinase list, otherwise, all kinases of the specified kin_type will be included. The default is None.
        non_canonical : bool, optional
            Return also non-canonical kinases. For tyrosine kinases only. The default is False.
        enrichment_type : str, optional
            Direction of fisher's exact test for kinase enrichment ('enriched','depleted', or 'both').
        rescore : bool, optional
            If True, Kinase Library scores or percentiles will be recalculated.
    
        Returns
        -------
        enrichemnt_results : pd.DataFrame
            pd.Dataframe with results of Kinase Enrichment for the specified KL method and threshold.
        """
        
        exceptions.check_kl_method(kl_method)
        exceptions.check_enrichment_type(enrichment_type)
        
        enrich_test_sides_dict = {'enriched': 'greater', 'depleted': 'less', 'both': 'two-sided'}
        test_alternative = enrich_test_sides_dict[enrichment_type]
        
        if kinases is None:
            kinases = data.get_kinase_list(kin_type, non_canonical=non_canonical)
        elif isinstance(kinases, str):
            kinases = [kinases]
        kinases = [x.upper() for x in kinases]
        exceptions.check_kin_list_type(kinases, kin_type=kin_type)
        
        data_att = kl_method+'s'
        kl_comp_direction = _global_vars.kl_method_comp_direction_dict[kl_method]
        
        if kl_method in ['score','score_rank']:
            if not hasattr(self.fg_pps, kin_type + '_' + data_att) or rescore:
                self.fg_pps.score(kin_type=kin_type,kinases=kinases)
            elif not set(kinases)<=set(getattr(self.fg_pps, kin_type + '_' + data_att)):
                print('Not all kinase scores were provided. Re-calculating scores for foreground data')
                self.fg_pps.score(kin_type=kin_type,kinases=kinases)
            if not hasattr(self.bg_pps, kin_type + '_' + data_att) or rescore:
                self.bg_pps.score(kin_type=kin_type,kinases=kinases)
            elif not set(kinases)<=set(getattr(self.bg_pps, kin_type + '_' + data_att)):
                print('Not all kinase scores were provided. Re-calculating scores for background data')
                self.bg_pps.score(kin_type=kin_type,kinases=kinases)
        elif kl_method in ['percentile','percentile_rank']:
            if not hasattr(self.fg_pps, kin_type + '_' + data_att) or rescore:
                print('\nCalculating percentiles for foreground data')
                self.fg_pps.percentile(kin_type=kin_type,kinases=kinases)
                self.phosprot_name = _global_vars.phosprot_name
            elif not set(kinases)<=set(getattr(self.fg_pps, kin_type + '_' + data_att)):
                print('Not all kinase percentiles were provided. Re-calculating percentiles for foreground data')
                self.fg_pps.percentile(kin_type=kin_type,kinases=kinases)
                self.phosprot_name = _global_vars.phosprot_name
            if not hasattr(self.bg_pps, kin_type + '_' + data_att) or rescore:
                print('\nCalculating percentiles for background data')
                self.bg_pps.percentile(kin_type=kin_type,kinases=kinases)
                self.phosprot_name = _global_vars.phosprot_name
            elif not set(kinases)<=set(getattr(self.bg_pps, kin_type + '_' + data_att)):
                print('Not all kinase percentiles were provided. Re-calculating percentiles for background data')
                self.bg_pps.percentile(kin_type=kin_type,kinases=kinases)
                self.phosprot_name = _global_vars.phosprot_name
        
        fg_score_data = getattr(self.fg_pps, kin_type + '_' + data_att)
        bg_score_data = getattr(self.bg_pps, kin_type + '_' + data_att)
        
        enrichment_data = pd.DataFrame(index = kinases, columns = ['fg_counts', 'fg_total',
                                                                   'bg_counts', 'bg_total',
                                                                   'fg_percent', 'log2_freq_factor',
                                                                   'fisher_pval', 'fisher_adj_pval'])
        enrichment_data['fg_total'] = fg_total = len(fg_score_data)
        enrichment_data['bg_total'] = bg_total = len(bg_score_data)
        enrichment_data['fg_counts'] = self._get_kinase_freq(fg_score_data, kl_thresh, kl_comp_direction)
        enrichment_data['bg_counts'] = self._get_kinase_freq(bg_score_data, kl_thresh, kl_comp_direction)
        
        enrichment_data['fg_percent'] = (enrichment_data['fg_counts']/enrichment_data['fg_total']*100).fillna(0)
        
        enrichment_contingency_table = pd.DataFrame({'fg_pos': enrichment_data['fg_counts'],
                                                     'fg_neg': enrichment_data['fg_total'] - enrichment_data['fg_counts'],
                                                     'bg_pos': enrichment_data['bg_counts'],
                                                     'bg_neg': enrichment_data['bg_total'] - enrichment_data['bg_counts']})

        corrected_enrichment_contingency_table = self._correct_contingency_table(enrichment_contingency_table)
        
        enrichment_data['log2_freq_factor'] = np.log2((corrected_enrichment_contingency_table['fg_pos'] / (corrected_enrichment_contingency_table['fg_pos'] + corrected_enrichment_contingency_table['fg_neg'])) / (corrected_enrichment_contingency_table['bg_pos'] / (corrected_enrichment_contingency_table['bg_pos'] + corrected_enrichment_contingency_table['bg_neg'])))
    
        fisher_pvals = []
        for fg_counts,bg_counts in zip(enrichment_data['fg_counts'],enrichment_data['bg_counts']):
            fisher_pvals.append(st.fisher_exact([[fg_counts, fg_total - fg_counts], [bg_counts, bg_total - bg_counts]],
                                                alternative=test_alternative)[1])
        enrichment_data['fisher_pval'] = fisher_pvals
        enrichment_data['fisher_adj_pval'] = multitest.multipletests(fisher_pvals, method = 'fdr_bh')[1]

        enrichemnt_results = EnrichmentResults(enrichment_results=enrichment_data, pps_data=self, kin_type=kin_type,
                                               kl_method=kl_method, kl_thresh=kl_thresh, enrichment_type=enrichment_type,
                                               tested_kins=kinases, data_att=data_att, kl_comp_direction=kl_comp_direction)
        
        return enrichemnt_results

#%%

class EnrichmentResults(object):
    """
    Class for binary kinase enrichment results using Fisher's exact test.

    Parameters
    ----------
    enrichment_results : pd.DataFrame
        Dataframe containing Kinase Library enrichment results.
    pps_data : kl.EnrichmentData
        Object initialized from the foreground and background dataframes used to calculate provided enrichment_results.
    kin_type : str
        Kinase type ('ser_thr' or 'tyrosine').
    kl_method : str
        Kinase Library scoring method ('score', 'score_rank', 'percentile', 'percentile_rank').
    kl_thresh : int
        The threshold to be used for the specified kl_method.
    enrichment_type : str
        Direction of fisher's exact test for kinase enrichment ('enriched','depleted', or 'both').
    tested_kins : list
        List of kinases included in the Kinase Library enrichment.
    data_att : str
        Score type ('scores', 'score_ranks', 'percentiles', 'percentile_ranks').
    kl_comp_direction : str
        Dictates if kinases above or below the specified threshold are used ('higher','lower').
    """
    
    def __init__(self, enrichment_results, pps_data,
                 kin_type, kl_method, kl_thresh, enrichment_type,
                 tested_kins, data_att, kl_comp_direction):
        
        self.enrichment_results = enrichment_results
        self.pps_data = pps_data
        self.kin_type = kin_type
        self.kl_method = kl_method
        self.kl_thresh = kl_thresh
        self.enrichment_type = enrichment_type
        self.tested_kins = tested_kins
        self._data_att = data_att
        self._kl_comp_direction = kl_comp_direction
        
        if kl_method in ['percentile','percentile_rank']:
            self.phosprot_name = pps_data.phosprot_name
        
        
    def enriched_subs(self, kinases, data_columns=None, as_dataframe=False,
                      save_to_excel=False, output_dir=None, file_prefix=None):
        """
        Function to save an excel file containing the subset of substrates that drove specific kinases' enrichment.

        Parameters
        ----------
        kinases : list
            List of kinases for enriched substrates. Substrates provided are those that drove that kinase's enrichment.
        data_columns : list, optional
            Columns from original data to be included with each enriched substrate. Defaults to None, including all original columns.
        as_dataframe : bool, optional
            If true, return dataframe instead of dictionary. Relevant only if one kinase is provided. Default is False.
        save_to_excel : bool, optional
            If True, excel file containing enriched substrates will be saved to the output_dir.
        output_dir : str, optional
            Location for enriched substrates excel file to be saved.
        file_prefix : str, optional
            Prefix for the files name.
           
        Returns
        -------
        enrich_subs_dict : dict
            Dictionary with the substrates that drove enrichment for each kinase.
        """
        
        if kinases == []:
            print('No kinases provided.')
            return({})
        
        if isinstance(kinases, str):
            kinases = [kinases]
        kinases = [x.upper() for x in kinases]
        if not (set(kinases) <= set(self.tested_kins)):
            missing_kinases = list(set(kinases) - set(self.tested_kins))
            raise ValueError(f'Certain kinases are not in the enrichment results ({missing_kinases}).')
        
        if data_columns is None:
            data_columns = getattr(self.pps_data.fg_pps, self.kin_type+'_data').columns.to_list()

        if save_to_excel:
            if output_dir is None:
                raise ValueError('Please provide output directory.')
            output_dir = output_dir.rstrip('/')
            os.makedirs(output_dir, exist_ok=True)
            
            if file_prefix is not None:
                writer = pd.ExcelWriter(f'{output_dir}/{file_prefix}_{self.kin_type}_{self._data_att}_thresh_{self.kl_thresh}.xlsx')
            else:
                writer = pd.ExcelWriter(f'{output_dir}/{self.kin_type}_{self._data_att}_thresh_{self.kl_thresh}.xlsx')
        score_data = self.pps_data.fg_pps.merge_data_scores(self.kin_type, self._data_att)

        enrich_subs_dict = {}
        for kin in kinases:
            if self._kl_comp_direction == 'higher':
                enriched_kin_subs = score_data[score_data[kin] >= self.kl_thresh][data_columns + [kin]].sort_values(by=kin, ascending=False)
            elif self._kl_comp_direction == 'lower':
                enriched_kin_subs = score_data[score_data[kin] <= self.kl_thresh][data_columns + [kin]].sort_values(by=kin, ascending=True)
            
            enrich_subs_dict[kin] = enriched_kin_subs
            
            if save_to_excel:
                enriched_kin_subs.to_excel(writer, sheet_name=kin, index=False)

        if save_to_excel:
            writer.close()
        
        if len(kinases)==1 and as_dataframe:
            return(enriched_kin_subs)
        
        return(enrich_subs_dict)
    
    
    def enriched_kins(self, sig_lff=0, sig_pval=0.1, adj_pval=True):
        """
        Returns a list of all kinases enriched above a p-value and frequency-factor thresholds.

        Parameters
        ----------
        sig_lff : float, optional
            Threshold to determine the Log frequency factor cutoff of kinase enrichment. The default is 0.
        sig_pval : float, optional
            Threshold to determine the p-value significance of kinase enrichment. The default is 0.1 (adjusted p-values).
        adj_pval : bool, optional
            If True use adjusted p-value for calculation of statistical significance. The default is True.
           
        Returns
        -------
        enriched_kins : pd.DataFrame
            List of kinases enriched according to the designated p-value and freq_facor thresholds.
        """
        
        if adj_pval:
            pval_col = 'fisher_adj_pval'
        else:
            pval_col = 'fisher_pval'
                
        enriched_kins = list(self.enrichment_results[(self.enrichment_results[pval_col] <= sig_pval) & (self.enrichment_results['log2_freq_factor'] >= sig_lff)].index)
        return enriched_kins
    
    
    def depleted_kins(self, sig_lff=0, sig_pval=0.1, adj_pval=True):
        """
        Returns a list of all kinases depleted above a p-value and frequency-factor thresholds.

        Parameters
        ----------
        sig_lff : float, optional
            Threshold to determine the Log frequency factor cutoff of kinase enrichment. The default is 0.
        sig_pval : float, optional
            Threshold to determine the p-value significance of kinase enrichment. The default is 0.1 (adjusted p-values).
        adj_pval : bool, optional
            If True use adjusted p-value for calculation of statistical significance. The default is True.
           
        Returns
        -------
        depleted_kins : pd.DataFrame
            List of kinases depleted according to the designated p-value and freq_facor thresholds.
        """
        
        if adj_pval:
            pval_col = 'fisher_adj_pval'
        else:
            pval_col = 'fisher_pval'
                
        depleted_kins = list(self.enrichment_results[(self.enrichment_results[pval_col] <= sig_pval) & (self.enrichment_results['log2_freq_factor'] <= -sig_lff)].index)
        return depleted_kins
        
        
    def plot_volcano(self, sig_lff=0, sig_pval=0.1, fg_percent_thresh=0, fg_percent_col='fg_percent', kinases=None,
                     lff_col='log2_freq_factor', adj_pval=True, highlight_kins=None, ignore_depleted=False,
                     kins_label_dict=None, label_kins=None, adjust_labels=True, labels_fontsize=7,
                     symmetric_xaxis=True, grid=True, max_window=False,
                     title=None, stats=True, xlabel='log$_2$(Frequency Factor)', ylabel=None,
                     plot=True, save_fig=False, return_fig=False,
                     ax=None, **plot_kwargs):
        """
        Returns a volcano plot of the Kinase Library enrichment results.

        Parameters
        ----------
        sig_lff : float, optional
            Significance threshold for logFF in the enrichment results. The default is 0.
        sig_pval : float, optional
            Significance threshold for and adjusted p-value in the enrichment results. The default is 0.1.
        fg_percent_thresh : float, optional
            Minimun percentage of foreground substrates mapped to a kinase in order to plot it in the volcano.
        fg_percent_col : str, optional
            Column name of percentage of foreground substrates mapped to each kinase.
        kinases : list, optional
            If provided, kinases to plot in the volcano plot. The default is None.
        lff_col : str, optional
            Log frequency factor column name used for volcano plot. The default is 'log2_freq_factor'.
        adj_pval : bool, optional
            If True use adjusted p-value for calculation of statistical significance. The default is True.
        highlight_kins : list, optional
            List of kinases to be marked in yellow on the kinase enrichment volcano plot.
        ignore_depleted : bool, optional
            Ignore kinases that their FF is negative (depleted). The default is False.
        kins_label_dict : dict, optional
            Dictionary with customized labels for each kinase. The default is None.
        label_kins : list, optional
            List of kinases to label on volcano plot. The default is None.
            If none, all significant kinases will be labelled plus any non-significant kinases marked for highlighting.
        adjust_labels : bool, optional
            If True, labels will be adjusted to avoid other markers and text on volcano plot. The default is True.
        symmetric_xaxis : bool, optional
            If True, x-axis will be made symmetric to its maximum absolute value. The default is True.
        grid : bool, optional
            If True, a grid is provided on the enrichment results volcano plot. The default is True.
        max_window : bool, optional
            For plotting and data visualization purposes; if True, plotting window will be maximized. The default is False.
            Must be False if an axis is provided to the function.
        title : str, optional
            Title for the volcano plot. The default is False.
        stats : bool, optional
            Plotting DE stats in the title. The default is True.
        xlabel : str, optional
            x-axis label for the volcano plot. The default is 'log$_2$(Frequency Factor)'.
        ylabel : str, optional
            y-axis label for the volcano plot. The default is determined based on the adjusted p-value status.
        plot : bool, optional
            Whether or not to plot the produced enrichment volcano plot. The default is True.
            Will be automatically changed to False if an axis is provided.
        save_fig : str, optional
            Path to file for saving the volcano plot. The default is False.
            Must be False if an axis is provided.
        return_fig : bool, optional
            If true, the volcano plot will be returned as a plt.figure object. The default is False.
        ax : plt.axes, optional
            Axes provided to plot the kinase enrichment volcano onto. The default is None.
        **plot_kwargs : optional
            Optional keyword arguments to be passed to the plot_volcano function.

        Returns
        -------
        If return_fig, the kinase enrichment volcano plot.
        """

        if stats:
            fg_size = len(getattr(self.pps_data.fg_pps, self.kin_type+'_'+self.kl_method+'s'))
            bg_size = len(getattr(self.pps_data.bg_pps, self.kin_type+'_'+self.kl_method+'s'))
            title = '\n'.join(filter(None, [title,f'Foreground: {fg_size}; Background: {bg_size}']))
        
        if adj_pval:
            pval_col = 'fisher_adj_pval'
            if ylabel is None:
                ylabel = '-log$_{10}$(Adjusted p-value)'
        else:
            pval_col = 'fisher_pval'
            if ylabel is None:
                ylabel = '-log$_{10}$(Nominal p-value)'
        
        fg_percent_kins = self.enrichment_results[self.enrichment_results[fg_percent_col] >= fg_percent_thresh].index.to_list()
        if kinases is not None:
            kinases = [x for x in kinases if x in fg_percent_kins]
        else:
            kinases = fg_percent_kins

        if kins_label_dict:
            kinases = [kins_label_dict[x] for x in kinases]
            return enrichment.plot_volcano(self.enrichment_results.rename(kins_label_dict), sig_lff=sig_lff, sig_pval=sig_pval, kinases=kinases,
                                           lff_col=lff_col, pval_col=pval_col, highlight_kins=highlight_kins, ignore_depleted=ignore_depleted,
                                           label_kins=label_kins, adjust_labels=adjust_labels, labels_fontsize=labels_fontsize,
                                           symmetric_xaxis=symmetric_xaxis, grid=grid, max_window=max_window,
                                           title=title, xlabel=xlabel, ylabel=ylabel,
                                            plot=plot, save_fig=save_fig, return_fig=return_fig,
                                            ax=ax, **plot_kwargs)
        else:
            return enrichment.plot_volcano(self.enrichment_results, sig_lff=sig_lff, sig_pval=sig_pval, kinases=kinases,
                                           lff_col=lff_col, pval_col=pval_col, highlight_kins=highlight_kins, ignore_depleted=ignore_depleted,
                                           label_kins=label_kins, adjust_labels=adjust_labels, labels_fontsize=labels_fontsize,
                                           symmetric_xaxis=symmetric_xaxis, grid=grid, max_window=max_window,
                                           title=title, xlabel=xlabel, ylabel=ylabel,
                                           plot=plot, save_fig=save_fig, return_fig=return_fig,
                                           ax=ax, **plot_kwargs)