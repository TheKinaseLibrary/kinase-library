"""
########################################################
# The Kinase Library - Motif Enrichment Analysis (MEA) #
########################################################
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import gseapy as gp
tqdm.pandas()

from ..utils import _global_vars, exceptions, utils
from ..modules import data, enrichment
from ..objects import core
from ..objects import phosphoproteomics as pps
from ..logger import logger

#%%

class RankedPhosData(object):
    """
    Class for ranked phosphorylation data.

    Parameters
    ----------
    """

    def __init__(self, dp_data, rank_col,
                 seq_col=None,
                 subs_pad=False, pp=False,
                 drop_invalid_subs=True,
                 new_seq_phos_res_cols=True,
                 suppress_warnings=False):
        
        if seq_col is None:
            seq_col = _global_vars.default_seq_col
        
        self.dp_data = dp_data
        self.rank_col = rank_col
        self.dp_data_pps = pps.PhosphoProteomics(dp_data, seq_col=seq_col, pad=subs_pad, pp=pp, drop_invalid_subs=drop_invalid_subs, new_seq_phos_res_cols=new_seq_phos_res_cols, suppress_warnings=suppress_warnings)
        
    
    def submit_scores(self, kin_type, scores, suppress_messages=False):
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
        
        self.dp_data_pps.submit_scores(kin_type=kin_type, scores=scores, suppress_messages=suppress_messages)
        
        
    def submit_percentiles(self, kin_type, percentiles, phosprot_name=None, suppress_messages=False):
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
        
        if phosprot_name is None:
            phosprot_name = _global_vars.phosprot_name
        self.phosprot_name = phosprot_name
            
        self.dp_data_pps.submit_percentiles(kin_type=kin_type, percentiles=percentiles, phosprot_name=phosprot_name, suppress_messages=suppress_messages)
        
    
    def _create_kin_sub_sets(self, thresh, comp_direction):
        
        print('\nGenerating kinase-substartes sets')
        logger.info('Generating kinase-substartes sets')
        kin_sub_sets = enrichment.create_kin_sub_sets(data_values=self.data_kl_values, threshold=thresh, comp_direction=comp_direction)
        self.kin_sub_sets = kin_sub_sets
        
        return(kin_sub_sets)
    
    
    def mea(self, kin_type, kl_method, kl_thresh,
            kinases=None, non_canonical=False,
            rescore=False, weight=1,
            threads=4, min_size=1, max_size=100000,
            permutation_num=1000, seed=112123,
            gseapy_verbose=False):
        """
        Kinase enrichment analysis based on pre-ranked GSEA substrates list.
    
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
        rescore : bool, optional
            If True, Kinase Library scores or percentiles will be recalculated.
        **GSEApy parameters: weight, threads, min_size, max_size, permutation_num, seed, gseapy_verbose
    
        Returns
        -------
        enrichemnt_results : pd.DataFrame
            pd.Dataframe with results of MEA for the specified KL method and threshold.
        """
        
        exceptions.check_kl_method(kl_method)

        if kinases is None:
            kinases = data.get_kinase_list(kin_type, non_canonical=non_canonical)
        elif isinstance(kinases, str):
            kinases = [kinases]
        kinases = [x.upper() for x in kinases]
        exceptions.check_kin_list_type(kinases, kin_type=kin_type)
        
        data_att = kl_method+'s'
        kl_comp_direction = _global_vars.kl_method_comp_direction_dict[kl_method]

        if kl_method in ['score','score_rank']:
            if not hasattr(self.dp_data_pps, kin_type + '_' + data_att) or rescore:
                print('\nCalculating scores for data')
                logger.info('Calculating scores for data')
                self.dp_data_pps.score(kin_type=kin_type,kinases=kinases)
            elif not set(kinases)<=set(getattr(self.dp_data_pps, kin_type + '_' + data_att)):
                print('Not all kinase scores were provided. Re-calculating scores for data')
                self.dp_data_pps.score(kin_type=kin_type,kinases=kinases)
        elif kl_method in ['percentile','percentile_rank']:
            if not hasattr(self.dp_data_pps, kin_type + '_' + data_att) or rescore:
                print('\nCalculating percentiles for data')
                logger.info('Calculating percentiles for data')
                self.dp_data_pps.percentile(kin_type=kin_type,kinases=kinases)
                self.phosprot_name = _global_vars.phosprot_name
            elif not set(kinases)<=set(getattr(self.dp_data_pps, kin_type + '_' + data_att)):
                print('Not all kinase percentiles were provided. Re-calculating percentiles for data')
                self.dp_data_pps.percentile(kin_type=kin_type,kinases=kinases)
                self.phosprot_name = _global_vars.phosprot_name
        
        self.data_kl_values = getattr(self.dp_data_pps, kin_type + '_' + data_att)
        
        kin_sub_sets = self._create_kin_sub_sets(thresh=kl_thresh, comp_direction=kl_comp_direction)

        ranked_subs = self.dp_data_pps.data.set_index(_global_vars.default_seq_col)[self.rank_col].sort_values(ascending=False)
        
        prerank_results = gp.prerank(rnk=ranked_subs,
                             gene_sets=kin_sub_sets,
                             weight=weight,
                             threads=threads,
                             min_size=min_size,
                             max_size=max_size,
                             permutation_num=permutation_num,
                             seed=seed,
                             verbose=gseapy_verbose)
        
        res_col_converter = {'Term': 'Kinase', 'ES': 'ES', 'NES': 'NES', 'NOM p-val': 'p-value', 'FDR q-val': 'FDR', 'Tag %': 'Subs fraction', 'Lead_genes': 'Leading substrates'}

        enrichment_data = prerank_results.res2d.drop(['Name', 'FWER p-val', 'Gene %'], axis=1).rename(columns=res_col_converter)
        enrichment_data['p-value'] = enrichment_data['p-value'].replace(0,1/permutation_num).astype(float) #Setting p-value of zero to 1/(# of permutations)
        enrichment_data['FDR'] = enrichment_data['FDR'].replace(0,enrichment_data['FDR'][enrichment_data['FDR'] != 0].min()).astype(float) #Setting FDR of zero to lowest FDR in data
        sorted_enrichment_data = enrichment_data.sort_values('Kinase').set_index('Kinase').reindex(data.get_kinase_list(kin_type, non_canonical=non_canonical))
        
        enrichemnt_results = MeaEnrichmentResults(enrichment_results=sorted_enrichment_data, pps_data=self, gseapy_obj=prerank_results,
                                                   kin_type=kin_type, kl_method=kl_method, kl_thresh=kl_thresh, tested_kins=kinases,
                                                   data_att=data_att, kl_comp_direction=kl_comp_direction)
        
        return enrichemnt_results

#%%

class MeaEnrichmentResults(object):
    """
    Class for continuous kinase enrichment results using GSEA (weighted Kolmogorov–Smirnov statistic).

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
    
    def __init__(self, enrichment_results, pps_data, gseapy_obj,
                 kin_type, kl_method, kl_thresh, tested_kins,
                 data_att, kl_comp_direction):
        
        self.enrichment_results = enrichment_results
        self.pps_data = pps_data
        self.gseapy_obj = gseapy_obj
        self.kin_type = kin_type
        self.kl_method = kl_method
        self.kl_thresh = kl_thresh
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
            data_columns = getattr(self.pps_data.dp_data_pps, self.kin_type+'_data').columns.to_list()

        if save_to_excel:
            if output_dir is None:
                raise ValueError('Please provide output directory.')
            output_dir = output_dir.rstrip('/')
            os.makedirs(output_dir, exist_ok=True)
            
            if file_prefix is not None:
                writer = pd.ExcelWriter(f'{output_dir}/enriched_subs/{file_prefix}_{self.kin_type}_{self._data_att}_thresh_{self.kl_thresh}.xlsx')
            else:
                writer = pd.ExcelWriter(f'{output_dir}/enriched_subs/{self.kin_type}_{self._data_att}_thresh_{self.kl_thresh}.xlsx')
        score_data = self.pps_data.dp_data_pps.merge_data_scores(self.kin_type, self._data_att)

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
    
    
    def activated_kins(self, sig_es=0, sig_pval=0.1, adj_pval=True, norm_es=True):
        """
        Returns a list of all kinases with significant p-value/FDR and ES/NES above threshold.

        Parameters
        ----------
        sig_es : float, optional
            Threshold to determine the enrichment score cutoff of kinase enrichment. The default is 0.
        sig_pval : float, optional
            Threshold to determine the p-value significance of kinase enrichment. The default is 0.1 (adjusted p-values).
        adj_pval : bool, optional
            If True use adjusted p-value for calculation of statistical significance. The default is True.
        norm_es : bool, optional
            If True use NES for calculation of statistical significance. The default is True.
           
        Returns
        -------
        activated_kins : list
            List of kinases enriched with significant p-value/FDR and ES/NES above threshold.
        """
        
        if sig_es<0:
            raise ValueError('sig_es must be zero or positive.')
        
        if adj_pval:
            pval_col = 'FDR'
        else:
            pval_col = 'p-value'
        
        if norm_es:
            es_col = 'NES'
        else:
            es_col = 'ES'
        
        activated_kins = list(self.enrichment_results[(st_kin_enrich.enrichment_results[pval_col] <= sig_pval) & (self.enrichment_results[es_col] >= sig_es)].index)
        
        return activated_kins
    
    
    def inhibited_kins(self, sig_es=0, sig_pval=0.1, adj_pval=True):
        """
        Returns a list of all kinases with significant p-value/FDR and ES/NES below negative threshold.

        Parameters
        ----------
        sig_es : float, optional
            Threshold to determine the enrichment score cutoff of kinase enrichment. The default is 0.
        sig_pval : float, optional
            Threshold to determine the p-value significance of kinase enrichment. The default is 0.1 (adjusted p-values).
        adj_pval : bool, optional
            If True use adjusted p-value for calculation of statistical significance. The default is True.
        norm_es : bool, optional
            If True use NES for calculation of statistical significance. The default is True.
           
        Returns
        -------
        inhibited_kins : list
            List of kinases enriched with significant p-value/FDR and ES/NES below negative threshold.
        """
        
        if sig_es<0:
            raise ValueError('sig_es must be zero or positive (for inhibited kinases: -sig_es will be used).')
            
        if adj_pval:
            pval_col = 'FDR'
        else:
            pval_col = 'p-value'
        
        if norm_es:
            es_col = 'NES'
        else:
            es_col = 'ES'
        
        inhibited_kins = list(self.enrichment_results[(st_kin_enrich.enrichment_results[pval_col] <= sig_pval) & (self.enrichment_results[es_col] >= -sig_es)].index)
        
        return inhibited_kins
        
        
    def plot_volcano(self, sig_es=0, sig_pval=0.1, adj_pval=True, kinases=None,
                     es_col='NES', pval_col=None, highlight_kins=None,
                     kins_label_dict=None, label_kins=None, adjust_labels=True, labels_fontsize=7,
                     symmetric_xaxis=True, grid=True, max_window=False,
                     title=None, stats=True, xlabel='NES', ylabel=None,
                     plot=True, save_fig=False, return_fig=False,
                     ax=None, **plot_kwargs):
        """
        Returns a volcano plot of the Kinase Library enrichment results.

        Parameters
        ----------
        sig_es : float, optional
            Significance threshold for logFF in the enrichment results. The default is 0.
        sig_pval : float, optional
            Significance threshold for and adjusted p-value in the enrichment results. The default is 0.1.
        fg_percent_thresh : float, optional
            Minimun percentage of foreground substrates mapped to a kinase in order to plot it in the volcano.
        fg_percent_col : str, optional
            Column name of percentage of foreground substrates mapped to each kinase.
        kinases : list, optional
            If provided, kinases to plot in the volcano plot. The default is None.
        es_col : str, optional
            Enrichment score column name used for volcano plot. The default is 'NES'.
        pval_col : str, optional
            P-value column name used for volcano plot. The defulat is None and will be determined based on self.adj_pval.
        highlight_kins : list, optional
            List of kinases to be marked in yellow on the kinase enrichment volcano plot.
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
            x-axis label for the volcano plot. The default is 'NES'.
        ylabel : str, optional
            y-axis label for the volcano plot. The default is '-log$_{10}$(FDR)'.
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
        if pval_col is None:
            if adj_pval:
                pval_col='FDR'
            else:
                pval_col='p-value'
        
        if ylabel is None:
            ylabel='-log$_{10}$('+pval_col+')'
        
        if kinases is None:
            kinases = self.tested_kins

        if kins_label_dict:
            kinases = [kins_label_dict[x] for x in kinases]
            return enrichment.plot_volcano(self.enrichment_results.rename(kins_label_dict), sig_lff=sig_es, sig_pval=sig_pval, kinases=kinases,
                                           lff_col=es_col, pval_col=pval_col, highlight_kins=highlight_kins, ignore_depleted=ignore_depleted,
                                           label_kins=label_kins, adjust_labels=adjust_labels, labels_fontsize=labels_fontsize,
                                           symmetric_xaxis=symmetric_xaxis, grid=grid, max_window=max_window,
                                           title=title, xlabel=xlabel, ylabel=ylabel,
                                           plot=plot, save_fig=save_fig, return_fig=return_fig,
                                           ax=ax, **plot_kwargs)
        else:
            return enrichment.plot_volcano(self.enrichment_results, sig_lff=sig_es, sig_pval=sig_pval, kinases=kinases,
                                           lff_col=es_col, pval_col=pval_col, highlight_kins=highlight_kins,
                                           label_kins=label_kins, adjust_labels=adjust_labels, labels_fontsize=labels_fontsize,
                                           symmetric_xaxis=symmetric_xaxis, grid=grid, max_window=max_window,
                                           title=title, xlabel=xlabel, ylabel=ylabel,
                                           plot=plot, save_fig=save_fig, return_fig=return_fig,
                                           ax=ax, **plot_kwargs)
