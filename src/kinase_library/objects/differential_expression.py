"""
####################################################
# Kinase Library - Differential Expression Objects #
####################################################
"""

from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import pandas as pd

from ..utils import _global_vars, exceptions
from ..modules import enrichment
from . import phosphoproteomics as pps
tqdm.pandas()

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

#%%
class DiffExpData(object):
    """
    Class for differential expression data.

    Parameters
    ----------
    de_data : pd.DataFrame
        DataFrame containing differential expression data (must include sequence and logFC columns).
    kin_type : str
        Kinase type ('ser_thr' or 'tyrosine').
    lfc_col : str
        LogFC column name of the differential expression analysis.
    lfc_thresh : float, optional
        LogFC cuttoff used to define up, down, and unregulated sites.
    pval_col : str, optional
        P-value (or adjusted p-value) column name of the differential expression analysis.
    pval_thresh : float, optional
        Significance threshold corresponding to the p-value column. The default is 0.1.
    percent_rank : str optional
        Method by which to sort data from spliting based on percent top and bottom. Need to be either 'logFC' or 'pvalue'.
    percent_thresh : float, optional
        Percent top and bottom sites. The default is 20.
    seq_col : str, optional
        Substrates column name in the differential expression data. The default is 'SITE_+/-7_AA'.
    pad : bool, optional
        How many padding '_' to add from each side of the substrates. The default is False.
    pp : bool, optional
        Treat phospho-residues (s/t/y) within the sequence as phosphopriming. The default is False.
    drop_invalid_subs : bool, optional
        Drop rows with invalid substrates. The default is True.
    drop_de_na : bool, optional
        Drop rows with NaN values in the logFC column. The default is True.
    new_seq_phos_res_cols : bool, optional
        Create a new sequence column or phosphorylated residue column even if one already exists. The default is True.
    suppress_warnings : bool, optional
        Do not print warnings. The default is False.
    """

    def __init__(self, de_data, kin_type, lfc_col, lfc_thresh=0,
                 pval_col=None, pval_thresh=0.1,
                 percent_rank=None, percent_thresh=20,
                 seq_col='SITE_+/-7_AA', pad=False, pp=False,
                 drop_invalid_subs=True, drop_de_na=True,
                 new_seq_phos_res_cols=True, suppress_warnings=False):
        
        exceptions.check_kin_type(kin_type)
        
        self.kin_type = kin_type
        self.pp = pp
        self.seq_col = seq_col
        self.de_lfc_thresh = lfc_thresh
        self.de_pval_thresh = pval_thresh
        self._suppress_warnings = suppress_warnings
        
        self.full_de_data = de_data # Both ser_thr and tyrosine sites
        full_de_data_pps = pps.PhosphoProteomics(de_data, seq_col=seq_col, pad=pad, pp=pp, drop_invalid_subs=drop_invalid_subs, new_seq_phos_res_cols=new_seq_phos_res_cols, suppress_warnings=suppress_warnings)
        self.omited_entries = full_de_data_pps.omited_entries
        self.de_data = getattr(full_de_data_pps, kin_type+'_data') # Only the specifeid kinase type sites
        self.de_data_pps = pps.PhosphoProteomics(self.de_data, drop_invalid_subs=drop_invalid_subs, new_seq_phos_res_cols=False)

        valid_de_data, de_sites, dropped_enteries = enrichment.de_regulated_sites(getattr(self.de_data_pps, kin_type+'_data'), lfc_col, lfc_thresh=lfc_thresh, pval_col=pval_col, pval_thresh=pval_thresh, percent_rank=percent_rank, percent_thresh=percent_thresh, drop_na=drop_de_na, suppress_warnings=suppress_warnings)
        self.valid_de_data = valid_de_data
        self.de_dropped_enteries = dropped_enteries
        self.upreg_sites_data = de_sites['upreg']
        self.upreg_sites_pps = pps.PhosphoProteomics(de_sites['upreg'], seq_col=seq_col, pad=pad, pp=pp, drop_invalid_subs=drop_invalid_subs, new_seq_phos_res_cols=new_seq_phos_res_cols, suppress_warnings=suppress_warnings)
        self.downreg_sites_data = de_sites['downreg']
        self.downreg_sites_pps = pps.PhosphoProteomics(de_sites['downreg'], seq_col=seq_col, pad=pad, pp=pp, drop_invalid_subs=drop_invalid_subs, new_seq_phos_res_cols=new_seq_phos_res_cols, suppress_warnings=suppress_warnings)
        self.unreg_sites_data = de_sites['unreg']
        self.unreg_sites_pps = pps.PhosphoProteomics(de_sites['unreg'], seq_col=seq_col, pad=pad, pp=pp, drop_invalid_subs=drop_invalid_subs, new_seq_phos_res_cols=new_seq_phos_res_cols, suppress_warnings=suppress_warnings)

        
    @staticmethod
    def _score_data(data, kin_type, score_metric, kinases=None, pp=None,
                    validate_aa=True, suppress_warnings=False):
        """
        Private method for scoring all sites in the data based on a specified score metric (score, percentile).
    
        Parameters
        ----------
        data : pd.DataFrame
            Phosphoproteomics data to be scored. Must include a sequence column with the name 'SITE_+/-7_AA'. 
        kin_type : str
            Kinase type. The default is None.
        score_metric : str
            Determines if Kinase Library 'score' or 'percentile' will be returned.
            This will later be used for enrichment. The default calculates both percentile and score dataframes.
        kinases : list, optional
            The kinases included for the specified score_metric. If None, all kinases or kin_type will be returned.
        pp : bool, optional
            Phospho-priming residues (s/t/y). The default is None (will be inherited from the object).
        validate_aa : bool, optional
            Validating amino acids. The default is True.
        suppress_warnings : bool, optional
            Do not print warnings. The default is False.

        Returns
        -------
        data_scores : pd.DataFrame
            pd.Dataframe with the specified Kinase Library score_metric.
        """
        
        exceptions.check_scoring_metric(score_metric)
        data_pps = pps.PhosphoProteomics(data, pp=pp, validate_aa=validate_aa, new_seq_phos_res_cols=False, suppress_warnings=suppress_warnings)
        
        if score_metric == 'score':
            data_scores = data_pps.score(kin_type=kin_type, kinases=kinases, pp=pp, values_only=True)
        elif score_metric == 'percentile':
            data_scores = data_pps.percentile(kin_type=kin_type, kinases=kinases, pp=pp, values_only=True)
            
        return(data_scores)
            
    
    def kinase_enrichment(self, kl_method, kl_thresh,
                          kinases=None, enrichment_type='enriched',
                          ke_sig_lff=0, ke_sig_pval=0.1, adj_pval=True,
                          non_canonical=False, rescore=False):
        """
        Function that performs kinase enrichment, returning a DiffExpEnrichmentResults object for the given condition.

        Parameters
        ---------- 
        kl_method : str
            Kinase Library scoring method ('score', 'score_rank', 'percentile', 'percentile_rank').
        kl_thresh : int
            The threshold to be used for the specified kl_method.
        kinases : list, optional
            If provided, kinase enrichment will only be calculated for the specified kinase list, otherwise, all kinases of the specified kin_type will be included. The default is None.
        enrichment_type : str, optional
            Direction of fisher's exact test for kinase enrichment ('enriched','depleted', or 'both').
        ke_sig_lff : float, optional
            Significance threshold for log frequency factor of kinase enrichment. The default is 0.
        ke_sig_pval : float, optional
            Significance threshold for adjusted p-value of kinase enrichment. The default is 0.1.
        adj_pval : bool, optional
            If True use adjusted p-value for calculation of statistical significance. The default is True.
        non_canonical : bool, optional
            Return also non-canonical kinases. For tyrosine kinases only. The default is False.
        rescore : bool, optional
            If True, all scores or percentiles will be recalculated.
        
        Returns
        -------
        de_enrichment_results : DiffExpEnrichmentResults
            Enrichment results object for the specified method, threshold, and log frequency factor / adjusted p-value cutoffs.
        """
        
        exceptions.check_kl_method(kl_method)
        exceptions.check_enrichment_type(enrichment_type)
        
        kin_type = self.kin_type
        data_att = kl_method+'s'
        kl_comp_direction = _global_vars.kl_method_comp_direction_dict[kl_method]
        
        upreg_enrichment_data = pps.EnrichmentData(foreground=self.upreg_sites_data, background=self.unreg_sites_data,
                                                   fg_seq_col=self.seq_col, bg_seq_col=self.seq_col, kin_type=kin_type,
                                                   new_seq_phos_res_cols=False,
                                                   fg_pp=self.pp, bg_pp=self.pp,
                                                   suppress_warnings=self._suppress_warnings)
        downreg_enrichment_data = pps.EnrichmentData(foreground=self.downreg_sites_data, background=self.unreg_sites_data,
                                                     fg_seq_col=self.seq_col, bg_seq_col=self.seq_col, kin_type=kin_type,
                                                     new_seq_phos_res_cols=False,
                                                     fg_pp=self.pp, bg_pp=self.pp,
                                                     suppress_warnings=self._suppress_warnings)

        if kl_method in ['score','score_rank']:
            if not (hasattr(self.upreg_sites_pps, kin_type+'_scores') and
                    hasattr(self.downreg_sites_pps, kin_type+'_scores') and
                    hasattr(self.unreg_sites_pps, kin_type+'_scores') and
                    hasattr(self.upreg_sites_pps, kin_type+'_score_ranks') and
                    hasattr(self.downreg_sites_pps, kin_type+'_score_ranks') and
                    hasattr(self.unreg_sites_pps, kin_type+'_score_ranks')) or rescore:
                print('\nCalculating scores for upregulated sites ({} substrates)'.format(len(self.upreg_sites_data)))
                upreg_sites_score = self.upreg_sites_pps.score(kin_type=kin_type, kinases=kinases, pp=self.pp, non_canonical=non_canonical, values_only=True)
                print('\nCalculating scores for downregulated sites ({} substrates)'.format(len(self.downreg_sites_data)))
                downreg_sites_score = self.downreg_sites_pps.score(kin_type=kin_type, kinases=kinases, pp=self.pp, non_canonical=non_canonical, values_only=True)
                print('\nCalculating scores for background (unregulated) sites ({} substrates)'.format(len(self.unreg_sites_data)))
                unreg_sites_score = self.unreg_sites_pps.score(kin_type=kin_type, kinases=kinases, pp=self.pp, non_canonical=non_canonical, values_only=True)
            else:
                upreg_sites_score = getattr(self.upreg_sites_pps, kin_type+'_scores')
                downreg_sites_score = getattr(self.downreg_sites_pps, kin_type+'_scores')
                unreg_sites_score = getattr(self.unreg_sites_pps, kin_type+'_scores')
                
            upreg_enrichment_data.submit_scores(data_type='fg', scores=upreg_sites_score, suppress_messages=True)
            upreg_enrichment_data.submit_scores(data_type='bg', scores=unreg_sites_score, suppress_messages=True)
            downreg_enrichment_data.submit_scores(data_type='fg', scores=downreg_sites_score, suppress_messages=True)
            downreg_enrichment_data.submit_scores(data_type='bg', scores=unreg_sites_score, suppress_messages=True)
            
        elif kl_method in ['percentile','percentile_rank']:
            if not (hasattr(self.upreg_sites_pps, kin_type+'_percentiles') and
                    hasattr(self.downreg_sites_pps, kin_type+'_percentiles') and
                    hasattr(self.unreg_sites_pps, kin_type+'_percentiles') and
                    hasattr(self.upreg_sites_pps, kin_type+'_percentile_ranks') and
                    hasattr(self.downreg_sites_pps, kin_type+'_percentile_ranks') and
                    hasattr(self.unreg_sites_pps, kin_type+'_percentile_ranks')) or rescore:
                print('\nCalculating percentiles for upregulated sites ({} substrates)'.format(len(self.upreg_sites_data)))
                upreg_sites_percentile = self.upreg_sites_pps.percentile(kin_type=kin_type, kinases=kinases, pp=self.pp, non_canonical=non_canonical, values_only=True)
                print('\nCalculating percentiles for downregulated sites ({} substrates)'.format(len(self.downreg_sites_data)))
                downreg_sites_percentile = self.downreg_sites_pps.percentile(kin_type=kin_type, kinases=kinases, pp=self.pp, non_canonical=non_canonical, values_only=True)
                print('\nCalculating percentiles for background (unregulated) sites ({} substrates)'.format(len(self.unreg_sites_data)))
                unreg_sites_percentile = self.unreg_sites_pps.percentile(kin_type=kin_type, kinases=kinases, pp=self.pp, non_canonical=non_canonical, values_only=True)
                self.phosprot_name = _global_vars.phosprot_name
            else:
                upreg_sites_percentile = getattr(self.upreg_sites_pps, kin_type+'_percentiles')
                downreg_sites_percentile = getattr(self.downreg_sites_pps, kin_type+'_percentiles')
                unreg_sites_percentile = getattr(self.unreg_sites_pps, kin_type+'_percentiles')
                
            upreg_enrichment_data.submit_percentiles(data_type='fg', percentiles=upreg_sites_percentile, suppress_messages=True)
            upreg_enrichment_data.submit_percentiles(data_type='bg', percentiles=unreg_sites_percentile, suppress_messages=True)
            downreg_enrichment_data.submit_percentiles(data_type='fg', percentiles=downreg_sites_percentile, suppress_messages=True)
            downreg_enrichment_data.submit_percentiles(data_type='bg', percentiles=unreg_sites_percentile, suppress_messages=True)
            
        upreg_enrichment_results = upreg_enrichment_data.kinase_enrichment(kl_method=kl_method, kl_thresh=kl_thresh, kinases=kinases, pp=self.pp, enrichment_type=enrichment_type, adj_pval=adj_pval, non_canonical=non_canonical)
        downreg_enrichment_results = downreg_enrichment_data.kinase_enrichment(kl_method=kl_method, kl_thresh=kl_thresh, kinases=kinases, pp=self.pp, enrichment_type=enrichment_type, adj_pval=adj_pval, non_canonical=non_canonical)
        
        if enrichment_type == 'both':
            warnings.warn('Enrichment side is set to \'both\', this might produce unexpectged results in combined enrichment results.')
            
        de_enrichment_results = DiffExpEnrichmentResults(upreg_enrichment_results=upreg_enrichment_results,
                                                         downreg_enrichment_results=downreg_enrichment_results,
                                                         kin_type=kin_type, kl_method=kl_method, kl_thresh=kl_thresh,
                                                         diff_exp_data=self, enrichment_type=enrichment_type,
                                                         ke_sig_lff=ke_sig_lff, ke_sig_pval=ke_sig_pval, adj_pval=adj_pval,
                                                         data_att=data_att, kl_comp_direction=kl_comp_direction)
        
        return de_enrichment_results
    
    
    def submit_scores(self, scores, sites_type=['upregulated','downregulated','unregulated'], suppress_messages=False):
        """
        Submitting scores for up/down/unregulated substrates.

        Parameters
        ----------
        scores : pd.DataFrame
            Dataframe with sites scores.
            Index must contain all the values in 'seq_col' with no duplicates.
            Columns must contain valid kinase names.
        sites_type : str
            Sites type: upregulated, downregulated, or unregulated.
        suppress_messages : bool, optional
            Suppress messages. the default is False.

        Raises
        ------
        ValueError
            Raise error if sites type is not valid.

        Returns
        -------
        None.
        """
        
        if isinstance(sites_type, str):
            sites_type = [sites_type]
        
        for st_tp in sites_type:
            exceptions.check_de_sites_type(st_tp)
            
            if st_tp == 'upregulated':
                self.upreg_sites_pps.submit_scores(kin_type=self.kin_type, scores=scores, suppress_messages=suppress_messages)
            elif st_tp == 'downregulated':
                self.downreg_sites_pps.submit_scores(kin_type=self.kin_type, scores=scores, suppress_messages=suppress_messages)
            elif st_tp == 'unregulated':
                self.unreg_sites_pps.submit_scores(kin_type=self.kin_type, scores=scores, suppress_messages=suppress_messages)
        
        
    def submit_percentiles(self, percentiles, sites_type=['upregulated','downregulated','unregulated'], phosprot_name=None, suppress_messages=False):
        """
        Submitting percentiles for up/down/unregulated substrates.

        Parameters
        ----------
        percentiles : pd.DataFrame
            Dataframe with sites percentiles.
            Index must contain all the values in 'seq_col' with no duplicates.
            Columns must contain valid kinase names.
        sites_type : str
            Sites type: upregulated, downregulated, or unregulated.
        phosprot_name : str, optional
            Name of phosphoproteome database.
        suppress_messages : bool, optional
            Suppress messages. the default is False.
        
        Raises
        ------
        ValueError
            Raise error if sites type is not valid.
            
        Returns
        -------
        None.
        """
        
        if isinstance(sites_type, str):
            sites_type = [sites_type]
        
        if phosprot_name is None:
            phosprot_name = _global_vars.phosprot_name
        self.phosprot_name = phosprot_name
        
        for st_tp in sites_type:
            exceptions.check_de_sites_type(st_tp)
        
            if st_tp == 'upregulated':
                self.upreg_sites_pps.submit_percentiles(kin_type=self.kin_type, percentiles=percentiles, phosprot_name=phosprot_name, suppress_messages=suppress_messages)
            elif st_tp == 'downregulated':
                self.downreg_sites_pps.submit_percentiles(kin_type=self.kin_type, percentiles=percentiles, phosprot_name=phosprot_name, suppress_messages=suppress_messages)
            elif st_tp == 'unregulated':
                self.unreg_sites_pps.submit_percentiles(kin_type=self.kin_type, percentiles=percentiles, phosprot_name=phosprot_name, suppress_messages=suppress_messages)

#%%

class DiffExpEnrichmentResults(object):
    """
    Class for differential expression results.

    upreg_enrichment_results : kl.EnrichmentResults
        Enrichment results object for upregulated substrates based on Kinase Library method/threshold, logFC threshold, and p-value cutoff.
    downreg_enrichment_results : kl.EnrichmentResults
        Enrichment results object for downregulated substrates based on Kinase Library method/threshold, logFC threshold, and p-value cutoff.
    combined_enrichment_results : pd.DataFrame
        Combined enrichment results for upregulated and downregulated enrichments.
    diff_exp_data : kl.DiffExpData
        Differential expression data object corresponding to the DiffExpEnrichmentResults object. 
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
    ke_sig_lff : float
        Minimum log frequency factor output of Kinase Library enrichment that is deemed as significant.
    ke_sig_pval : float
        Maximum p-value output of Kinase Library enrichment that is deemed as significant.
    adj_pval : bool
        If True use adjusted p-value for calculation of statistical significance. Otherwise, use nominal p-value.
    data_att : str
        Score type ('scores', 'score_ranks', 'percentiles', 'percentile_ranks').
    kl_comp_direction : str
        Dictates if kinases above or below the specified threshold are used ('higher','lower').
    """

    def __init__(self, upreg_enrichment_results, downreg_enrichment_results,
                 diff_exp_data, kin_type, kl_method, kl_thresh, enrichment_type,
                 ke_sig_lff, ke_sig_pval, adj_pval,
                 data_att, kl_comp_direction):
        
        self.upreg_enrichment_results = upreg_enrichment_results
        self.downreg_enrichment_results = downreg_enrichment_results
        self.diff_exp_data = diff_exp_data
        self.de_lfc_thresh = diff_exp_data.de_lfc_thresh
        self.de_pval_thresh = diff_exp_data.de_pval_thresh
        self.kin_type = kin_type
        self.kl_method = kl_method
        self.kl_thresh = kl_thresh
        self.enrichment_type = enrichment_type
        self.ke_sig_lff = ke_sig_lff
        self.ke_sig_pval = ke_sig_pval
        self.adj_pval = adj_pval
        self._data_att = data_att
        self._kl_comp_direction = kl_comp_direction
        
        self.combined_enrichment_results = self._combine_enrichments()
        self.tested_kins = self.combined_enrichment_results.index.to_list()
        
        self.activated_kins = upreg_enrichment_results.enriched_kins(sig_pval=ke_sig_pval, sig_lff=ke_sig_lff)
        self.inhibited_kins = downreg_enrichment_results.enriched_kins(sig_pval=ke_sig_pval, sig_lff=ke_sig_lff)
        self.contradicting_kins = [x for x in self.activated_kins if x in self.inhibited_kins]
        
        if kl_method in ['percentile','percentile_rank']:
            self.phosprot_name = diff_exp_data.phosprot_name
        
    
    def _combine_enrichments(self):
        """
        Private function to combine upregulated and downregulated enrichment results to be displayed in the same volcano plot or bubblemap.
    
        Returns
        -------
        combined_down_up_enrich: pd.DataFrame
            Combined enrichment results for upregulated and downregulated enrichments.
        """
        
        upreg_enriched = self.upreg_enrichment_results.enrichment_results
        downreg_enriched = self.downreg_enrichment_results.enrichment_results
        combined_down_up_enrich = downreg_enriched.join(upreg_enriched, lsuffix = '_downreg', rsuffix = '_upreg')
        direction = []
        combined_freq_factors = []
        combined_adj_pvals = []
        
        for kin,enrich_data in combined_down_up_enrich.iterrows():
            freq_down = enrich_data['log2_freq_factor_downreg']
            freq_up = enrich_data['log2_freq_factor_upreg']
            if self.adj_pval:
                adj_pval_down = enrich_data['adj_fisher_pval_downreg']
                adj_pval_up = enrich_data['adj_fisher_pval_upreg']
            else:
                adj_pval_down = enrich_data['fisher_pval_downreg']
                adj_pval_up = enrich_data['fisher_pval_upreg']
            
            if adj_pval_down == adj_pval_up: # kinase is same adjusted p-value (most likely 1)
                max_freq = max([freq_down, freq_up], key=abs) # highest absolute value of log2(frequency)
                if max_freq == freq_down:
                    direction.append('-')
                    combined_freq_factors.append(-freq_down)
                else:
                    direction.append('+')
                    combined_freq_factors.append(freq_up)
                combined_adj_pvals.append(adj_pval_down)
            else:
                min_pval = min(adj_pval_down, adj_pval_up)
                if min_pval == adj_pval_down:
                    direction.append('-')
                    combined_freq_factors.append(-freq_down)
                else:
                    direction.append('+')
                    combined_freq_factors.append(freq_up)
                combined_adj_pvals.append(min_pval)
            
            # kinase is significant both in upregulated and downregulated enrichments (contradicting kinases)
            if (max(adj_pval_down, adj_pval_up) <= self.ke_sig_pval) and (min(freq_down, freq_up, key=abs) >= self.ke_sig_lff): 
                direction = direction[:-1]
                direction.append('0')
                
        combined_down_up_enrich['most_sig_direction'] = direction
        combined_down_up_enrich['most_sig_log2_freq_factor'] = combined_freq_factors
        combined_down_up_enrich['most_sig_adj_fisher_pval'] = combined_adj_pvals
    
        return combined_down_up_enrich
    
    
    def _get_cont_kins_data(self):
        """
        Private method for populating combined enrichment results table with most significant adjusted p-values and frequency factors for kinases that are enriched in both upregulated and downregulated sites.
    
        Returns
        -------
        combined_cont_kins_data : pd.DataFrame
            pd.Dataframe with exploded frequency factor and p-value columns including information for 'contradicting' kinases.
        """
        
        combined_cont_kins = self.combined_enrichment_results.copy()
        
        combined_cont_kins['most_sig_log2_freq_factor'] = combined_cont_kins.apply(lambda x: [-x['log2_freq_factor_downreg'],x['log2_freq_factor_upreg']] if x['most_sig_direction'] == '0' else [x['most_sig_log2_freq_factor']], axis=1)
        if self.adj_pval:
            combined_cont_kins['most_sig_adj_fisher_pval'] = combined_cont_kins.apply(lambda x: [x['adj_fisher_pval_downreg'],x['adj_fisher_pval_upreg']] if x['most_sig_direction'] == '0' else [x['most_sig_adj_fisher_pval']], axis=1)
        else:
            combined_cont_kins['most_sig_adj_fisher_pval'] = combined_cont_kins.apply(lambda x: [x['fisher_pval_downreg'],x['fisher_pval_upreg']] if x['most_sig_direction'] == '0' else [x['most_sig_adj_fisher_pval']], axis=1)
        combined_cont_kins['most_sig_direction'] = combined_cont_kins.apply(lambda x: ['-','+'] if x['most_sig_direction'] == '0' else [x['most_sig_direction']], axis=1)
        
        combined_cont_kins_data = combined_cont_kins.explode(column=['most_sig_log2_freq_factor','most_sig_adj_fisher_pval','most_sig_direction'])
        
        return(combined_cont_kins_data)
    
    
    def enriched_subs(self, kinases, activity_type, data_columns=None,
                      save_to_excel=False, output_dir=None, file_prefix=None):
        """
        Function to save an excel file containing the subset of substrates that drove specific kinases' enrichment.

        Parameters
        ----------
        kinases : list
            List of kinases for enriched substrates. Substrates provided are those that drove that kinase's enrichment.
        data_columns : list, optional
            Columns from original data to be included with each enriched substrate. Defaults to None, including all original columns.
        save_to_excel : bool, optional
            If True, excel file containing enriched substrates will be saved to the output_dir.
        output_dir : str, optional
            Location for enriched substrates excel file to be saved. Must be True if save_to_excel.
        file_prefix : str, optional
            Prefix for the files name.
           
        Returns
        -------
        enrich_subs_dict : dict
            Dictionary with the substrates that drove enrichment for each kinase.
        """
        
        if activity_type not in ['activated','inhibited','both']:
            raise ValueError('activity_type must be either \'activated\', \'inhibited\' or  \'both\'.')
        
        if file_prefix is not None:
            full_file_prefix = file_prefix + '_' + activity_type
        else:
            full_file_prefix = activity_type
                
        if activity_type == 'activated':
            return self.upreg_enrichment_results.enriched_subs(kinases=kinases, data_columns=data_columns,
                                                               save_to_excel=save_to_excel, output_dir=output_dir,
                                                               file_prefix=full_file_prefix)
        elif activity_type == 'inhibited':
            return self.downreg_enrichment_results.enriched_subs(kinases=kinases, data_columns=data_columns,
                                                                 save_to_excel=save_to_excel, output_dir=output_dir,
                                                                 file_prefix=full_file_prefix)
        elif activity_type == 'both':
            downreg_subs =  self.downreg_enrichment_results.enriched_subs(kinases=kinases, data_columns=data_columns,save_to_excel=False, output_dir=output_dir,file_prefix=full_file_prefix)
            upreg_subs = self.upreg_enrichment_results.enriched_subs(kinases=kinases, data_columns=data_columns,save_to_excel=False, output_dir=output_dir,file_prefix=full_file_prefix)
            if file_prefix is not None:
                writer = pd.ExcelWriter(f'{output_dir}/enriched_subs/{file_prefix}_{self.kin_type}_{self._data_att}_thresh_{self.kl_thresh}.xlsx')
            else:
                writer = pd.ExcelWriter(f'{output_dir}/enriched_subs/{self.kin_type}_{self._data_att}_thresh_{self.kl_thresh}.xlsx')
            enrich_subs_dict = {}
            for kin in kinases:
                if save_to_excel:
                    enriched_kin_subs = pd.concat([downreg_subs[kin],upreg_subs[kin]]).sort_values(by=kin, ascending=True)
                    enriched_kin_subs.to_excel(writer, sheet_name=kin, index=False)
                enrich_subs_dict[kin] = enriched_kin_subs
            if save_to_excel:
                writer.save()
            return enrich_subs_dict   
    
    def get_activated_kins(self, sig_pval=0.1, sig_lff=0,
                           pval_col=None, lff_col='log2_freq_factor'):
        """
        Returns a list of all kinases enriched above a p-value and frequency-factor thresholds in the upregulated enrichment results.

        Parameters
        ----------
        sig_pval : float, optional
            Threshold to determine the p-value significance of kinase enrichment. The default is 0.1 (adjusted p-values).
        sig_lff : float, optional
            Threshold to determine the Log frequency factor cutoff of kinase enrichment. The default is 0.
        pval_col : str, optional
            Name of column containing the p-value output of kinase enrichment. The defulat is None and will be determined based on self.adj_pval.
        lff_col : str, optional
            Name of column containing the log2 frequency factor output of kinase enrichment.
           
        Returns
        -------
        activated_kins : list
            List of kinases enriched above the designated p-value and freq_factor thresholds in the upregulated data.
        """
        
        if pval_col is None:
            if self.adj_pval:
                pval_col='adj_fisher_pval'
            else:
                pval_col='fisher_pval'
        
        return self.upreg_enrichment_results.enriched_kins(sig_pval=sig_pval, sig_lff=sig_lff, pval_col=pval_col, lff_col=lff_col)
    
    
    def get_inhibited_kins(self, sig_pval=0.1, sig_lff=0,
                           pval_col='adj_fisher_pval', lff_col='log2_freq_factor'):
        """
        Returns a list of all kinases enriched above a p-value and frequency-factor thresholds in the downregulated enrichment results.

        Parameters
        ----------
        sig_pval : float, optional
            Threshold to determine the p-value significance of kinase enrichment. The default is 0.1 (adjusted p-values).
        sig_lff : float, optional
            Threshold to determine the Log frequency factor cutoff of kinase enrichment. The default is 0.
        pval_col : str, optional
            Name of column containing the p-value output of kinase enrichment. The defulat is None and will be determined based on self.adj_pval.
        lff_col : str, optional
            Name of column containing the log2 frequency factor output of kinase enrichment.
           
        Returns
        -------
        inhibited_kins : list
            List of kinases enriched above the designated p-value and freq_factor thresholds in the upregulated data.
        """
        
        if pval_col is None:
            if self.adj_pval:
                pval_col='adj_fisher_pval'
            else:
                pval_col='fisher_pval'
        
        return self.downreg_enrichment_results.enriched_kins(sig_pval=sig_pval, sig_lff=sig_lff, pval_col=pval_col, lff_col=lff_col)
    
    
    def get_contradicting_kins(self, sig_pval=0.1, sig_lff=0,
                               pval_col=None, lff_col='log2_freq_factor'):
        """
        Returns a list of all kinases enriched above a p-value and frequency-factor thresholds in both the upregulated and downregulated enrichment results.

        Parameters
        ----------
        sig_pval : float, optional
            Threshold to determine the p-value significance of kinase enrichment. The default is 0.1 (adjusted p-values).
        sig_lff : float, optional
            Threshold to determine the Log frequency factor cutoff of kinase enrichment. The default is 0.
        pval_col : str, optional
            Name of column containing the p-value output of kinase enrichment. The defulat is None and will be determined based on self.adj_pval.
        lff_col : str, optional
            Name of column containing the log2 frequency factor output of kinase enrichment.
           
        Returns
        -------
        contradicting_kins : list
            List of kinases enriched above the designated p-value and freq_facor thresholds in both the upregulated and downregulated data.
        """
        
        if pval_col is None:
            if self.adj_pval:
                pval_col='adj_fisher_pval'
            else:
                pval_col='fisher_pval'
        
        return [x for x in self.get_activated_kins(sig_pval=sig_pval, sig_lff=sig_lff, pval_col=pval_col, lff_col=lff_col) if
                x in self.get_inhibited_kins(sig_pval=sig_pval, sig_lff=sig_lff, pval_col=pval_col, lff_col=lff_col)]
    
    
    def plot_volcano(self, enrichment_type='combined', plot_sig_lff=0, plot_sig_pval=0.1, kinases=None,
                     lff_col=None, pval_col=None, plot_cont_kins=True, highlight_kins=None,
                     fg_percent_thresh=0, fg_percent_col=None, ignore_depleted=True,
                     kins_label_dict=None, label_kins=None, adjust_labels=True, labels_fontsize=7,
                     symmetric_xaxis=True, grid=True, max_window=False,
                     title=None, stats=True, xlabel='log$_2$(Frequency Factor)', ylabel='-log$_{10}$(Adjusted p-value)',
                     plot=True, save_fig=False, return_fig=False,
                     ax=None, **plot_kwargs):
        """
        Returns a volcano plot of the Kinase Library differential expression enrichment results.

        Parameters
        ----------
        enrichment_type : str, optional
            Site subset on which enrichment is calculated ('upregulated','downregulated', or 'combined').
        plot_sig_lff : float, optional
            Significance threshold for logFF in the enrichment results. The default is 0.
        plot_sig_pval : float, optional
            Significance threshold for and adjusted p-value in the enrichment results. The default is 0.1.
        kinases : list, optional
            If provided, kinases to plot in the volcano plot. The default is None.
        lff_col : str, optional
            Log frequency factor column name used for volcano plot. The default is 'log2_freq_factor'.
        pval_col : str, optional
            P-value column name used for volcano plot. The default is 'adj_fisher_pval'.
        plot_cont_kins : bool, optional
            If False, kinases enriched in both upregulated and downregulated sites will be excluded from the volcano.
            If True, they will be highlighted in yellow.
        highlight_kins : list, optional
            List of kinases to be marked in yellow on the kinase enrichment volcano plot.
        fg_percent_thresh : float, optional
            Minimun percentage of foreground substrates predicted for a kinase in order to plot it in the volcano.
        fg_percent_col : str, optional
            Column name of percentage of foreground substrates mapped to each kinase.
        ignore_depleted : bool, optional
            Ignore kinases that their FF is negative (depleted). The default is False.
        kins_label_dict : dict, optional
            Dictionary with customized labels for each kinase. The default is None.
        label_kins : list, optional
            List of kinases to label on volcano plot. The default is None.
            If none, all significant kinases will be labelled plus any non-significant kinases marked for highlighting.
        adjust_labels : bool, optional
            If True, labels will be adjusted to avoid other markers and text on volcano plot. The default is True.
        labels_fontsize : int, optional
            Font size used for the volcano's kinase labels, defaults to 7.
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
            y-axis label for the volcano plot. The default is ''-log$_{10}$(Adjusted p-value)'.
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
        **plot_kwargs: optional
            Optional keyword arguments to be passed to the plot_volcano function.

        Returns
        -------
        If return_fig, the kinase enrichment volcano plot.
        """

        exceptions.check_de_enrichment_type(enrichment_type)
        
        if highlight_kins is None:
            highlight_kins = []
        if plot_cont_kins:
            highlight_kins = highlight_kins + self.get_contradicting_kins(sig_pval=plot_sig_pval, sig_lff=plot_sig_lff)
        
        if enrichment_type == 'upregulated':
            if lff_col is None:
                lff_col='log2_freq_factor'
            if pval_col is None:
                if self.adj_pval:
                    pval_col='adj_fisher_pval'
                else:
                    pval_col='fisher_pval'
            if fg_percent_col is None:
                fg_percent_col='fg_percent'
            if stats:
                title = '\n'.join(filter(None, [title,f'Upreg: {len(self.diff_exp_data.upreg_sites_data)}; Unreg: {len(self.diff_exp_data.unreg_sites_data)}']))
            
            fg_percent_kins = self.upreg_enrichment_results.enrichment_results[self.upreg_enrichment_results.enrichment_results[fg_percent_col] >= fg_percent_thresh].index.to_list()
            if kinases is not None:
                kinases = [x for x in kinases if x in fg_percent_kins]
            else:
                kinases = fg_percent_kins
                
            if kins_label_dict:
                kinases = [kins_label_dict[x] for x in kinases]
                return enrichment.plot_volcano(self.upreg_enrichment_results.enrichment_results.rename(kins_label_dict),
                                               sig_lff=plot_sig_lff, sig_pval=plot_sig_pval, kinases=kinases,
                                               lff_col=lff_col, pval_col=pval_col, highlight_kins=highlight_kins, ignore_depleted=ignore_depleted,
                                               label_kins=label_kins, adjust_labels=adjust_labels, labels_fontsize=labels_fontsize,
                                               symmetric_xaxis=symmetric_xaxis, grid=grid, max_window=max_window,
                                               title=title, xlabel=xlabel, ylabel=ylabel,
                                               plot=plot, save_fig=save_fig, return_fig=return_fig,
                                               ax=ax, **plot_kwargs)
            else:
                return enrichment.plot_volcano(self.upreg_enrichment_results.enrichment_results, sig_lff=plot_sig_lff, sig_pval=plot_sig_pval, kinases=kinases,
                                               lff_col=lff_col, pval_col=pval_col, highlight_kins=highlight_kins, ignore_depleted=ignore_depleted,
                                               label_kins=label_kins, adjust_labels=adjust_labels, labels_fontsize=labels_fontsize,
                                               symmetric_xaxis=symmetric_xaxis, grid=grid, max_window=max_window,
                                               title=title, xlabel=xlabel, ylabel=ylabel,
                                               plot=plot, save_fig=save_fig, return_fig=return_fig,
                                               ax=ax, **plot_kwargs)
        
        elif enrichment_type == 'downregulated':
            if lff_col is None:
                lff_col='log2_freq_factor'
            if pval_col is None:
                if self.adj_pval:
                    pval_col='adj_fisher_pval'
                else:
                    pval_col='fisher_pval'
            if fg_percent_col is None:
                fg_percent_col='fg_percent'
            if stats:
                title = '\n'.join(filter(None, [title,f'Downreg: {len(self.diff_exp_data.downreg_sites_data)}; Unreg: {len(self.diff_exp_data.unreg_sites_data)}']))
            
            fg_percent_kins = self.downreg_enrichment_results.enrichment_results[self.downreg_enrichment_results.enrichment_results[fg_percent_col] >= fg_percent_thresh].index.to_list()
            if kinases is not None:
                kinases = [x for x in kinases if x in fg_percent_kins]
            else:
                kinases = fg_percent_kins
            
            if kins_label_dict:
                kinases = [kins_label_dict[x] for x in kinases]
                return enrichment.plot_volcano(self.downreg_enrichment_results.enrichment_results.rename(kins_label_dict),
                                               sig_lff=plot_sig_lff, sig_pval=plot_sig_pval, kinases=kinases,
                                               lff_col=lff_col, pval_col=pval_col, highlight_kins=highlight_kins, ignore_depleted=ignore_depleted,
                                               label_kins=label_kins, adjust_labels=adjust_labels, labels_fontsize=labels_fontsize,
                                               symmetric_xaxis=symmetric_xaxis, grid=grid, max_window=max_window,
                                               title=title, xlabel=xlabel, ylabel=ylabel,
                                               plot=plot, save_fig=save_fig, return_fig=return_fig,
                                               ax=ax, **plot_kwargs)
            else:
                return enrichment.plot_volcano(self.downreg_enrichment_results.enrichment_results, sig_lff=plot_sig_lff, sig_pval=plot_sig_pval, kinases=kinases,
                                               lff_col=lff_col, pval_col=pval_col, highlight_kins=highlight_kins, ignore_depleted=ignore_depleted,
                                               label_kins=label_kins, adjust_labels=adjust_labels, labels_fontsize=labels_fontsize,
                                               symmetric_xaxis=symmetric_xaxis, grid=grid, max_window=max_window,
                                               title=title, xlabel=xlabel, ylabel=ylabel,
                                               plot=plot, save_fig=save_fig, return_fig=return_fig,
                                               ax=ax, **plot_kwargs)
        
        elif enrichment_type == 'combined':
            
            if lff_col is None:
                lff_col='most_sig_log2_freq_factor'
            if pval_col is None:
                pval_col='most_sig_adj_fisher_pval'
            if fg_percent_col is None:
                fg_percent_col='fg_percent_combined'
            if plot_cont_kins:
                combined_cont_kins_data = self._get_cont_kins_data()
            else:
                combined_cont_kins_data = self.combined_enrichment_results.copy()
                combined_cont_kins_data = combined_cont_kins_data[~(combined_cont_kins_data['most_sig_direction'] == '0')]
            if stats:
                title = '\n'.join(filter(None, [title,f'Upreg: {len(self.diff_exp_data.upreg_sites_data)}; Downreg: {len(self.diff_exp_data.downreg_sites_data)}; Unreg: {len(self.diff_exp_data.unreg_sites_data)}']))
            
            combined_cont_kins_data['fg_percent_combined'] = combined_cont_kins_data['fg_percent_downreg']*(combined_cont_kins_data['most_sig_direction'] == '-') + combined_cont_kins_data['fg_percent_upreg']*(combined_cont_kins_data['most_sig_direction'] == '+')
            
            # Removing only rows of contradicting kinases that do not pass the fg_percent threshold (due to duplicated index and plotting)
            temp_combined_cont_kins_data = combined_cont_kins_data.reset_index()
            drop_ind = temp_combined_cont_kins_data[(temp_combined_cont_kins_data[fg_percent_col] < fg_percent_thresh) & (temp_combined_cont_kins_data['index'].isin(self.get_contradicting_kins(sig_pval=plot_sig_pval, sig_lff=plot_sig_lff)))].index
            combined_cont_kins_data = temp_combined_cont_kins_data.drop(drop_ind).set_index('index')
            
            fg_percent_kins = combined_cont_kins_data[combined_cont_kins_data[fg_percent_col] >= fg_percent_thresh].index.to_list()
            if kinases is not None:
                kinases = [x for x in kinases if x in fg_percent_kins]
            else:
                kinases = fg_percent_kins
            
            if kins_label_dict:
                kinases = [kins_label_dict[x] for x in kinases]
                return enrichment.plot_volcano(combined_cont_kins_data.rename(kins_label_dict),
                                               sig_lff=plot_sig_lff, sig_pval=plot_sig_pval, kinases=kinases,
                                               lff_col=lff_col, pval_col=pval_col, highlight_kins=highlight_kins, ignore_depleted=ignore_depleted,
                                               label_kins=label_kins, adjust_labels=adjust_labels, labels_fontsize=labels_fontsize,
                                               symmetric_xaxis=symmetric_xaxis, grid=grid, max_window=max_window,
                                               title=title, xlabel=xlabel, ylabel=ylabel,
                                               plot=plot, save_fig=save_fig, return_fig=return_fig,
                                               ax=ax, **plot_kwargs)
            else:
                return enrichment.plot_volcano(combined_cont_kins_data, sig_lff=plot_sig_lff, sig_pval=plot_sig_pval, kinases=kinases,
                                               lff_col=lff_col, pval_col=pval_col, highlight_kins=highlight_kins, ignore_depleted=ignore_depleted,
                                               label_kins=label_kins, adjust_labels=adjust_labels, labels_fontsize=labels_fontsize,
                                               symmetric_xaxis=symmetric_xaxis, grid=grid, max_window=max_window,
                                               title=title, xlabel=xlabel, ylabel=ylabel,
                                               plot=plot, save_fig=save_fig, return_fig=return_fig,
                                               ax=ax, **plot_kwargs)

    def plot_down_up_comb_volcanos(self, plot_sig_lff=0, plot_sig_pval=0.1, kinases=None,
                                   plot_cont_kins=True, highlight_kins=None, ignore_depleted=False,
                                   label_kins=None, adjust_labels=True, labels_fontsize=7,
                                   symmetric_xaxis=True, grid=True, max_window=False,
                                   title=None, xlabel='log$_2$(Frequency Factor)', ylabel='-log$_{10}$(Adjusted p-value)',
                                   plot=True, save_fig=False, return_fig=False,
                                   ax=None, **plot_kwargs):
        """
        Returns a 1x3 figure containing downregulated, upregulated, and combined volcano plots of the Kinase Library differential expression enrichment results.

        Parameters
        ----------
        plot_sig_lff : float, optional
            Significance threshold for logFF in the enrichment results. The default is 0.
        plot_sig_pval : float, optional
            Significance threshold for and adjusted p-value in the enrichment results. The default is 0.1.
        kinases : list, optional
            If provided, kinases to plot in the volcano plot. The default is None.
        plot_cont_kins : bool, optional
            If False, kinases enriched in both upregulated and downregulated sites will be excluded from the volcano.
            If True, they will be highlighted in yellow.
        highlight_kins : list, optional
            List of kinases to be marked in yellow on the kinase enrichment volcano plots.
        ignore_depleted : bool, optional
            Ignore kinases that their FF is negative (depleted). The default is False.
        label_kins : list, optional
            List of kinases to label on volcano plots. The default is None.
            If none, all significant kinases will be labelled plus any non-significant kinases marked for highlighting.
        adjust_labels : bool, optional
            If True, labels will be adjusted to avoid other markers and text on volcano plots. The default is True.
        symmetric_xaxis : bool, optional
            If True, x-axis will be made symmetric to its maximum absolute value. The default is True.
        grid : bool, optional
            If True, a grid is provided on the enrichment results volcano plots. The default is True.
        max_window : bool, optional
            For plotting and data visualization purposes; if True, plotting window will be maximized. The default is False.
            Must be False if an axis is provided to the function.
        title : str, optional
            Title for the figure. The default is False.
        xlabel : str, optional
            x-axis label for the volcano plots. The default is 'log$_2$(Frequency Factor)'.
        ylabel : str, optional
            y-axis label for the volcano plots. The default is ''-log$_{10}$(Adjusted p-value)'.
        plot : bool, optional
            Whether or not to plot the produced enrichment figure. The default is True.
            Will be automatically changed to False if an axis is provided.
        save_fig : str, optional
            Path to file for saving the figure. The default is False.
            Must be False if an axis is provided.
        return_fig : bool, optional
            If true, the figure will be returned as a plt.figure object. The default is False.
        ax : plt.axes, optional
            Axes provided to plot the kinase enrichment figure onto. The default is None.
        **plot_kwargs: optional
            Optional keyword arguments to be passed to the plot_volcano function.
            
        Returns
        -------
        If return_fig, the figure containing downregulated, upregulated, and combined kinase enrichment volcano plots.
        """

        enrichment_types = ['downregulated','upregulated','combined']
        plot_titles = ['Downregulated','Upregulated','Combined']
        
        if ax is None:
            existing_ax = False
            w,h = plt.figaspect(1/3)
            fig,ax = plt.subplots(1, 3, figsize=(w,h))
        else:
            if len(ax) != 3:
                raise ValueError('\'ax\' must contain 3 axes objects.')
            existing_ax = True
            plot = False
            if max_window or save_fig or return_fig:
                raise ValueError('When Axes provided, \'max_window\', \'save_fig\', and \'return_fig\' must be False.')
        
        for i in range(3):
            self.plot_volcano(enrichment_type=enrichment_types[i],
                              plot_sig_lff=plot_sig_lff, plot_sig_pval=plot_sig_pval, kinases=kinases,
                              plot_cont_kins=plot_cont_kins, highlight_kins=highlight_kins, ignore_depleted=ignore_depleted,
                              label_kins=label_kins, adjust_labels=adjust_labels, labels_fontsize=labels_fontsize,
                              symmetric_xaxis=symmetric_xaxis, grid=grid, max_window=max_window,
                              title=plot_titles[i], xlabel=xlabel, ylabel=ylabel,
                              ax=ax[i], **plot_kwargs)
        
        if not existing_ax:
            fig.suptitle(title)
            fig.tight_layout()
        
        if save_fig:
            fig.savefig(save_fig, dpi=1000)
            
        if not plot and not existing_ax:
            plt.close(fig)
                
        if return_fig:
            return fig

#%% Static functions

def plot_3x3_volcanos(de_data, kin_type, kl_method, kl_thresh, de_lfc_col, de_lfc_thresh=[0,0.5,1],
                      de_pval_col=None, de_pval_thresh=[0.1,0.1,0.1], drop_de_na=True, kinases=None,
                      seq_col='SITE_+/-7_AA', ke_plot_sig_lff=0, ke_plot_sig_pval=0.1,
                      plot_cont_kins=True, highlight_kins=None, ignore_depleted=True,
                      label_kins=None, adjust_labels=True, labels_fontsize=7, title=None,
                      plot=True, save_fig=False, return_fig=False,
                      suppress_warnings=True,
                      scoring_kwargs={},
                      diff_exp_kwargs={},
                      enrichment_kwargs={},
                      plotting_kwargs={}):
    """
    Returns a 3x3 figure containing downregulated, upregulated, and combined volcano plots of the Kinase Library differential expression enrichment results for three logFC thresholds.

    Parameters
    ----------
    de_data : pd.DataFrame
        DataFrame containing differential expression data (must include sequence and logFC columns).
    kin_type : str
        Kinase type ('ser_thr' or 'tyrosine').
    kl_method : str
        Kinase Library scoring method ('score', 'score_rank', 'percentile', 'percentile_rank').
    kl_thresh : int
        The threshold to be used for the specified kl_method.
    de_lfc_col : str
        LogFC column name for Kinase Library enrichment analysis.
    de_lfc_thresh : list, optional
        List of three logFC cuttoffs used to define up, down, and unregulated sites.
    de_pval_col : str, optional
        P-value column name used to define a site's significance.
    de_pval_thresh : list, optional
        List of three significance threshold corresponding to the p-value column. The default is [0.1]*3.
    drop_de_na : bool, optional
        Drop de_data rows with NaN values in the logFC column. The default is True.
    kinases : list, optional
        If provided, kinase enrichment will only be calculated for the specified kinase list, otherwise, all kinases of the specified kin_type will be included. The default is None.
    seq_col='SITE_+/-7_AA'
        Substrates column name in the differential expression data. The default is 'SITE_+/-7_AA'.
    ke_plot_sig_lff : float, optional
        Significance threshold for logFF in the enrichment results. The default is 0.
    ke_plot_sig_pval : float, optional
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
    diff_exp_kwargs : dict, optional
        Optional keyword arguments to be passed to the PhosphoProteomics initialization function.
    enrichment_kwargs : dict, optional
        Optional keyword arguments to be passed to the kinase_enrichment function.
    plotting_kwargs : dict, optional
        Optional keyword arguments to be passed to the plot_volcano function.
        
    Returns
    -------
    If return_fig, the 3x3 figure containing downregulated, upregulated, and combined kinase enrichment volcano plots.
    """
    
    if len(de_lfc_thresh) != 3:
        raise ValueError('\'de_lfc_thresh\' must contain exactly three values.')
    if de_pval_col is not None and len(de_pval_thresh) != 3:
        raise ValueError('\'de_pval_thresh\' must contain exactly three values.')
    
    exceptions.check_kl_method(kl_method)
    print('Calculating scores for all sites')
    de_data_pps = pps.PhosphoProteomics(data=de_data, seq_col=seq_col, **diff_exp_kwargs)
    if kl_method in ['score','score_rank']:
        scores = de_data_pps.score(kin_type=kin_type, kinases=kinases, values_only=True, **scoring_kwargs)
    elif kl_method in ['percentile','percentile_rank']:
        percentiles = de_data_pps.percentile(kin_type=kin_type, kinases=kinases, values_only=True, **scoring_kwargs)
        
    fig = plt.figure(constrained_layout=True)
    figManager = fig.canvas.manager
    figManager.window.showMaximized()
    subfigs = fig.subfigures(nrows=3, ncols=1)
        
    for i,(lfc,pval) in enumerate(zip(de_lfc_thresh,de_pval_thresh)):
        
        subfigs[i].suptitle(r'$\bf{' + f'DE\ logFC\ threshold:\ {lfc}' + f'\ /\ DE\ p-value\ threshold:\ {pval}'*(de_pval_col is not None) + '}$')
        ax = subfigs[i].subplots(nrows=1, ncols=3)
        
        print(f'\nLogFC threshold: {lfc}' + f' / p-value threshold: {pval}'*(de_pval_col is not None))
        diff_exp_data = DiffExpData(de_data=de_data, kin_type=kin_type,
                                    lfc_col=de_lfc_col, lfc_thresh=lfc,
                                    pval_col=de_pval_col, pval_thresh=pval,
                                    seq_col=seq_col, drop_de_na=drop_de_na,
                                    **diff_exp_kwargs)
        if kl_method in ['score','score_rank']:
            diff_exp_data.submit_scores(scores, suppress_messages=suppress_warnings)
        elif kl_method in ['percentile','percentile_rank']:
            diff_exp_data.submit_percentiles(percentiles, suppress_messages=suppress_warnings)
        
        enrich_results = diff_exp_data.kinase_enrichment(kl_method=kl_method, kl_thresh=kl_thresh,
                                                       **enrichment_kwargs)
            
        enrich_results.plot_down_up_comb_volcanos(plot_sig_lff=ke_plot_sig_lff, plot_sig_pval=ke_plot_sig_pval, kinases=kinases,
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
    