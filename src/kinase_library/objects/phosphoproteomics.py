"""
##########################################
# The Kinase Library - Phosphoproteomics #
##########################################
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.stats import multitest
from tqdm import tqdm
import gseapy as gp
tqdm.pandas()

from ..utils import _global_vars, exceptions, utils
from ..modules import data, enrichment
from . import core
from ..logger import logger

#%%

class PhosphoProteomics(object):
    """
    Class for phosphoproteomics data.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with the phosphoproteomics data.
    seq_col : str, optional
        Column with the sequences. The default is None (will be set as _global_vars.default_seq_col).
    pad : tuple, optional
        How many padding '_' to add from each side of the substrates. The default is False.
    pp : bool, optional
        Phospho-residues (s/t/y). The default is False.
    drop_invalid_subs : bool, optional
        Drop rows with invalid substrates. The default is True.
    new_seq_phos_res_cols : bool, optional
        Create a new sequence column or phosphorylated residue column even if they already exists. The default is True.
        Sequence column name: _global_vars.default_seq_col.
        Phosphorylated residue column name: 'phos_res'.
    suppress_warnings : bool, optional
        Do not print warnings. The default is False.

    Examples
    -------
    >>> data = pd.read_csv('./../databases/substrates/Kinase_Substrate_Dataset_count_07_2021.txt', sep='\t', skiprows=3)
    >>> pps = kl.PhosphoProteomics(data)
    >>> pps.data
              KINASE KIN_ACC_ID   GENE  ...     CST_CAT# phos_res     SITE_+/-7_AA
        0      DYRK2     Q5U4C9  Dyrk2  ...          NaN        s  LGSSRPSsAPGMLPL
        1       PAK2     Q64303   Pak2  ...  9128; 98195        s  RTPGRPLsSYGMDSR
        2       PAK2     Q64303   Pak2  ...          NaN        s  GVRRRRLsNVSLTGL
        3       PAK2     Q64303   Pak2  ...          NaN        s  LHCLRRDsHKIDNYL
        4       PAK2     Q64303   Pak2  ...          NaN        s  IRCLRRDsHKVDNYL
             ...        ...    ...  ...          ...      ...              ...
        21387   ULK2     Q8IYT8   ULK2  ...          NaN        s  QRVLDTSsLTQSAPA
        21388   ULK2     Q8IYT8   ULK2  ...          NaN        s  DTSSLTQsAPASPTN
        21389   ULK2     Q8IYT8   ULK2  ...          NaN        s  LAQPINFsVSLSNSH
        21390   ULK2     Q8IYT8   ULK2  ...        13857        s  ESSPILTsFELVKVP
        21391   ULK2     Q8IYT8   ULK2  ...          NaN        s  THRRMVVsMPNLQDI
    >>> pps.substrates
        0        LGSSRPSsAPGMLPL
        1        RTPGRPLsSYGMDSR
        2        GVRRRRLsNVSLTGL
        3        LHCLRRDsHKIDNYL
        4        IRCLRRDsHKVDNYL
                      ...
        21387    QRVLDTSsLTQSAPA
        21388    DTSSLTQsAPASPTN
        21389    LAQPINFsVSLSNSH
        21390    ESSPILTsFELVKVP
        21391    THRRMVVsMPNLQDI
    """
        
    def __init__(self, data, seq_col=None,
                 pad=False, pp=False,
                 drop_invalid_subs=True,
                 new_seq_phos_res_cols=True,
                 suppress_warnings=False):
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError('\'data\' must be a pd.DataFrame.')
        
        if seq_col is None:
            seq_col = _global_vars.default_seq_col
        
        if drop_invalid_subs:
            processed_data,omited_entries = utils.filter_invalid_subs(data=data, seq_col=seq_col, suppress_warnings=suppress_warnings)
        else:
            processed_data = data.copy()
            omited_entries = []
        self.omited_entries = omited_entries
        if len(omited_entries)>0 and not suppress_warnings:
            print('Use the \'omited_entries\' attribute to view dropped enteries due to invalid sequences.')
        
        subs_list = processed_data[seq_col]
        if pad:
            subs_list = processed_data[seq_col].apply(lambda x: '_'*pad[0] + x + '_'*pad[1])
        subs_list = subs_list.apply(utils.sequence_to_substrate, pp=pp, validate_phos_res=drop_invalid_subs, validate_aa=drop_invalid_subs)
        
        phos_res = subs_list.str.lower().str[7]
        
        if new_seq_phos_res_cols:
            processed_data = processed_data.rename({_global_vars.default_seq_col: 'ORIGINAL_'+_global_vars.default_seq_col, 'phos_res': 'original_phos_res'}, axis=1)
        processed_data['phos_res'] = phos_res
        processed_data[_global_vars.default_seq_col] = subs_list
        
        self.data = processed_data
        self.original_data = data
        self.seq_col = seq_col
        self.substrates = processed_data[_global_vars.default_seq_col]
        self.phos_res = processed_data['phos_res']
        self.pp = pp
        
        self.ser_thr_data = processed_data[processed_data['phos_res'].isin(['S','T','s','t'])]
        self.ser_thr_substrates = self.ser_thr_data[_global_vars.default_seq_col]
        self._ser_thr_phos_res = self.ser_thr_data['phos_res']
        self.tyrosine_data = processed_data[processed_data['phos_res'].isin(['Y','y'])]
        self.tyrosine_substrates = self.tyrosine_data[_global_vars.default_seq_col]
        self._tyrosine_phos_res = self.tyrosine_data['phos_res']
    
    
    @classmethod
    def from_file(cls, data_file, seq_col=None, pad=False, pp=False, drop_invalid_subs=True, new_seq_phos_res_cols=True, suppress_warnings=False, **file_args):
        """
        Create PhosphoProteomics object from file.
    
        Parameters
        ----------
        data_file : str
            Phosphoproteomics file.
        seq_col : str, optional
            Column with the sequences. The default is None (will be set as _global_vars.default_seq_col).
        pad : tuple, optional
            How many padding '_' to add from each side of teh substrates. The default is False.
        pp : bool, optional
            Phospho-residues (s/t/y). The default is False.
        drop_invalid_subs : bool, optional
            Drop rows with invalid substrates. The default is True.
        new_seq_phos_res_cols : bool, optional
            Create a new sequence column or phosphorylated residue column even if they already exists. The default is True.
            Sequence column name: _global_vars.default_seq_col.
        suppress_warnings : bool, optional
            Do not print warnings. The default is False.
        **file_args : args
            Key arguments for pd.read_csv().
        
        Returns
        -------
        pps : kl.PhosphoProteomics
            PhosphoProteomics object with the data from the file.
        
        Examples
        -------
        >>> pps = kl.PhosphoProteomics(data_file='./../databases/substrates/Kinase_Substrate_Dataset_count_07_2021.txt', skiprows=3)
        >>> pps.data
                  KINASE KIN_ACC_ID   GENE  ...     CST_CAT# phos_res     SITE_+/-7_AA
            0      DYRK2     Q5U4C9  Dyrk2  ...          NaN        s  LGSSRPSsAPGMLPL
            1       PAK2     Q64303   Pak2  ...  9128; 98195        s  RTPGRPLsSYGMDSR
            2       PAK2     Q64303   Pak2  ...          NaN        s  GVRRRRLsNVSLTGL
            3       PAK2     Q64303   Pak2  ...          NaN        s  LHCLRRDsHKIDNYL
            4       PAK2     Q64303   Pak2  ...          NaN        s  IRCLRRDsHKVDNYL
                 ...        ...    ...  ...          ...      ...              ...
            21387   ULK2     Q8IYT8   ULK2  ...          NaN        s  QRVLDTSsLTQSAPA
            21388   ULK2     Q8IYT8   ULK2  ...          NaN        s  DTSSLTQsAPASPTN
            21389   ULK2     Q8IYT8   ULK2  ...          NaN        s  LAQPINFsVSLSNSH
            21390   ULK2     Q8IYT8   ULK2  ...        13857        s  ESSPILTsFELVKVP
            21391   ULK2     Q8IYT8   ULK2  ...          NaN        s  THRRMVVsMPNLQDI
        >>> pps.substrates
            0        LGSSRPSsAPGMLPL
            1        RTPGRPLsSYGMDSR
            2        GVRRRRLsNVSLTGL
            3        LHCLRRDsHKIDNYL
            4        IRCLRRDsHKVDNYL
                          ...
            21387    QRVLDTSsLTQSAPA
            21388    DTSSLTQsAPASPTN
            21389    LAQPINFsVSLSNSH
            21390    ESSPILTsFELVKVP
            21391    THRRMVVsMPNLQDI
        """
        
        file_type = data_file.split('.')[-1]
                    
        if file_type == 'parquet':
            data = pq.read_table(data_file).to_pandas()
        elif file_type in ['xlsx','xls']:
            data = pd.read_excel(data_file, **file_args)
        elif file_type == 'csv':
            data = pd.read_csv(data_file, **file_args)
        else:
            data = pd.read_csv(data_file, sep = '\t', **file_args)
        
        if seq_col is None:
            seq_col = _global_vars.default_seq_col
            
        pps = cls(data, seq_col=seq_col, pad=pad, pp=pp, drop_invalid_subs=drop_invalid_subs, new_seq_phos_res_cols=new_seq_phos_res_cols, suppress_warnings=suppress_warnings)
        
        return(pps)
    
        
    def _calculate_subs_binary_matrix(self, kin_type=['ser_thr','tyrosine'], pp=None, pos=None):
        """
        Making a binary matrix for a substrate.
    
        Parameters
        ----------
        kin_type : str or list, optional
            Kinase type. The default is ['ser_thr','tyrosine'].
        pp : bool, optional
            Phospho-priming residues (s/t/y). The default is None (will be determined by the object).
        pos : list, optional
            List of positions to use in the matrix rows. The default is None.

        Returns
        -------
        Setting self.*kin_type*_bin_matrix attribute for binary matrix.
        """
        
        if isinstance(kin_type, str):
            kin_type = [kin_type]
            
        if pp is None:
            pp = self.pp
        
        for kt in kin_type:
            exceptions.check_kin_type(kt)
            
            aa_labels = data.get_aa()
            if pos is None:
                pos = data.get_positions(kt)
            
            substrates = getattr(self, kt + '_substrates')
            
            subs_mat = utils.sub_binary_matrix(substrates, aa=aa_labels, pos=pos, pp=pp)
            setattr(self, '_' + kt + '_bin_matrix', subs_mat)
    
    
    def score(self, kin_type=None, kinases=None, st_fav=True,
              non_canonical=False, values_only=False, log2_score=True,
              pos=None, round_digits=3, return_values=True):
        """
        Calculate score of the phosphoproteomics data for the given kinases.
        
        Score is being computed in a vectorized way:
            1. Making binary matrix for the substrates.
            2. Converting kinase matrix (norm-scaled) to log2
            3. Performing dot-product (summing the corresponding log2 of the kinase matrix)

        Parameters
        ----------
        kin_type : str, optional
            Kinase type. The default is None.
            If not specified, will be inferred from the first kinase in the list.
        kinases : str or list, optional
            List of kinase names to score by.
        st_fav : bool, optional
            S/T favorability. The default is True.
        non_canonical : bool, optional
            Return also non-canonical kinases. For tyrosine kinases only. The default is False.
        values_only : bool, optional
            Return only score values (substrates as index, kinases as columns). The default is False.
        log2_score : bool, optional
            Return scores as log2. The default is True.
        pos : list, optional
            List of positions to use in the matrix rows. The default is None.
        round_digits : int, optional
            Number of decimal digits. The default is 4.
        return_values : bool, optional
            If False, will set attributes but will not return values. The default is True.

        Returns
        -------
        data_score_output : pd.DataFrame
            Original data with:
                * additional column for the phospho-residue
                * additional column with the -/+7 amino acids substrate
                * scores for all specificed kinases
        """
        
        if all(v is None for v in [kin_type, kinases]):
            raise ValueError('Either list of kinases or kinase type must be provided.')
        
        if kinases is None:
            kinases = data.get_kinase_list(kin_type, non_canonical=non_canonical)
        elif isinstance(kinases, str):
            kinases = [kinases]
        
        kinases = [x.upper() for x in kinases]
        exceptions.check_kin_name(kinases)
        
        if kin_type is None:
            kin_type = data.get_kinase_type(kinases[0])
        else:
            exceptions.check_kin_type(kin_type)
        exceptions.check_kin_list_type(kinases, kin_type=kin_type)
        
        print('Scoring '+str(len(getattr(self,kin_type+'_substrates')))+' '+kin_type+' substrates')
        logger.info('Scoring '+str(len(getattr(self,kin_type+'_substrates')))+' '+kin_type+' substrates')
        if not hasattr(self, '_' + kin_type + '_bin_matrix'):
            self._calculate_subs_binary_matrix(kin_type=kin_type, pp=self.pp, pos=pos)
        subs_bin_mat = getattr(self, '_' + kin_type + '_bin_matrix')
        
        # Using table with all the matrices concatenated (log2)
        kin_mat_log2 = data.get_multiple_matrices(kinases, kin_type=kin_type, mat_type='log2', pos=pos)

        # matrices are in log2 space
        score_log2 = pd.DataFrame(np.dot(subs_bin_mat,kin_mat_log2.transpose()), index = getattr(self, kin_type + '_substrates'), columns = kinases).round(round_digits)
        if (kin_type == 'ser_thr') and st_fav:
            st_fav_scores = data.get_st_fav(kinases)[getattr(self, '_' + kin_type + '_phos_res').str.upper()].transpose()
            st_fav_scores.index = score_log2.index
            st_fav_scores_log2 = np.log2(st_fav_scores)
            score_log2 = score_log2 + st_fav_scores_log2
        score = np.power(2,score_log2)
        
        if log2_score:
            score_output = score_log2
        else:
            score_output = score
        
        score_output = score_output.round(round_digits)
        score_rank_output = score_output.rank(method='min', ascending=False, axis=1).astype(int)

        data_index = getattr(self, kin_type + '_data').index
        data_score_output = pd.concat([getattr(self, kin_type + '_data').reset_index(drop=True),score_output.reset_index(drop=True)], axis=1)
        data_score_output.index = data_index
        
        setattr(self, kin_type+'_scores', score_output)
        setattr(self, kin_type+'_score_ranks', score_rank_output)
        setattr(self, kin_type+'_scored_kins', kinases)
        
        if return_values:
            if values_only:
                return(score_output)
            return(data_score_output)
    
    
    def percentile(self, kin_type=None, kinases=None,
                   st_fav=True, non_canonical=False,
                   subs_scores=None, subs_scores_format=None,
                   values_only=False, customized_scored_phosprot=None,
                   pos=None, phosprot_path='./../databases/substrates',
                   round_digits=2, return_values=True):
        """
        Calculate the percentile score of the phosphoproteomics data for the given kinases.
        
        After score is being computed, the percentile of that score is being
        computed based on a basal scored phosphoproteome.

        Parameters
        ----------
        kin_type : str, optional
            Kinase type. The default is None.
            If not specified, will be inferred from the first kinase in the list.
        kinases : str or list, optional
            List of kinase names to score by.
        st_fav : bool, optional
            S/T favorability. The default is True.
        non_canonical : bool, optional
            Return also non-canonical kinases. For tyrosine kinases only. The default is False.
        subs_scores : pd.DataFrame, optional
            Optional input scores for all the substrates (as index) and kinases (as columns). The default is None.
        subs_scores_format : str, optional
            Score format if 'subs_scores' is provided ('linear' or 'log2'). The default is None.
        values_only : bool, optional
            Return only percentile values (substrates as index, kinases as columns). The default is False.
        non_canonical : bool, optional
            Return also non-canonical kinases. For tyrosine kinases only. The default is False.
        customized_scored_phosprot : kl.ScoredPhosphoProteome, optional
            Customized phosphoproteome object. The default is None.
        pos : list, optional
            List of positions to use in the matrix rows. The default is None.
        phosprot_path : str, optional
            Path to scored phosphoproteome files. The default is './../databases/substrates/scored_phosprots'.
        round_digits : int, optional
            Number of decimal digits. The default is 2.
        return_values : bool, optional
            If False, will set attributes but will not return values. The default is True.

        Returns
        -------
        data_percent_output : pd.DataFrame
            Original data with:
                * additional column for the phospho-residue
                * additional column with the -/+7 amino acids substrate
                * percentiles for all specificed kinases
        """
        
        if all(v is None for v in [kin_type, kinases]):
            raise ValueError('Either list of kinases or kinase type must be provided.')
        
        if kinases is None:
            kinases = data.get_kinase_list(kin_type, non_canonical=non_canonical)
        elif isinstance(kinases, str):
            kinases = [kinases]
        
        kinases = [x.upper() for x in kinases]
        exceptions.check_kin_name(kinases)
        
        if kin_type is None:
            kin_type = data.get_kinase_type(kinases[0])
        else:
            exceptions.check_kin_type(kin_type)
        exceptions.check_kin_list_type(kinases, kin_type=kin_type)
        
        percent_output = []

        if subs_scores is None:
            if hasattr(self, kin_type+'_scores'):
                score = getattr(self, kin_type+'_scores')
                if (set(kinases)-set(score.columns)):
                    score = self.score(kinases=kinases, kin_type=kin_type, values_only=True, log2_score=True, st_fav=st_fav, non_canonical=non_canonical, pos=pos)
            else:
                score = self.score(kinases=kinases, kin_type=kin_type, values_only=True, log2_score=True, st_fav=st_fav, non_canonical=non_canonical, pos=pos)
        else:
            if subs_scores_format is None:
                raise ValueError('Please specify the format of input score data (\'subs_scores_format\').')
            elif subs_scores_format not in ['linear','log2']:
                raise ValueError('Please provide valid value for \'subs_scores_format\': \'linear\' or \'log2\'.')
            
            if (subs_scores_format == 'linear'):
                score = np.log2(subs_scores)
            else:
                score = subs_scores.copy()
        
        if len(score) == 0: # Data is empty - return empty dataframe
            percent_output = score.copy()
            setattr(self, kin_type+'_percentiles', percent_output)
            setattr(self, kin_type+'_percentile_ranks', percent_output)
            setattr(self, kin_type+'_percentiled_kins', percent_output)
            data_percent_output = pd.concat([getattr(self, kin_type + '_data').reset_index(drop=True),percent_output.reset_index(drop=True)], axis=1)
            if return_values:
                if values_only:
                    return(percent_output)
                return(data_percent_output)
        
        if customized_scored_phosprot is not None:
            all_scored_phosprot = customized_scored_phosprot
        else:
            all_scored_phosprot = core.ScoredPhosphoProteome(phosprot_name=_global_vars.phosprot_name, phosprot_path=phosprot_path)
        
        if kin_type is None:
            kin_type = data.get_kinase_type(kinases[0])
        else:
            exceptions.check_kin_type(kin_type)
        exceptions.check_kin_list_type(kinases, kin_type=kin_type)
        
        if kin_type == 'ser_thr':
            scored_phosprot = all_scored_phosprot.ser_thr_scores
        elif kin_type == 'tyrosine':
            scored_phosprot = all_scored_phosprot.tyrosine_scores
        else:
            raise ValueError('Wrong kinase type.')
        scored_phosprot = scored_phosprot.loc[:,kinases] # only for requested kinases if subset
        
        # If scored phopshoproteome is linear values - converting it to log2 values
        if not all_scored_phosprot.log2_values:
            scored_phosprot = np.log2(scored_phosprot)
        
        print('Calculating percentile for '+str(len(getattr(self,kin_type+'_substrates')))+' '+kin_type+' substrates')
        logger.info('Calculating percentile for '+str(len(getattr(self,kin_type+'_substrates')))+' '+kin_type+' substrates')
        tqdm.pandas()
        percent_output = scored_phosprot.progress_apply(lambda x: x.sort_values().searchsorted(score[x.name], side='right'))/len(scored_phosprot)*100
        percent_output.index = score.index
        
        percent_output = percent_output.round(round_digits)
        percent_rank_output = percent_output.rank(method='min', ascending=False, axis=1).astype(int)
        
        data_index = getattr(self, kin_type + '_data').index
        data_percent_output = pd.concat([getattr(self, kin_type + '_data').reset_index(drop=True),percent_output.reset_index(drop=True)], axis=1)
        data_percent_output.index = data_index
        
        setattr(self, kin_type+'_percentiles', percent_output)
        setattr(self, kin_type+'_percentile_ranks', percent_rank_output)
        setattr(self, kin_type+'_percentiled_kins', kinases)
        
        self.phosprot_name = _global_vars.phosprot_name
        
        if return_values:
            if values_only:
                return(percent_output)
            return(data_percent_output)
    
    
    def rank(self, metric, kin_type=None, kinases=None,
             st_fav=True, non_canonical=False,
             pos=None, rank_kinases=None, values_only=False,
             score_round_digits=3, percentile_round_digits=2):
        """
        Calculate ranks of kinases based on scoring metric.

        Parameters
        ----------
        metric : str
            Scoring metric ('score' or 'percentile').
        kin_type : str, optional
            Kinase type. The default is None.
            If not specified, will be inferred from the first kinase in the list.
        kinases : str or list, optional
            List of kinase names to display in the rank results.
        st_fav : bool, optional
            S/T favorability. The default is True.
        non_canonical : bool, optional
            Return also non-canonical kinases. For tyrosine kinases only. The default is False.
        pos : list, optional
            List of positions to use in the matrix rows. The default is None.
        rank_kinases : str or list
            List of kinase names to rank by (subseting the kinome).
        values_only : bool, optional
            Return only percentile values (substrates as index, kinases as columns). The default is False.
        score_round_digits : int, optional
            Number of decimal digits for score. The default is 4.
        percentile_round_digits : int, optional
            Number of decimal digits for percentile. The default is 2.

        Raises
        ------
        ValueError
            Raise error if both kinase type and list of kinases are not specified.

        Returns
        -------
        ranks : pd.DataFrame
            Ranks of the kinases based on the specified scoring metric.
        """
        
        if all(v is None for v in [kin_type, kinases]):
            raise ValueError('Either list of kinases or kinase type must be provided.')
        if kinases is None:
            if rank_kinases:
                kinases = rank_kinases
            else:
                kinases = data.get_kinase_list(kin_type, non_canonical=non_canonical)
        elif isinstance(kinases, str):
            kinases = [kinases]
        kinases = [x.upper() for x in kinases]
        if kin_type is None:
            kin_type = data.get_kinase_type(kinases[0])
        else:
            exceptions.check_kin_type(kin_type)
        if rank_kinases is None:
            rank_kinases = data.get_kinase_list(kin_type, non_canonical=non_canonical)
        elif isinstance(rank_kinases, str):
            rank_kinases = [rank_kinases]
        rank_kinases = [x.upper() for x in rank_kinases]
        if [x for x in kinases if x not in rank_kinases]:
            raise ValueError('kinases must be a subset of rank_kinases.')
        exceptions.check_kin_list_type(rank_kinases, kin_type=kin_type)
        exceptions.check_kin_list_type(kinases, kin_type=kin_type)
        exceptions.check_scoring_metric(metric)
        
        if metric == 'score':
            self.score(kin_type=kin_type, kinases=rank_kinases, st_fav=st_fav, non_canonical=non_canonical, return_values=False, pos=pos, round_digits=score_round_digits)
        elif metric == 'percentile':
            self.percentile(kin_type=kin_type, kinases=rank_kinases, st_fav=st_fav, non_canonical=non_canonical, return_values=False, pos=pos, round_digits=percentile_round_digits)
        
        rank_output = getattr(self, kin_type+'_'+metric+'_ranks')[kinases]
        
        data_index = getattr(self, kin_type + '_data').index
        data_rank_output = pd.concat([getattr(self, kin_type + '_data').reset_index(drop=True),rank_output.reset_index(drop=True)], axis=1)
        data_rank_output.index = data_index
        
        if values_only:
            return(rank_output)
        return(data_rank_output)

    
    def predict(self, metric=['score','percentile'], kin_type=None, kinases=None,
                st_fav=True, non_canonical=False, values_only=False,
                score_promiscuity_threshold=1, percentile_promiscuity_threshold=90,
                pos=None, score_round_digits=3, percentile_round_digits=2):
        """
        Generating full prediction table (scores, score-ranks, percentiles, percentile-ranks)

        Parameters
        ----------
        metric : str or list, optional
            Scoring metric ('score' or 'percentile'). The default is both.
        kin_type : str, optional
            Kinase type. The default is None.
            If not specified, will be inferred from the first kinase in the list.
        kinases : str or list, optional
            List of kinase names to score by.
        st_fav : bool, optional
            S/T favorability. The default is True.
        non_canonical : bool, optional
            Return also non-canonical kinases. For tyrosine kinases only. The default is False.
        values_only : bool, optional
            Return only percentile values (substrates as index, kinases as columns). The default is False.
        score_promiscuity_threshold : float, optional
            Score threshold above which kinases are considered predicted.
        percentile_promiscuity_threshold : float, optional
            Percentile threshold above which kinases are considered predicted.
        pos : list, optional
            List of positions to use in the matrix rows. The default is None.
        score_round_digits : int, optional
            Number of decimal digits for score. The default is 4.
        percentile_round_digits : int, optional
            Number of decimal digits for percentile. The default is 2.

        Raises
        ------
        ValueError
            Raise error if both kinase type and list of kinases are not specified.

        Returns
        -------
        prediction_output : pd.DataFrame
            Table with all four outputs (scores, score-ranks, percentiles, percentile-ranks) for every kinase.

        """

        if all(v is None for v in [kin_type, kinases]):
            raise ValueError('Either list of kinases or kinase type must be provided.')
        if kinases is None:
            kinases = data.get_kinase_list(kin_type, non_canonical=non_canonical)
        elif isinstance(kinases, str):
            kinases = [kinases]
        kinases = [x.upper() for x in kinases]
        if kin_type is None:
            kin_type = data.get_kinase_type(kinases[0])
        else:
            exceptions.check_kin_type(kin_type)
        exceptions.check_kin_list_type(kinases, kin_type=kin_type)

        if isinstance(metric, str):
            metric = [metric]
        
        prediction_output = pd.DataFrame(index=getattr(self, kin_type+'_substrates'))

        if 'score' in metric:
            score_ranks = self.rank('score', kin_type=kin_type, kinases=kinases, st_fav=st_fav, non_canonical=non_canonical, pos=pos, values_only=True, score_round_digits=score_round_digits)
            scores = getattr(self, kin_type+'_scores')[kinases]
            score_promis = self.promiscuity_index(kin_type=kin_type, kinases=kinases, metric='score', threshold=score_promiscuity_threshold, pos=pos, st_fav=st_fav, non_canonical=non_canonical, values_only=True)
            prediction_output = pd.concat([prediction_output, score_promis], axis=1)
        if 'percentile' in metric:
            percentile_ranks = self.rank('percentile', kin_type=kin_type, kinases=kinases, st_fav=st_fav, non_canonical=non_canonical, pos=pos, values_only=True, percentile_round_digits=percentile_round_digits)
            percentiles = getattr(self, kin_type+'_percentiles')[kinases]
            percent_promis = self.promiscuity_index(kin_type=kin_type, kinases=kinases, metric='percentile', threshold=percentile_promiscuity_threshold, pos=pos, st_fav=st_fav, non_canonical=non_canonical, values_only=True)
            prediction_output = pd.concat([prediction_output, percent_promis], axis=1)
        
        for kin in kinases:
            if 'score' in metric:
                score_df = pd.DataFrame({kin+'_score': scores[kin], kin+'_score_rank': score_ranks[kin]})
                prediction_output = pd.concat([prediction_output, score_df], axis=1)

            if 'percentile' in metric:
                percentile_df = pd.DataFrame({kin+'_percentile': percentiles[kin], kin+'_percentile_rank': percentile_ranks[kin]})
                prediction_output = pd.concat([prediction_output, percentile_df], axis=1)
        
        data_index = getattr(self, kin_type + '_data').index
        data_prediction_output = pd.concat([getattr(self, kin_type + '_data').reset_index(drop=True),prediction_output.reset_index(drop=True)], axis=1)
        data_prediction_output.index = data_index
        
        if values_only:
            return(prediction_output)
        return(data_prediction_output)
    
    
    def promiscuity_index(self, kin_type=None, kinases=None,
                          metric='percentile', threshold=90, pos=None,
                          st_fav=True, non_canonical=False,
                          values_only=False):
        """
        Generating Promiscuity Index for list of substrates.

        Parameters
        ----------
        kin_type : str, optional
            Kinase type. The default is None.
            If not specified, will be inferred from the first kinase in the list.
        kinases : str or list, optional
            List of kinase names to score by.
        metric : str, optional
            Scoring metric ('score' or 'percentile').
        threshold : float, optional
            Prediction threshold value above which kinases are considered predicted.
        pos : list, optional
            List of positions to use in the matrix rows. The default is None.
        st_fav : bool, optional
            S/T favorability. The default is True.
        non_canonical : bool, optional
            Return also non-canonical kinases. For tyrosine kinases only. The default is False.
        values_only : bool, optional
            Return only promiscuity values. The default is False.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        if all(v is None for v in [kin_type, kinases]):
            raise ValueError('Either list of kinases or kinase type must be provided.')
        if kinases is None:
            kinases = data.get_kinase_list(kin_type, non_canonical=non_canonical)
        elif isinstance(kinases, str):
            kinases = [kinases]
        kinases = [x.upper() for x in kinases]
        if kin_type is None:
            kin_type = data.get_kinase_type(kinases[0])
        else:
            exceptions.check_kin_type(kin_type)
        exceptions.check_kin_list_type(kinases, kin_type=kin_type)
        
        if not hasattr(self, kin_type+'_'+metric+'s'):
            if metric == 'score':
                self.score(kin_type=kin_type, kinases=kinases, st_fav=st_fav, non_canonical=non_canonical, pos=pos, return_values=False)
            elif metric == 'percentile':
                self.percentile(kin_type=kin_type, kinases=kinases, st_fav=st_fav, non_canonical=non_canonical, pos=pos, return_values=False)
        
        metric_data = getattr(self, kin_type+'_'+metric+'s')
        promis_idx = (metric_data >= threshold).sum(axis=1)
        promis_idx.name = metric.capitalize() + ' Promiscuity Index'
        
        setattr(self, kin_type+'_'+metric+'_'+'promiscuity_index', promis_idx)
        
        data_index = getattr(self, kin_type + '_data').index
        data_promis_output = pd.concat([getattr(self, kin_type + '_data').reset_index(drop=True),promis_idx.reset_index(drop=True)], axis=1)
        data_promis_output.index = data_index
        
        if values_only:
            return(promis_idx)
        return(data_promis_output)
        
    
    def submit_scores(self, kin_type, scores, suppress_messages=False):
        """
        Submitting scores for the substrates.

        Parameters
        ----------
        kin_type : str
            Kinase type.
        scores : pd.DataFrame
            Dataframe with site scores.
            Index must contain all the values in 'seq_col' with no duplicates.
            Columns must contain valid kinase names.
        suppress_messages : bool, optional
            Suppress messages. The default is False.

        Returns
        -------
        None.
        """
        
        exceptions.check_kin_type(kin_type)
        if ~(scores.columns.isin(data.get_kinase_list(kin_type, non_canonical=True)).all()):
            raise ValueError(f'Score columns must contain only valid {kin_type} kinases. Use kl.get_kinase_list() to get the list of valid kinases.')
        
        data_subs = getattr(self, kin_type + '_substrates')
        scores_unique = scores[~scores.index.duplicated(keep='first')]
        
        if not set(data_subs) <= set(scores_unique.index):
            raise ValueError('Scores must be provided for all substrates in the data.')
        
        if scores_unique.isna().any().any():
            raise ValueError('Some score values are missing.')
            
        subs_scores = scores_unique.loc[data_subs]
        
        score_rank = subs_scores.rank(method='min', ascending=False, axis=1).astype(int)
        setattr(self, kin_type+'_scores', subs_scores)
        setattr(self, kin_type+'_score_ranks', score_rank)
        
        if not suppress_messages:
            print('Scores submitted successfully.')
        
        
    def submit_percentiles(self, kin_type, percentiles, phosprot_name=None, suppress_messages=False):
        """
        Submitting percentiles for the substrates.

        Parameters
        ----------
        kin_type : str
            Kinase type.
        percentiles : pd.DataFrame
            Dataframe with site percentiles.
            Index must contain all the values in 'seq_col' with no duplicates.
            Columns must contain valid kinase names.
        phosprot_name : str, optional
            Name of phosphoproteome database.
        suppress_messages : bool, optional
            Suppress messages. the default is False.

        Returns
        -------
        None.
        """
        
        exceptions.check_kin_type(kin_type)
        if ~(percentiles.columns.isin(data.get_kinase_list(kin_type, non_canonical=True)).all()):
            raise ValueError(f'Percentile columns must contain only valid {kin_type} kinases. Use kl.get_kinase_list() to get the list of valid kinases.')
        if (percentiles.max().max()>100) or (percentiles.min().min()<0):
            raise ValueError('Percentile values must be between 0-100.')

        data_subs = getattr(self, kin_type + '_substrates')
        percentiles_unique = percentiles[~percentiles.index.duplicated(keep='first')]
        
        if not set(data_subs) <= set(percentiles_unique.index):
            raise ValueError('Percentiles must be provided for all substrates in the data.')
        
        if percentiles_unique.isna().any().any():
            raise ValueError('Some percentile values are missing.')
        
        subs_percentiles = percentiles_unique.loc[data_subs]
        
        percentile_rank = subs_percentiles.rank(method='min', ascending=False, axis=1).astype(int)
        setattr(self, kin_type+'_percentiles', subs_percentiles)
        setattr(self, kin_type+'_percentile_ranks', percentile_rank)
        
        if phosprot_name is None:
            phosprot_name = _global_vars.phosprot_name
        self.phosprot_name = phosprot_name
        
        if not suppress_messages:
            print('Percentiles submitted successfully.')
    
    
    def merge_data_scores(self, kin_type, score_type):
        """
        Merging phosphoproteome data and score data.

        Parameters
        ----------
        kin_type : str
            Kinase type (ser_thr or tyrosine).
        score_type : str
            Score type ('scores', 'score_ranks', 'percentiles', 'percentile_ranks').

        Returns
        -------
        merged_data : dataframe
            Merged dataframe of the phosphoproteome data and score data.
        """
        
        exceptions.check_kin_type(kin_type)
        exceptions.check_score_type(score_type)
        
        data = getattr(self, kin_type+'_data').set_index(_global_vars.default_seq_col, drop=False)
        scores = getattr(self, kin_type+'_'+score_type)
        merged_data = pd.concat([data,scores], axis=1)
        
        return(merged_data)
        