"""
##############################################
# Kinase Library - Phosphoproteomics Objects #
##############################################
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.stats import multitest
import pyarrow.parquet as pq
from tqdm import tqdm
from functools import partial
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
from ..logger import logger, TqdmToLoggerWithStatus
import gseapy as gp

from ..utils import _global_vars, exceptions, utils
from ..modules import data, enrichment
from . import core

#%%

class PhosphoProteomics(object):
    """
    Class for phosphoproteomics data.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with the phosphoproteomics data.
    seq_col : str, optional
        Column with the sequences. The default is 'SITE_+/-7_AA'.
    pad : tuple, optional
        How many padding '_' to add from each side of the substrates. The default is False.
    pp : bool, optional
        Phospho-residues (s/t/y). The default is False.
    drop_invalid_subs : bool, optional
        Drop rows with invalid substrates. The default is True.
    new_seq_phos_res_cols : bool, optional
        Create a new sequence column or phosphorylated residue column even if they already exists. The default is True.
        Sequence column name: 'SITE_+/-7_AA'.
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
        
    def __init__(self, data, seq_col='SITE_+/-7_AA',
                 pad=False, pp=False,
                 drop_invalid_subs=True,
                 new_seq_phos_res_cols=True,
                 suppress_warnings=False):
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError('\'data\' must be a pd.DataFrame.')
        
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
            processed_data = processed_data.rename({'SITE_+/-7_AA': 'ORIGINAL_SITE_+/-7_AA', 'phos_res': 'original_phos_res'}, axis=1)
        processed_data['phos_res'] = phos_res
        processed_data['SITE_+/-7_AA'] = subs_list
        
        self.data = processed_data
        self.original_data = data
        self.seq_col = seq_col
        self.substrates = processed_data['SITE_+/-7_AA']
        self.phos_res = processed_data['phos_res']
        self.pp = pp
        
        self.ser_thr_data = processed_data[processed_data['phos_res'].isin(['S','T','s','t'])]
        self.ser_thr_substrates = self.ser_thr_data['SITE_+/-7_AA']
        self._ser_thr_phos_res = self.ser_thr_data['phos_res']
        self.tyrosine_data = processed_data[processed_data['phos_res'].isin(['Y','y'])]
        self.tyrosine_substrates = self.tyrosine_data['SITE_+/-7_AA']
        self._tyrosine_phos_res = self.tyrosine_data['phos_res']
    
    
    @classmethod
    def from_file(cls, data_file, seq_col='SITE_+/-7_AA', pad=False, pp=False, drop_invalid_subs=True, new_seq_phos_res_cols=True, suppress_warnings=False, **file_args):
        """
        Create PhosphoProteomics object from file.
    
        Parameters
        ----------
        data_file : str
            Phosphoproteomics file.
        seq_col : str, optional
            Column with the sequences. The default is 'SITE_+/-7_AA'.
        pad : tuple, optional
            How many padding '_' to add from each side of teh substrates. The default is False.
        pp : bool, optional
            Phospho-residues (s/t/y). The default is False.
        drop_invalid_subs : bool, optional
            Drop rows with invalid substrates. The default is True.
        new_seq_phos_res_cols : bool, optional
            Create a new sequence column or phosphorylated residue column even if they already exists. The default is True.
            Sequence column name: 'SITE_+/-7_AA'.
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
    
    
    def score(self, kin_type=None, kinases=None, pp=None, st_fav=True,
              non_canonical=False, values_only=False, log2_score=True,
              pos=None, log2_matrix=False, round_digits=4,
              return_values=True):
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
        pp : bool, optional
            Phosphopriming. The default is None.
            If not specified, will be determined from the object.
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
        log2_matrix : bool, optional
            Whether kinase matrices are already log2 transformed. The default is False.
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
        
        if pp is None:
            pp = self.pp
        
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
            self._calculate_subs_binary_matrix(kin_type=kin_type, pp=pp, pos=pos)
        subs_bin_mat = getattr(self, '_' + kin_type + '_bin_matrix')
        
        # Using table with all the matrices concatenated (log2)
        kin_mat = data.get_multiple_matrices(kinases, kin_type=kin_type, mat_type='norm_scaled', pos=pos)
        if not log2_matrix:
            kin_mat_log2 = np.log2(kin_mat)
        else:
            kin_mat_log2 = kin_mat.copy()
        
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
                   pp=None, st_fav=True, non_canonical=False,
                   subs_scores=None, subs_scores_format=None,
                   values_only=False, customized_scored_phosprot=None,
                   pos=None, log2_matrix=False,
                   phosprot_path='./../databases/substrates',
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
        pp : bool, optional
            Phosphopriming. The default is None.
            If not specified, will be determined from the object.
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
        log2_matrix : bool, optional
            Whether kinase matrices are already log2 transformed. The default is False.
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
        
        if pp is None:
            pp = self.pp
        
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
            else:
                score = self.score(kinases=kinases, kin_type=kin_type, values_only=True, log2_score=True, pp=pp, st_fav=st_fav, non_canonical=non_canonical, log2_matrix=log2_matrix, pos=pos)
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
        
        tqdm_out = TqdmToLoggerWithStatus(logger)
        tqdm.pandas(file=tqdm_out, ascii=False)
        print('Calculating percentile for ' + str(len(getattr(self,kin_type+'_substrates')))+' '+kin_type+' substrates')
        logger.info('Calculating percentile for ' + str(len(getattr(self,kin_type+'_substrates')))+' '+kin_type+' substrates')
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
             pp=None, st_fav=True, non_canonical=False,
             pos=None, rank_kinases=None, values_only=False,
             score_round_digits=4, percentile_round_digits=2):
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
        pp : bool, optional
            Phosphopriming. The default is None.
            If not specified, will be determined from the object.
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
        
        if pp is None:
            pp = self.pp
            
        if metric == 'score':
            self.score(kin_type=kin_type, kinases=rank_kinases, pp=pp, st_fav=st_fav, non_canonical=non_canonical, return_values=False, pos=pos, round_digits=score_round_digits)
        elif metric == 'percentile':
            self.percentile(kin_type=kin_type, kinases=rank_kinases, pp=pp, st_fav=st_fav, non_canonical=non_canonical, return_values=False, pos=pos, round_digits=percentile_round_digits)
        
        rank_output = getattr(self, kin_type+'_'+metric+'_ranks')[kinases]
        
        data_index = getattr(self, kin_type + '_data').index
        data_rank_output = pd.concat([getattr(self, kin_type + '_data').reset_index(drop=True),rank_output.reset_index(drop=True)], axis=1)
        data_rank_output.index = data_index
        
        if values_only:
            return(rank_output)
        return(data_rank_output)

    
    def predict(self, metric=['score','percentile'], kin_type=None, kinases=None,
                pp=None, st_fav=True, non_canonical=False, values_only=False,
                pos=None, score_round_digits=4, percentile_round_digits=2):
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
        pp : bool, optional
            Phosphopriming. The default is None.
            If not specified, will be determined from the object.
        st_fav : bool, optional
            S/T favorability. The default is True.
        non_canonical : bool, optional
            Return also non-canonical kinases. For tyrosine kinases only. The default is False.
        values_only : bool, optional
            Return only percentile values (substrates as index, kinases as columns). The default is False.
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
        
        if pp is None:
            pp = self.pp

        prediction_output = pd.DataFrame(index=getattr(self, kin_type+'_substrates'))

        if 'score' in metric:
            score_ranks = self.rank('score', kin_type=kin_type, kinases=kinases, pp=pp, st_fav=st_fav, non_canonical=non_canonical, pos=pos, values_only=True, score_round_digits=score_round_digits)
            scores = getattr(self, kin_type+'_scores')[kinases]
        if 'percentile' in metric:
            percentile_ranks = self.rank('percentile', kin_type=kin_type, kinases=kinases, pp=pp, st_fav=st_fav, non_canonical=non_canonical, pos=pos, values_only=True, percentile_round_digits=percentile_round_digits)
            percentiles = getattr(self, kin_type+'_percentiles')[kinases]
        
        for kin in kinases:
            if 'score' in metric:
                score_df = pd.DataFrame(
                    {kin+'_score': scores[kin], kin+'_score_rank': score_ranks[kin]})
                prediction_output = pd.concat(
                    [prediction_output, score_df], axis=1)

            if 'percentile' in metric:
                percentile_df = pd.DataFrame(
                    {kin+'_percentile': percentiles[kin], kin+'_percentile_rank': percentile_ranks[kin]})
                prediction_output = pd.concat(
                    [prediction_output, percentile_df], axis=1)
        
        data_index = getattr(self, kin_type + '_data').index
        data_prediction_output = pd.concat([getattr(self, kin_type + '_data').reset_index(drop=True),prediction_output.reset_index(drop=True)], axis=1)
        data_prediction_output.index = data_index
        
        if values_only:
            return(prediction_output)
        return(data_prediction_output)
    
    
    def promiscuity_index(self, kin_type=None, kinases=None,
                          metric='percentile', threshold=90, pos=None,
                          pp=None, st_fav=True, non_canonical=False,
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
        pp : bool, optional
            Phosphopriming. The default is None.
            If not specified, will be determined from the object.
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
        
        if pp is None:
            pp = self.pp
            
        if not hasattr(self, kin_type+'_'+metric+'s'):
            if metric == 'score':
                self.score(kin_type=kin_type, kinases=kinases, pp=pp, st_fav=st_fav, non_canonical=non_canonical, pos=pos, return_values=False)
            elif metric == 'percentile':
                self.percentile(kin_type=kin_type, kinases=kinases, pp=pp, st_fav=st_fav, non_canonical=non_canonical, pos=pos, return_values=False)
        
        metric_data = getattr(self, kin_type+'_'+metric+'s')
        promis_idx = (metric_data >= threshold).sum(axis=1)
        promis_idx.name = 'Promiscuity Index'
        
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

        Raises
        ------
        ValueError
            Raise error if columns contain an invalid kinase name.

        Returns
        -------
        None.
        """
        
        exceptions.check_kin_type(kin_type)
        if ~(scores.columns.isin(data.get_kinase_list(kin_type, non_canonical=True)).all()):
            raise ValueError(f'Score columns must contain only valid {kin_type} kinases. Use kl.get_kinase_list() to get the list of valid kinases.')
        
        data_subs = getattr(self, kin_type + '_substrates')
        if scores.index.duplicated().any():
            if not suppress_messages:
                print(f'Warning: duplicated indices in the scores data. Dropping {scores.index.duplicated().sum()} duplicates (keeping first entry).')
            scores = scores[~scores.index.duplicated(keep='first')]
        
        if not set(data_subs) <= set(scores.index):
            raise ValueError('Scores must be provided for all substrates in the data.')
        
        subs_scores = scores.loc[data_subs]
        
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

        Raises
        ------
        ValueError
            Raise error if columns contain an invalid kinase name.
        ValueError
            Raise error if percentile values are not between 0-100.
        ValueError
            Raise error if length of percentile matrix is not the same as data matrix.

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
        if percentiles.index.duplicated().any():
            if not suppress_messages:
                print(f'Warning: duplicated indices in the percentiles data. Dropping {percentiles.index.duplicated().sum()} duplicates (keeping first entry).')
            percentiles = percentiles[~percentiles.index.duplicated(keep='first')]
        
        if not set(data_subs) <= set(percentiles.index):
            raise ValueError('Percentiles must be provided for all substrates in the data.')
        
        if percentiles.isna().any().any():
            raise ValueError('Some percentile values are missing.')
        
        subs_percentiles = percentiles.loc[data_subs]
        
        percentile_rank = subs_percentiles.rank(method='min', ascending=False, axis=1).astype(int)
        setattr(self, kin_type+'_percentiles', subs_percentiles)
        setattr(self, kin_type+'_percentile_ranks', percentile_rank)
        
        if phosprot_name is None:
            phosprot_name = _global_vars.phosprot_name
        else:
            if not suppress_messages:
                print(f'Phosphoproteome used: {phosprot_name}.')
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
        
        data = getattr(self, kin_type+'_data').set_index('SITE_+/-7_AA', drop=False)
        scores = getattr(self, kin_type+'_'+score_type)
        merged_data = pd.concat([data,scores], axis=1)
        
        return(merged_data)

#%%

class EnrichmentData(object):
    """
    Class for kinase enrichment data.

    Parameters
    ----------
    foreground : pd.DataFrame
        Dataframe with foreground substrates.
    background : pd.DataFrame
        Dataframe with background substrates.
    kin_type : str
        Kinase type ('ser_thr' or 'tyrosine').
    fg_seq_col : str, optional
        Substrates column name for the foreground data. The default is 'SITE_+/-7_AA'.
    bg_seq_col : str, optional
        Substrates column name for the background data. The default is 'SITE_+/-7_AA'.
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
        Sequence column name: 'SITE_+/-7_AA'. Phosphorylated residue column name: 'phos_res'.
    suppress_warnings : bool, optional
        Do not print warnings. The default is False.
    """

    def __init__(self, foreground, background, kin_type,
                 fg_seq_col='SITE_+/-7_AA', bg_seq_col='SITE_+/-7_AA',
                 fg_pad=False, bg_pad=False, fg_pp=False, bg_pp=False,
                 drop_invalid_subs=True,
                 new_seq_phos_res_cols=True,
                 suppress_warnings=False):
        
        if isinstance(foreground, (pd.Series, list)):
            foreground = utils.list_series_to_df(foreground, col_name=fg_seq_col)
        if isinstance(background, (pd.Series, list)):
            background = utils.list_series_to_df(background, col_name=bg_seq_col)
        
        self.fg_data = foreground
        self.bg_data = background
        self.kin_type = kin_type
        self.fg_pps = PhosphoProteomics(foreground, seq_col=fg_seq_col, pad=fg_pad, pp=fg_pp, drop_invalid_subs=drop_invalid_subs, new_seq_phos_res_cols=new_seq_phos_res_cols, suppress_warnings=suppress_warnings)
        self.bg_pps = PhosphoProteomics(background, seq_col=bg_seq_col, pad=bg_pad, pp=bg_pp, drop_invalid_subs=drop_invalid_subs, new_seq_phos_res_cols=new_seq_phos_res_cols, suppress_warnings=suppress_warnings)
        
    
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
        
        corrected_cont_table = cont_table[columns]
        pd.options.mode.chained_assignment = None # Turning off pandas SettingWithCopyWarning
        corrected_cont_table[corrected_cont_table.min(axis=1) == 0] = corrected_cont_table[corrected_cont_table.min(axis=1) == 0] + 0.5 # Applying Haldane correction (adding 0.5 to the cases with zero in one of the counts) - being used only for calculating log2(freq_factor), not for p-value
        pd.options.mode.chained_assignment = 'warn' # Turning on pandas SettingWithCopyWarning
        
        return(corrected_cont_table)
    
    
    def submit_scores(self, scores, data_type=['foreground','background'], suppress_messages=False):
        """
        Submitting scores for the foreground/background substrates.

        Parameters
        ----------
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
                self.fg_pps.submit_scores(kin_type=self.kin_type, scores=scores, suppress_messages=suppress_messages)
            elif dt in ['background','bg']:
                self.bg_pps.submit_scores(kin_type=self.kin_type, scores=scores, suppress_messages=suppress_messages)
        
        
    def submit_percentiles(self, percentiles, data_type=['foreground','background'], phosprot_name=None, suppress_messages=False):
        """
        Submitting percentiles for the foreground/background substrates.

        Parameters
        ----------
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
                self.fg_pps.submit_percentiles(kin_type=self.kin_type, percentiles=percentiles, phosprot_name=phosprot_name, suppress_messages=suppress_messages)
            elif dt in ['background','bg']:
                self.bg_pps.submit_percentiles(kin_type=self.kin_type, percentiles=percentiles, phosprot_name=phosprot_name, suppress_messages=suppress_messages)
        
        
    def kinase_enrichment(self, kl_method, kl_thresh,
                          kinases=None, pp=None, non_canonical=False,
                          enrichment_type='enriched', adj_pval=True, rescore=False):
        """
        Kinase enrichment analysis based on Fisher's exact test for foreground and background substarte lists.
    
        Parameters
        ----------
        kl_method : str
            Kinase Library scoring method ('score', 'score_rank', 'percentile', 'percentile_rank').
        kl_thresh : int
            The threshold to be used for the specified kl_method.
        kinases : list, optional
            If provided, kinase enrichment will only be calculated for the specified kinase list, otherwise, all kinases of the specified kin_type will be included. The default is None.
        pp : bool, optional
            Account for phospho-residues (s/t/y) in the sequence. The default is False.
        non_canonical : bool, optional
            Return also non-canonical kinases. For tyrosine kinases only. The default is False.
        enrichment_type : str, optional
            Direction of fisher's exact test for kinase enrichment ('enriched','depleted', or 'both').
        adj_pval : bool, optional
            If True use adjusted p-value for calculation of statistical significance. The default is True.
        rescore : bool, optional
            If True, Kinase Library scores or percentiles will be recalculated.
    
        Returns
        -------
        enrichemnt_results : pd.DataFrame
            pd.Dataframe with results of Kinase Enrichment for the specified KL method and threshold.
        """
        
        exceptions.check_kl_method(kl_method)
        exceptions.check_enrichment_type(enrichment_type)
        
        kin_type = self.kin_type
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
                self.fg_pps.score(kin_type=kin_type,kinases=kinases,pp=pp)
            elif not set(kinases)<=set(getattr(self.fg_pps, kin_type + '_' + data_att)):
                print('Not all kinase scores were provided. Re-calculating scores for foreground data')
                self.fg_pps.score(kin_type=kin_type,kinases=kinases,pp=pp)
            if not hasattr(self.bg_pps, kin_type + '_' + data_att) or rescore:
                self.bg_pps.score(kin_type=kin_type,kinases=kinases,pp=pp)
            elif not set(kinases)<=set(getattr(self.bg_pps, kin_type + '_' + data_att)):
                print('Not all kinase scores were provided. Re-calculating scores for background data')
                self.bg_pps.score(kin_type=kin_type,kinases=kinases,pp=pp)
        elif kl_method in ['percentile','percentile_rank']:
            if not hasattr(self.fg_pps, kin_type + '_' + data_att) or rescore:
                print('\nCalculating percentiles for foreground data')
                self.fg_pps.percentile(kin_type=kin_type,kinases=kinases,pp=pp)
                self.phosprot_name = _global_vars.phosprot_name
            elif not set(kinases)<=set(getattr(self.fg_pps, kin_type + '_' + data_att)):
                print('Not all kinase percentiles were provided. Re-calculating percentiles for foreground data')
                self.fg_pps.percentile(kin_type=kin_type,kinases=kinases,pp=pp)
                self.phosprot_name = _global_vars.phosprot_name
            if not hasattr(self.bg_pps, kin_type + '_' + data_att) or rescore:
                print('\nCalculating percentiles for background data')
                self.bg_pps.percentile(kin_type=kin_type,kinases=kinases,pp=pp)
                self.phosprot_name = _global_vars.phosprot_name
            elif not set(kinases)<=set(getattr(self.bg_pps, kin_type + '_' + data_att)):
                print('Not all kinase percentiles were provided. Re-calculating percentiles for background data')
                self.bg_pps.percentile(kin_type=kin_type,kinases=kinases,pp=pp)
                self.phosprot_name = _global_vars.phosprot_name
        
        fg_score_data = getattr(self.fg_pps, kin_type + '_' + data_att)
        bg_score_data = getattr(self.bg_pps, kin_type + '_' + data_att)
        
        enrichment_data = pd.DataFrame(index = kinases, columns = ['fg_counts', 'fg_total',
                                                                   'bg_counts', 'bg_total',
                                                                   'fg_percent', 'log2_freq_factor',
                                                                   'fisher_pval', 'adj_fisher_pval'])
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
        enrichment_data['adj_fisher_pval'] = multitest.multipletests(fisher_pvals, method = 'fdr_bh')[1]

        enrichemnt_results = EnrichmentResults(enrichment_results=enrichment_data, pps_data=self, kin_type=kin_type,
                                               kl_method=kl_method, kl_thresh=kl_thresh, enrichment_type=enrichment_type,
                                               tested_kins=kinases, adj_pval=adj_pval, data_att=data_att, kl_comp_direction=kl_comp_direction)
        
        return enrichemnt_results

#%%

class EnrichmentResults(object):
    """
    Class for kinase enrichment results.

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
    adj_pval : bool
        If True use adjusted p-value for calculation of statistical significance. Otherwise, use nominal p-value.
    data_att : str
        Score type ('scores', 'score_ranks', 'percentiles', 'percentile_ranks').
    kl_comp_direction : str
        Dictates if kinases above or below the specified threshold are used ('higher','lower').
    """
    
    def __init__(self, enrichment_results, pps_data,
                 kin_type, kl_method, kl_thresh, enrichment_type, tested_kins,
                 adj_pval, data_att, kl_comp_direction):
        
        self.enrichment_results = enrichment_results
        self.pps_data = pps_data
        self.kin_type = kin_type
        self.kl_method = kl_method
        self.kl_thresh = kl_thresh
        self.enrichment_type = enrichment_type
        self.tested_kins = tested_kins
        self.adj_pval = adj_pval
        self._data_att = data_att
        self._kl_comp_direction = kl_comp_direction
        
        if kl_method in ['percentile','percentile_rank']:
            self.phosprot_name = pps_data.phosprot_name
        
        
    def enriched_subs(self, kinases, data_columns=None,
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
            if not (os.path.isdir(output_dir+'/enriched_subs')):
                os.mkdir(output_dir+'/enriched_subs')
            
            if file_prefix is not None:
                writer = pd.ExcelWriter(f'{output_dir}/enriched_subs/{file_prefix}_{self.kin_type}_{self._data_att}_thresh_{self.kl_thresh}.xlsx')
            else:
                writer = pd.ExcelWriter(f'{output_dir}/enriched_subs/{self.kin_type}_{self._data_att}_thresh_{self.kl_thresh}.xlsx')
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
        
        return(enrich_subs_dict)
    
    
    def enriched_kins(self, sig_pval=0.1, sig_lff=0,
                      pval_col=None, lff_col='log2_freq_factor'):
        """
        Returns a list of all kinases enriched above a p-value and frequency-factor thresholds.

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
        enriched_kins : pd.DataFrame
            List of kinases enriched above the designated p-value and freq_facor thresholds.
        """
        
        if pval_col is None:
            if self.adj_pval:
                pval_col='adj_fisher_pval'
            else:
                pval_col='fisher_pval'
                
        enriched_kins = list(self.enrichment_results[(self.enrichment_results[pval_col] <= sig_pval) & (self.enrichment_results[lff_col] >= sig_lff)].index)
        return enriched_kins
    
    
    def depleted_kins(self, sig_pval=0.1, sig_lff=0,
                      pval_col=None, lff_col='log2_freq_factor'):
        """
        Returns a list of all kinases depleted above a p-value and frequency-factor thresholds.

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
        depleted_kins : pd.DataFrame
            List of kinases depleted above the designated p-value and freq_facor thresholds.
        """
        
        if pval_col is None:
            if self.adj_pval:
                pval_col='adj_fisher_pval'
            else:
                pval_col='fisher_pval'
                
        depleted_kins = list(self.enrichment_results[(self.enrichment_results[pval_col] <= sig_pval) & (self.enrichment_results[lff_col] <= sig_lff)].index)
        return depleted_kins
        
        
    def plot_volcano(self, sig_lff=0, sig_pval=0.1, fg_percent_thresh=0, fg_percent_col='fg_percent', kinases=None,
                     lff_col='log2_freq_factor', pval_col=None, highlight_kins=None, ignore_depleted=False,
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
        pval_col : str, optional
            P-value column name used for volcano plot. The defulat is None and will be determined based on self.adj_pval.
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
        
        if pval_col is None:
            if self.adj_pval:
                pval_col='adj_fisher_pval'
                if ylabel is None:
                    ylabel = '-log$_{10}$(Adjusted p-value)'
            else:
                pval_col='fisher_pval'
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

#%%

class MeaEnrichmentData(object):
    """
    Class for kinase enrichment data using GSEA method.

    Parameters
    ----------
    foreground : pd.DataFrame
        Dataframe with foreground substrates.
    background : pd.DataFrame
        Dataframe with background substrates.
    kin_type : str
        Kinase type ('ser_thr' or 'tyrosine').
    fg_seq_col : str, optional
        Substrates column name for the foreground data. The default is 'SITE_+/-7_AA'.
    bg_seq_col : str, optional
        Substrates column name for the background data. The default is 'SITE_+/-7_AA'.
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
        Sequence column name: 'SITE_+/-7_AA'. Phosphorylated residue column name: 'phos_res'.
    suppress_warnings : bool, optional
        Do not print warnings. The default is False.
    """

    def __init__(self, de_data, rank_col, kin_type,
                 seq_col='SITE_+/-7_AA',
                 subs_pad=False, pp=False,
                 drop_invalid_subs=True,
                 new_seq_phos_res_cols=True,
                 suppress_warnings=False):
        
        self.de_data = de_data
        self.rank_col = rank_col
        self.kin_type = kin_type
        self.de_data_pps = PhosphoProteomics(de_data, seq_col=seq_col, pad=subs_pad, pp=pp, drop_invalid_subs=drop_invalid_subs, new_seq_phos_res_cols=new_seq_phos_res_cols, suppress_warnings=suppress_warnings)
        
    
    def submit_scores(self, scores, suppress_messages=False):
        """
        Submitting scores for the foreground/background substrates.

        Parameters
        ----------
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
        
        self.de_data_pps.submit_scores(kin_type=self.kin_type, scores=scores, suppress_messages=suppress_messages)
        
        
    def submit_percentiles(self, percentiles, phosprot_name=None, suppress_messages=False):
        """
        Submitting percentiles for the foreground/background substrates.

        Parameters
        ----------
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
            
        self.de_data_pps.submit_percentiles(kin_type=self.kin_type, percentiles=percentiles, phosprot_name=phosprot_name, suppress_messages=suppress_messages)
        
    
    def _create_kin_sub_sets(self, thresh, comp_direction):
        
        print('\nGenerating kinase-substartes sets')
        kin_sub_sets = enrichment.create_kin_sub_sets(data_values=self.data_kl_values, threshold=thresh, comp_direction=comp_direction)
        self.kin_sub_sets = kin_sub_sets
        
        return(kin_sub_sets)
    
    
    def kinase_enrichment(self, kl_method, kl_thresh,
                          kinases=None, pp=None, non_canonical=False,
                          adj_pval=True, rescore=False, weight=1,
                          threads=4, min_size=1, max_size=100000,
                          permutation_num=1000, seed=112123,
                          verbose=False):
        """
        Kinase enrichment analysis based on pre-ranked GSEA substrates list.
    
        Parameters
        ----------
        kl_method : str
            Kinase Library scoring method ('score', 'score_rank', 'percentile', 'percentile_rank').
        kl_thresh : int
            The threshold to be used for the specified kl_method.
        kinases : list, optional
            If provided, kinase enrichment will only be calculated for the specified kinase list, otherwise, all kinases of the specified kin_type will be included. The default is None.
        pp : bool, optional
            Account for phospho-residues (s/t/y) in the sequence. The default is False.
        non_canonical : bool, optional
            Return also non-canonical kinases. For tyrosine kinases only. The default is False.
        enrichment_type : str, optional
            Direction of fisher's exact test for kinase enrichment ('enriched','depleted', or 'both').
        adj_pval : bool, optional
            If True use adjusted p-value for calculation of statistical significance. The default is True.
        rescore : bool, optional
            If True, Kinase Library scores or percentiles will be recalculated.
    
        Returns
        -------
        enrichemnt_results : pd.DataFrame
            pd.Dataframe with results of Kinase Enrichment for the specified KL method and threshold.
        """
        
        exceptions.check_kl_method(kl_method)

        kin_type = self.kin_type
        
        if kinases is None:
            kinases = data.get_kinase_list(kin_type, non_canonical=non_canonical)
        elif isinstance(kinases, str):
            kinases = [kinases]
        kinases = [x.upper() for x in kinases]
        exceptions.check_kin_list_type(kinases, kin_type=kin_type)
        
        data_att = kl_method+'s'
        kl_comp_direction = _global_vars.kl_method_comp_direction_dict[kl_method]

        if kl_method in ['score','score_rank']:
            if not hasattr(self.de_data_pps, kin_type + '_' + data_att) or rescore:
                self.de_data_pps.score(kin_type=kin_type,kinases=kinases,pp=pp)
            elif not set(kinases)<=set(getattr(self.de_data_pps, kin_type + '_' + data_att)):
                print('Not all kinase scores were provided. Re-calculating scores for data')
                self.de_data_pps.score(kin_type=kin_type,kinases=kinases,pp=pp)
        elif kl_method in ['percentile','percentile_rank']:
            if not hasattr(self.de_data_pps, kin_type + '_' + data_att) or rescore:
                print('\nCalculating percentiles for data')
                self.de_data_pps.percentile(kin_type=kin_type,kinases=kinases,pp=pp)
                self.phosprot_name = _global_vars.phosprot_name
            elif not set(kinases)<=set(getattr(self.de_data_pps, kin_type + '_' + data_att)):
                print('Not all kinase percentiles were provided. Re-calculating percentiles for data')
                self.de_data_pps.percentile(kin_type=kin_type,kinases=kinases,pp=pp)
                self.phosprot_name = _global_vars.phosprot_name
        
        self.data_kl_values = getattr(self.de_data_pps, kin_type + '_' + data_att)
        
        kin_sub_sets = self._create_kin_sub_sets(thresh=kl_thresh, comp_direction=kl_comp_direction)
        
        ranked_subs = self.de_data_pps.data.set_index('SITE_+/-7_AA')[self.rank_col].sort_values(ascending=False)
        
        prerank_results = gp.prerank(rnk=ranked_subs,
                             gene_sets=kin_sub_sets,
                             weight=weight,
                             threads=threads,
                             min_size=min_size,
                             max_size=max_size,
                             permutation_num=permutation_num,
                             seed=seed,
                             verbose=verbose)
        
        res_col_converter = {'Term': 'Kinase', 'ES': 'ES', 'NES': 'NES', 'NOM p-val': 'pvalue', 'FDR q-val': 'FDR', 'Tag %': 'Subs fraction', 'Lead_genes': 'Leading substrates'}

        enrichment_data = prerank_results.res2d.drop(['Name', 'FWER p-val', 'Gene %'], axis=1).rename(columns=res_col_converter)
        enrichment_data['pvalue'] = enrichment_data['pvalue'].replace(0,1/permutation_num).astype(float) #Setting p-value of zero to 1/(# of permutations)
        enrichment_data['FDR'] = enrichment_data['FDR'].replace(0,enrichment_data['FDR'][enrichment_data['FDR'] != 0].min()).astype(float) #Setting FDR of zero to lowest FDR in data
        sorted_enrichment_data = enrichment_data.sort_values('Kinase').set_index('Kinase').reindex(data.get_kinase_list(kin_type))
        
        enrichemnt_results = MeaEnrichmentResults(enrichment_results=sorted_enrichment_data, pps_data=self, gseapy_obj=prerank_results,
                                                   kin_type=kin_type, kl_method=kl_method, kl_thresh=kl_thresh, tested_kins=kinases,
                                                   data_att=data_att, kl_comp_direction=kl_comp_direction)
        
        return enrichemnt_results

#%%

class MeaEnrichmentResults(object):
    """
    Class for kinase enrichment results.

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
    adj_pval : bool
        If True use adjusted p-value for calculation of statistical significance. Otherwise, use nominal p-value.
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
        
        
    def enriched_subs(self, kinases, data_columns=None,
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
            data_columns = getattr(self.pps_data.de_data_pps, self.kin_type+'_data').columns.to_list()

        if save_to_excel:
            if output_dir is None:
                raise ValueError('Please provide output directory.')
            output_dir = output_dir.rstrip('/')
            if not (os.path.isdir(output_dir+'/enriched_subs')):
                os.mkdir(output_dir+'/enriched_subs')
            
            if file_prefix is not None:
                writer = pd.ExcelWriter(f'{output_dir}/enriched_subs/{file_prefix}_{self.kin_type}_{self._data_att}_thresh_{self.kl_thresh}.xlsx')
            else:
                writer = pd.ExcelWriter(f'{output_dir}/enriched_subs/{self.kin_type}_{self._data_att}_thresh_{self.kl_thresh}.xlsx')
        score_data = self.pps_data.de_data_pps.merge_data_scores(self.kin_type, self._data_att)

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
        
        return(enrich_subs_dict)
    
    
    def enriched_kins(self, sig_pval=0.1, sig_lff=0, adj_pval=True,
                      pval_col=None, lff_col='NES'):
        """
        Returns a list of all kinases enriched above a p-value and frequency-factor thresholds.

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
        enriched_kins : pd.DataFrame
            List of kinases enriched above the designated p-value and freq_facor thresholds.
        """
        
        if pval_col is None:
            if adj_pval:
                pval_col='adj_fisher_pval'
            else:
                pval_col='fisher_pval'
                
        enriched_kins = list(self.enrichment_results[(self.enrichment_results[pval_col] <= sig_pval) & (self.enrichment_results[lff_col] >= sig_lff)].index)
        return enriched_kins
        
        
    def plot_volcano(self, sig_lff=0, sig_pval=0.1, adj_pval=True, kinases=None,
                     lff_col='NES', pval_col=None, highlight_kins=None,
                     kins_label_dict=None, label_kins=None, adjust_labels=True, labels_fontsize=7,
                     symmetric_xaxis=True, grid=True, max_window=False,
                     title=None, stats=True, xlabel='NES', ylabel=None,
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
                pval_col='pvalue'
        
        if ylabel is None:
            ylabel='-log$_{10}$('+pval_col+')'
        
        if kinases is None:
            kinases = self.tested_kins

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
                                           lff_col=lff_col, pval_col=pval_col, highlight_kins=highlight_kins,
                                           label_kins=label_kins, adjust_labels=adjust_labels, labels_fontsize=labels_fontsize,
                                           symmetric_xaxis=symmetric_xaxis, grid=grid, max_window=max_window,
                                           title=title, xlabel=xlabel, ylabel=ylabel,
                                           plot=plot, save_fig=save_fig, return_fig=return_fig,
                                           ax=ax, **plot_kwargs)
        
        
        