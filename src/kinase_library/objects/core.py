"""
#####################################
# The Kinase Library - Core Objects #
#####################################
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.patches as patches
import seaborn as sns
import pyarrow.parquet as pq
from tqdm import tqdm

from ..utils import _global_vars, exceptions, utils
from ..modules import data

#%%

class Substrate(object):
    """
    Kinase Library object for a substrate.

    Parameters
    ----------
    seq : str
        Sequence with phosphoacceptor.
        'X' represents random amino acid.
        '_' represents truncation of the sequence.
    kin_type : str, optional
        Substrate type (if specified: 'ser_thr' or 'tyrosine'). The default is None.
    pp : bool, optional
        Phospho-priming (phospho-residues in the sequence). The default is False.
    aa : list, optional
        List of amino acids to use in the matrix columns. The default is None.
    pos : list, optional
        List of positions to use in the matrix rows. The default is None.
    phos_pos : int, optional
        Position of phosphoacceptor. The default is None.
    validate_phos_res : bool, optional
        validating phosphoacceptor. The default is True.
    validate_aa : bool, optional
        Validating amino acids. The default is True.

    Examples
    -------
    >>> seq = 'PFADRRVtPXYsPKH'
    >>> sub = kl.Substrate(seq, pp=True)
    >>> sub.substrate
        'PFADRRVtPXYsPKH'
    >>> sub.kin_type
        ser_thr'
    >>> sub.sub_bin_matrix
           -5  -4  -3  -2  -1   1   2   3   4
        P   0   0   0   0   0   1   0   0   0
        G   0   0   0   0   0   0   0   0   0
        A   1   0   0   0   0   0   0   0   0
        C   0   0   0   0   0   0   0   0   0
        S   0   0   0   0   0   0   0   0   0
        T   0   0   0   0   0   0   0   0   0
        V   0   0   0   0   1   0   0   0   0
        I   0   0   0   0   0   0   0   0   0
        L   0   0   0   0   0   0   0   0   0
        M   0   0   0   0   0   0   0   0   0
        F   0   0   0   0   0   0   0   0   0
        Y   0   0   0   0   0   0   0   1   0
        W   0   0   0   0   0   0   0   0   0
        H   0   0   0   0   0   0   0   0   0
        K   0   0   0   0   0   0   0   0   0
        R   0   0   1   1   0   0   0   0   0
        Q   0   0   0   0   0   0   0   0   0
        N   0   0   0   0   0   0   0   0   0
        D   0   1   0   0   0   0   0   0   0
        E   0   0   0   0   0   0   0   0   0
        s   0   0   0   0   0   0   0   0   1
        t   0   0   0   0   0   0   0   0   0
        y   0   0   0   0   0   0   0   0   0

    """
    
    def __init__(self, seq, kin_type=None, pp=False, phos_pos=None, aa=None, pos=None, validate_phos_res=True, validate_aa=True):
        
        sub = utils.sequence_to_substrate(seq=seq, pp=pp, phos_pos=phos_pos, kin_type=kin_type, validate_phos_res=validate_phos_res, validate_aa=validate_aa)

        if kin_type is None:
            kin_type = utils.substrate_type(sub)
        if aa is None:
            aa = data.get_aa()
        if pos is None:
            pos = data.get_positions(kin_type)
            
        self.sequence = seq
        self.length = len(seq)
        self.substrate = sub
        self.phos_res = sub[7].lower()
        self.pp = pp
        self.kin_type = kin_type
        self.sub_bin_matrix = utils.sub_binary_matrix(sub, pp=pp, aa=aa, pos=pos)
        self._aa = aa
        self._pos = pos
    
    
    def score(self, kinases=None, pp=None,
              st_fav=True, log2_score=True,
              non_canonical=False,
              output_type = 'series', sort_by = 'value',
              round_digits=4):
        """
        Calculate score of the substrate for the given kinases.
        
        Score is being computed in a vectorized way:
            1. Making binary matrix for the substrates.
            2. Converting kinase matrix (norm-scaled) to log2
            3. Performing dot-product (summing the corresponding log2 of the kinase matrix)

        Parameters
        ----------
        kinases : str or list, optional
            List of kinase names to score by. If none, will score by all the kinases. The Default is None.
        pp : bool, optional
            Phosphopriming. The default is None.
            If not specified, will be inferred from the substrate.
        st_fav : bool, optional
            S/T favorability. The default is True.
        log2_score : bool, optional
            Return scores as log2. The default is True.
        non_canonical : bool, optional
            Return also non-canonical kinases. For tyrosine kinases only. The default is False.
        output_type : str, optional
            Type of returned data. The default is 'series'.
            'series': pd.Series, kinases as index.
            'list': list of values (same order as input kinase list).
            'dict': dictionary (kinase -> value).
        sort_by : str, optional
            Sorting method for output. The default is 'value'.
            'input': keep the order of input kinases.
            'name': sort by kinase name.
            'value': sort by scores (high to low).
        round_digits : int, optional
            Number of decimal digits. The default is 4.

        Returns
        -------
        score_output : pd.Series, list, or dictionary
            Scores of the substrate for the specified kinases.
        """
        
        current_dir = os.path.dirname(__file__)
        mat_dir = os.path.join(current_dir, _global_vars.mat_dir)
        
        kin_type = self.kin_type
        
        if kinases is None:
            kinases = data.get_kinase_list(kin_type, non_canonical=non_canonical)
        elif isinstance(kinases, str):
            kinases = [kinases]
        
        exceptions.check_kin_list_type(kinases, kin_type=kin_type)
        exceptions.check_score_output_type(output_type)
        exceptions.check_output_sort_by(sort_by)
        
        kinases = [x.upper() for x in kinases]
        
        if pp is None:
            pp = self.pp
            sub_mat = self.sub_bin_matrix
        elif pp:
            sub_mat = self.sub_bin_matrix
        elif not pp:
            sub_mat = utils.sub_binary_matrix(self.substrate, pp=False, aa=self._aa, pos=self._pos)
        
        # Using table with all the matrices concatenated (log2)
        kin_mat_log2 = data.get_multiple_matrices(kinases, kin_type=kin_type, mat_type='log2')
        
        sub_vector = utils.flatten_matrix(sub_mat)

        # matrices are in log2 space
        score_log2 = pd.Series(np.dot(sub_vector,kin_mat_log2.transpose())[0], index = kinases)
        if (kin_type == 'ser_thr') and st_fav:
            st_fav_scores = data.get_st_fav(kinases)[self.phos_res.upper()]
            st_fav_scores_log2 = np.log2(st_fav_scores)
            score_log2 = score_log2 + st_fav_scores_log2
        score = np.power(2,score_log2)
        
        if log2_score:
            score_output = score_log2.round(round_digits)
        else:
            score_output = score.round(round_digits)
        
        if sort_by == 'name':
            score_output = score_output.sort_index()
        elif sort_by == 'value':
            score_output = score_output.sort_values(ascending=False)
        elif sort_by == 'input':
            score_output = score_output.loc[kinases]
        
        if output_type == 'list':
            score_output = list(score_output)
        elif output_type == 'dict':
            score_output = score_output.to_dict()
        
        return(score_output)
    
    
    def percentile(self, kinases=None,
                   customized_scored_phosprot=None,
                   pp=None, st_fav=True, non_canonical=False,
                   output_type='series', sort_by = 'value',
                   phosprot_path='./../databases/substrates',
                   round_digits=2):
        """
        Calculate the percentile score of the substrate for the given kinases.
        
        After score is being computed, the percentile of that score is being
        computed based on a basal scored phosphoproteome.

        Parameters
        ----------
        kinases : str or list, optional
            List of kinase names to score by. If none, will score by all the kinases. The Default is None.
        customized_scored_phosprot : kl.ScoredPhosphoProteome, optional
            Customized phosphoproteome object. The default is None.
        pp : bool, optional
            Phosphopriming. The default is None.
            If not specified, will be inferred from the substrate.
        st_fav : bool, optional
            S/T favorability. The default is True.
        non_canonical : bool, optional
            Return also non-canonical kinases. For tyrosine kinases only. The default is False.
        output_type : str, optional
            Type of returned data. The default is 'series'.
            'series': pd.Series, kinases as index.
            'list': list of values (same order as input kinase list).
            'dict': dictionary (kinase -> value).
        sort_by : str, optional
            Sorting method for output. The default is 'value'.
            'input': keep the order of input kinases.
            'name': sort by kinase name.
            'value': sort by percentile (high to low).
        phosprot_path : atr, optional
            Path to scored phosphoproteome files. The default is './../databases/substrates/scored_phosprots'.
        round_digits : int, optional
            Number of decimal digits. The default is 2.

        Returns
        -------
        percent_output : pd.Series, list, or dictionary
            Percentiles of the substrate for the specified kinases.
        """
        
        kin_type = self.kin_type
        
        if kinases is None:
            kinases = data.get_kinase_list(kin_type, non_canonical=non_canonical)
        elif isinstance(kinases, str):
            kinases = [kinases]
        
        exceptions.check_kin_list_type(kinases, kin_type=kin_type)
        exceptions.check_score_output_type(output_type)
        exceptions.check_output_sort_by(sort_by)
        
        kinases = [x.upper() for x in kinases]
        
        if pp is None:
            pp = self.pp
        
        if customized_scored_phosprot is None:
            all_scored_phosprot = _global_vars.all_scored_phosprot

        if kin_type == 'ser_thr':
            scored_phosprot = all_scored_phosprot.ser_thr_scores
        else:
            scored_phosprot = all_scored_phosprot.tyrosine_scores
            
        percent_output = []
        
        score = self.score(kinases=kinases, pp=pp, st_fav=st_fav)
        
        percent_output = round((scored_phosprot[score.index] <= score).sum() / len(scored_phosprot) * 100, round_digits)
        
        if sort_by == 'name':
            percent_output = percent_output.sort_index()
        elif sort_by == 'value':
            percent_output = percent_output.sort_values(ascending=False)
        elif sort_by == 'input':
            percent_output = percent_output.loc[kinases]
            
        if output_type == 'list':
            percent_output = list(percent_output)
        elif output_type == 'dict':
            percent_output = percent_output.to_dict()
        
        return(percent_output)
    
    
    def rank(self, kinases=None, method='percentile',
             pp=None, st_fav=True, non_canonical=False,
             output_type='series', sort_by='value',
             customized_scored_phosprot=None,
             phosprot_path='./../databases/substrates',
             score_round_digits=4, percentile_round_digits=2):
        """
        Returns a dataframe containing a ranking based on the Kinase Library score or percentile for all specified kinases.

        Parameters
        ----------
        kinases : list of str
            List of kinases for which to calculate rankings. The default is None, meaning all kinases of self.kin_type are used.
        method : str, optional
            Scoring method, must be 'score' or 'percentile'. The default is 'percentile'.
        pp : bool, optional
            Phosphopriming boolean to be sent into score and percentile. The default is None.
            If not specified, will be inferred from the substrate.
        st_fav : bool, optional
            S/T favorability. The default is True.
        non_canonical : bool, optional
            Return also non-canonical kinases. For tyrosine kinases only. The default is False.
        output_type : str, optional
            Format for function output; 'list', 'dict', or 'series' (default). 
        sort_by : str, optional
            Sorting method for output. The default is 'value'.
            'input': keep the order of input kinases.
            'name': sort by kinase name.
            'value': sort by rank (low to high).
        customized_scored_phosprot : kl.ScoredPhosphoProteome, optional
            Customized phosphoproteome object. The default is None.
        phosprot_path : atr, optional
            Path to scored phosphoproteome files. The default is './../databases/substrates/scored_phosprots'.
        score_round_digits : int, optional
            Number of decimal digits for score. The default is 4.
        percentile_round_digits : int, optional
            Number of decimal digits for percentile. The default is 2.

        Returns
        -------
        rank_output : pd.DataFrame
            Dataframe containing kinases as the index and Kinase Library score / percentile ranks as the columns.
        """

        kin_type = self.kin_type
        
        if kinases is None:
            kinases = data.get_kinase_list(kin_type, non_canonical=non_canonical)
        elif isinstance(kinases, str):
            kinases = [kinases]
        
        if method not in ['score','percentile']:
            raise ValueError('\'mothod\' must be either \'score\' or \'percentile\'.')
        exceptions.check_kin_list_type(kinases, kin_type=kin_type)
        exceptions.check_score_output_type(output_type)
        exceptions.check_output_sort_by(sort_by)
        
        kinases = [x.upper() for x in kinases]
        
        if method == 'score':
            score_values = self.score(pp=pp, st_fav=st_fav, non_canonical=non_canonical, round_digits=score_round_digits)
        else:
            score_values = self.percentile(customized_scored_phosprot=customized_scored_phosprot, pp=pp, st_fav=st_fav, non_canonical=non_canonical, phosprot_path=phosprot_path, round_digits=percentile_round_digits)

        rank_output = score_values.rank(method='min', ascending=False).astype(int).loc[kinases]
        
        
        if sort_by == 'name':
            rank_output = rank_output.sort_index()
        elif sort_by == 'value':
            rank_output = rank_output.sort_values(ascending=True)
        elif sort_by == 'input':
            rank_output = rank_output.loc[kinases]
        
        if output_type == 'list':
            rank_output = list(rank_output)
        elif output_type == 'dict':
            rank_output = rank_output.to_dict()
            
        return(rank_output)
    
    
    def predict(self, kinases=None,
                pp=None, st_fav=True, non_canonical=False,
                customized_scored_phosprot=None,
                phosprot_path='./../databases/substrates',
                log2_score=True,
                sort_by='percentile',
                score_round_digits=4, percentile_round_digits=2):
        """
        Generates a dataframe of scores, percentiles, and ranks (optional) for the given list of kinases. 

        Parameters
        ----------
        kinases : list of str, optional
            List of kinases for which to make predictions. The default is None, meaning all kinases of self.kin_type are used.
        pp : bool, optional
            Phosphopriming boolean to be sent into score and percentile. The default is None.
            If not specified, will be inferred from the substrate.
        st_fav : bool, optional
            S/T favorability. The default is True.
        non_canonical : bool, optional
            Return also non-canonical kinases. For tyrosine kinases only. The default is False.
        customized_scored_phosprot : kl.ScoredPhosphoProteome, optional
            Customized phosphoproteome object. The default is None.
        phosprot_path : atr, optional
            Path to scored phosphoproteome files. The default is './../databases/substrates/scored_phosprots'.
        log2_score : bool, optional
            Return scores as log2. The default is True.
        sort_by : str, optional
            Sorting method for output. The default is 'percentile'.
            'input': keep the order of input kinases.
            'name': sort by kinase name.
            'score': sort by score (high to low).
            'precentile': sort by precentile (high to low).
        score_round_digits : int, optional
            Number of decimal digits for score. The default is 4.
        percentile_round_digits : int, optional
            Number of decimal digits for percentile. The default is 2.

        Returns
        -------
        prediction_output : pd.DataFrame
            Dataframe containing kinases as the index and scoring metrics as the columns.
        """
        
        kin_type = self.kin_type
        
        if kinases is None:
            kinases = data.get_kinase_list(kin_type, non_canonical=non_canonical)
        elif isinstance(kinases, str):
            kinases = [kinases]
        
        exceptions.check_kin_list_type(kinases, kin_type=kin_type)
        exceptions.check_output_sort_by(sort_by)
        
        kinases = [x.upper() for x in kinases]

        score = self.score(kinases=kinases, log2_score=log2_score, pp=pp, st_fav=st_fav, non_canonical=non_canonical, round_digits=score_round_digits)
        score_rank = self.rank(kinases=kinases, method='score', pp=pp, st_fav=st_fav, non_canonical=non_canonical, score_round_digits=score_round_digits)
        percentile = self.percentile(kinases, customized_scored_phosprot=customized_scored_phosprot, pp=pp, st_fav=st_fav, non_canonical=non_canonical, round_digits=percentile_round_digits)
        percentile_rank = self.rank(kinases=kinases, method='percentile', pp=pp, st_fav=st_fav, non_canonical=non_canonical, customized_scored_phosprot=customized_scored_phosprot, phosprot_path=phosprot_path, percentile_round_digits=percentile_round_digits)

        prediction_output = pd.concat([score.rename('Score'),
                                       score_rank.rename('Score Rank'),
                                       percentile.rename('Percentile'),
                                       percentile_rank.rename('Percentile Rank')], axis=1)
        
        if sort_by == 'name':
            prediction_output = prediction_output.sort_index()
        elif sort_by == 'score':
            prediction_output = prediction_output.sort_values(by='Score', ascending=False)
        elif sort_by == 'percentile':
            prediction_output = prediction_output.sort_values(by='Percentile', ascending=False)
        elif sort_by == 'input':
            prediction_output = prediction_output.loc[kinases]
        
        return(prediction_output)
    
    
    def un_primed(self):
        """
        Returning un-primed substrate.

        Returns
        -------
        un_primed_sub : str
            15-mer with no phosphorylated residues.
        """
        
        un_primed_sub = utils.unprime_substrate(self.substrate)
        return(un_primed_sub)


#%%

class Kinase(object):
    """
    Kinase Library object for a kinase.

    Parameters
    ----------
    name : str
        Kinase name.
    matrix : np.ndarray or pd.DataFrame
        Kinase matrix.
    random_aa_value : float
        Value of random amino acid in the matrix.
    mat_type : str, optional
        Matrix type ('densitometry', 'raw', 'norm', 'log2', or customized). The default is 'log2'.
    kin_type : str
        Kinase type.
    family : str, optional
        Kinase family. The default is None.
    pp : bool, optional
        Phospho-residues in the matrix (phospho-priming). The default is True.
    k_mod : bool, optional
        Modified lysine (acetylation and tri-methylation). The default is False.
    phos_acc_fav : dict, optional
        Central phosphoacceptor favorability. The default is None.
    cols : list, optional
        Matrix columns. Must fit the shape of the matrix. The default is None.
    rows : list, optional
        Matrix rows. Must fit the shape of the matrix. The default is None.

    Examples
    -------
    >>> matrix = kl.get_matrix('SRPK1')
    >>> kinase = kl.Kinase('SRPK1', matrix)
    >>> kinase.name
        'SRPK1'
    >>> kinase.matrix
               -5      -4      -3      -2      -1       1       2       3       4
        P  0.0594  0.0812  0.0353  0.0604  0.1116  0.2793  0.0786  0.0996  0.0798
        G  0.0753  0.0672  0.0373  0.0790  0.0564  0.0554  0.0609  0.0536  0.0661
        A  0.0889  0.0713  0.0477  0.0970  0.0661  0.0500  0.0724  0.0401  0.0611
        C  0.0814  0.0588  0.0384  0.0673  0.0462  0.0504  0.0666  0.0393  0.0635
        S  0.0525  0.0484  0.0255  0.0577  0.0436  0.0364  0.0609  0.0294  0.0581
        T  0.0525  0.0484  0.0255  0.0577  0.0436  0.0364  0.0609  0.0294  0.0581
        V  0.0517  0.0388  0.0202  0.0317  0.0411  0.0424  0.0421  0.0269  0.0414
        I  0.0468  0.0433  0.0219  0.0300  0.0326  0.0357  0.0346  0.0270  0.0484
        L  0.0464  0.0426  0.0192  0.0253  0.0436  0.0411  0.0367  0.0270  0.0460
        M  0.0525  0.0484  0.0178  0.0294  0.0425  0.0364  0.0421  0.0294  0.0455
        F  0.0588  0.0420  0.0175  0.0375  0.0280  0.0228  0.0368  0.0250  0.0436
        Y  0.0488  0.0454  0.0189  0.0349  0.0336  0.0222  0.0357  0.0258  0.0438
        W  0.0611  0.0432  0.0183  0.0363  0.0247  0.0245  0.0342  0.0266  0.0521
        H  0.0561  0.0562  0.0267  0.1012  0.0709  0.0331  0.0947  0.0410  0.0581
        K  0.0927  0.0945  0.0599  0.0787  0.1291  0.0910  0.0673  0.0843  0.0866
        R  0.0851  0.1347  0.5477  0.0944  0.1279  0.1368  0.0746  0.3521  0.0822
        Q  0.0397  0.0512  0.0350  0.0407  0.0685  0.0548  0.0577  0.0411  0.0645
        N  0.0510  0.0566  0.0255  0.0701  0.0602  0.0360  0.0677  0.0468  0.0620
        D  0.0453  0.0401  0.0298  0.0957  0.0286  0.0161  0.0859  0.0293  0.0667
        E  0.0403  0.0431  0.0214  0.0577  0.0347  0.0222  0.0781  0.0242  0.0521
        s  0.0450  0.0454  0.0300  0.0401  0.0322  0.0287  0.0740  0.0344  0.0701
        t  0.0450  0.0454  0.0300  0.0401  0.0322  0.0287  0.0740  0.0344  0.0701
        y  0.0557  0.0838  0.0408  0.0820  0.0443  0.0272  0.0559  0.0384  0.0622
    >>> kinase.kin_type
        'ser_thr'
    >>> kinase.family
        'CMGC'
    >>> kinase.info
        MATRIX_NAME                                                           SRPK1
        GENENAME                                                              SRPK1
        TYPE                                                                ser_thr
        SUBTYPE                                                                 STK
        FAM                                                                    CMGC
        UNIPROT_ID                                                           Q96SB4
        UNIPROT_ENTRY_NAME                                              SRPK1_HUMAN
        EMBL_ID                                                              U09564
        EMBL                                                             AAA20530.1
        ENSEMBL_GENE_ID                                             ENSG00000096063
        ENSEMBL_TRS_ID            ENST00000373825;ENST00000361690;ENST0000034616...
        ENSEMBL_PRO_ID                                              ENSP00000362931
        P_ENTREZGENEID                                                       6732.0
        P_GI                                                               82407376
        PIR                                                                  S45337
        REFSEQ_NT_ID                                                    NM_003137.4
        P_REFSEQ_AC                                                     NP_003128.3
        PDB_ID                                                                 1WAK
        BIOGRID_ID                                                         112610.0
        DIP_ID                                                           DIP-33888N
        STRING_ID                                              9606.ENSP00000362931
        CHEMBL_ID                                                        CHEMBL4375
        DRUGBANK_ID                                                             NaN
        GUIDETOPHARMACOLOGY_ID                                               2208.0
        BIOMUTA_ID                                                            SRPK1
        DMDM_ID                                                         209572680.0
        CPTAC_ID                                                                NaN
        PROTEOMICSDB_ID                                                       78098
        DNASU_ID                                                             6732.0
        REACTOME_ID                                                             NaN
        CHITARS_ID                                                            SRPK1
        GENEWIKI_ID                                                           SRPK1
        GENOMERNAI_ID                                                        6732.0
        HGNC_ID                                                          HGNC:11305
        Name: SRPK1, dtype: object
        """
    
    def __init__(self, name, matrix, random_aa_value, mat_type='log2',
                 kin_type=None, family=None, pp=True, k_mod=False,
                 phos_acc_fav=None, cols=None, rows=None):
        
        input_type = type(matrix)
        if input_type == np.ndarray:
            if rows is None or cols is None:
                raise Exception('If matrix is provided as np.ndarray, cols and rows must be specified.')
            df_mat = utils.matrix_to_df(mat=matrix, kin_type=kin_type, pp=pp, k_mod=k_mod, mat_type=mat_type, cols=cols, rows=rows)
        else:
            df_mat = matrix.copy()
        
        # if df_mat.min().min() <= 0:
        #     raise Exception('Matrix cannot contain zero or negative values.')
        
        try:
            info = data.get_kinase_info(name, kin_type=kin_type)
        except:
            info = 'N/A'
        
        if mat_type == 'log2':
            lin_mat = np.power(2,df_mat)*random_aa_value
            log2_mat = df_mat.copy()
        elif mat_type in ['densitometry','raw','norm']:
            lin_mat = df_mat.copy()
            log2_mat = np.log2(df_mat/random_aa_value)
        elif mat_type == 'customized':
            lin_mat = df_mat.copy()
            log2_mat = df_mat.copy()
        else:
            raise ValueError('\'mat_type\' must be one of the followings: \'densitometry\', \'raw\', \'norm\', \'log2\', or \'customized\'.')
        
        self.name = name
        self.matrix = df_mat
        self.random_aa_value = random_aa_value
        self.norm_matrix = lin_mat
        self.log2_matrix = log2_mat
        self.mat_type = mat_type
        self.kin_type = kin_type
        self.family = family
        self.amino_acids = list(df_mat.index)
        self.positions = list(df_mat.columns)
        self.pp = pp
        self.k_mod = k_mod
        self.phos_acc_fav = phos_acc_fav
        self.info = info
        
        if kin_type == 'ser_thr':
            self.st_fav = self.phos_acc_fav
        
    
    def _get_matrix(self, kinase, kin_type=None, mat_type='log2', aa=None, pos=None, transpose=False):
        """
        Read kinase matrix from file as pd.DataFrame.
    
        Parameters
        ----------
        kinase : str
            Kinase name.
        kin_type : str
            Kinase type (ser_thr, tyrosine).
        mat_type : str, optional
            Matrix type (raw, normalized, or normalized-scaled).
        aa : list, optional
            List of amino acids to select. The default is None.
        pos : list, optional
            Positions to select. The default is None.
        transpose : bool, optional
            If True, return transposed matrix. The default is False.
    
        Returns
        -------
        kin_matrix : pd.DataFrame
            Kinase matrix as dataframe.
        """
        
        current_dir = os.path.dirname(__file__)
        mat_dir = os.path.join(current_dir, _global_vars.mat_dir)
        
        exceptions.check_kin_name(kinase)
        if kin_type is None:
            kin_type = data.get_kinase_type(kinase)
        else:
            exceptions.check_kin_type(kin_type)
        exceptions.check_mat_type(mat_type)
        
        kinase = kinase.upper()
        
        if pos is None:
            pos = data.get_positions(kin_type)
        if aa is None:
            aa = data.get_aa()
        
        kin_list = [name.split('.')[0] for name in os.listdir(mat_dir + '/' + kin_type + '/' + mat_type + '/') if (name.startswith(".") == False)]
        kin_list.sort()
    
        exceptions.check_kin_name(kinase, kin_list)
        
        full_matrix = pd.read_csv(mat_dir + '/' + kin_type + '/' + mat_type + '/' + kinase + '.tsv', sep = '\t', index_col = 0)
        kin_matrix = full_matrix.loc[pos, aa]
        
        # Matrices are saved with amino acids as columns and positions as rows
        # However default presentation is amino acids as rows and positions as columns
        if not transpose:
            kin_matrix = kin_matrix.transpose()
        
        return(kin_matrix)
    
    
    def _plot_st_fav(self, title='S/T Favorability', ax=None, labels_loc = 'bottom', value_annot=True,
                     bar_color='red', x_text=None, y_text=None, ha_text='center', va_text='top',
                     xticks_fontsize=8, annot_fontsize=10, title_fontsize=10):
        """
        Private function to plot the kinase's s/t favorability.
    
        Parameters
        ----------
        title : str, optional
            Title for the plot. If no string is provided, the default is 'S/T Favorability'.
        ax : matplotlib.axes, optional
            Axis on which to plot s/t favorability. If None, a new subplot will be generated.
        labels_loc : str, optional
            Where to draw and label ticks. The default is 'bottom'.
        value_annot : bool, optional
            If true, s/t favorability will be annotated on the plot.
        bar_color : str, optional
            Color for the s/t favorability line. The default is 'red'.
        x_text : str, optional
            X-coordinate of the favorability value annotation, if not provided the value will be inferred by the s/t ratio.
        y_text : str, optional
            Y-coordinate of the favorability value annotation, the default is None.
        ha_text : str, optional
            Horizontal alignment of favorability text. The default is 'center'.
        va_text : str, optional
            Vertical alignment of the annotation text. The default is 'top'.
        xticks_fontsize : int, optional
            Font size for the x-axis tick labels. The default is 8.
        annot_fontsize : int, optional
            Font size for the favorability text annotation. The default is 10.
        title_fontsize : int, optional
            Font size for the plot title. The default is 10.
    
        Returns
        -------
        None
        """

        if labels_loc not in ['top','bottom']:
            raise ValueError('\'labels_loc\' must be either \'top\' or \'bottom\'.')
        if ax is None:
            w,h = plt.figaspect(1/5)
            fig,ax = plt.subplots(figsize=(w,h))
        
        st_fav_ratio = np.round(self.st_fav['T']/(self.st_fav['S'] + self.st_fav['T']),2)
        st_fav_percent = int(st_fav_ratio*100)
        ax.axvline(st_fav_ratio, c=bar_color, lw=3)
        ax.set_xticks(ticks=[0,0.5,1])
        ax.set_xticklabels(labels=['Serine\nSpecific','Dual\nSpecific','Threonine\nSpecific'], fontsize=xticks_fontsize, weight = 'bold')
        ax.tick_params(top=(labels_loc=='top'), labeltop=(labels_loc=='top'),
                       bottom=(labels_loc=='bottom'), labelbottom=(labels_loc=='bottom'),
                       left=False, labelleft=False)
        
        if x_text is None:
            if st_fav_percent>=50:
                x_text = 0.01
                ha_text = 'left'
            else:
                x_text = 0.99
                ha_text = 'right'
        if y_text is None:
            y_text = 0.95
        if value_annot:
            ax.text(x_text, y_text, f'S:T = {st_fav_percent}%:{100-st_fav_percent}%',
                    fontsize=annot_fontsize, color='red', weight='bold', ha=ha_text, va=va_text)
        ax.set_title(title, fontsize=title_fontsize, weight = 'bold')
        
        try:
            fig.tight_layout()
        except:
            pass
    
    
    def get_value(self, pos, aa):
        """
        Returns matrix value at certain amino acid and position.

        Parameters
        ----------
        pos : int
            position.
        aa : character
            Amino acid.

        Returns
        -------
        Matrix value.
        """

        if pos not in self.positions:
            raise Exception('Invalid position.')
        if aa not in self.amino_acids:
            raise Exception('Invalid amino acid.')
        
        return(self.matrix.loc[aa,pos])
    
    
    def heatmap(self, zero_pos=True, square=True, title=None,
                xticks_fontsize=None, yticks_fontsize=None, cmap=None, mat_scale='log',
                drop_aa=[], drop_pos=[], replace_aa_labels={'s' :'pS', 't' :'pT', 'y':'pY'},
                plot=True, return_fig=False, ax=None, cbar=True, cbar_ax=None, label=True):
        """
        Make heatmap of the matrix.

        Parameters
        ----------
        title : str, optional
            Title of heatmap. The default is None.
        xticks_fontsize : float, optional
            x-ticks labels size. The Default is 8 if 'square' is True, and 14 is 'square' is False.
        yticks_fontsize : float, optional
            y-ticks labels size. The Default is 8.
        cmap : plt.colormap, optional
            Colormap for the heatmap. The default is None (will be ['darkblue','blue','white','IndianRed','red']).
        mat_scale : string, optional
            Scale of matrix - linear-scale ('linear') or log-scale ('log').
        drop_aa : list, optional
            Amino acids to drop. The default is [].
        drop_pos : list, optional
            Position to drop. The default is [].
        replace_aa_labels : dict, optional
            Renameing amino acids. The default is {'s' :'pS', 't' :'pT', 'y':'pY'}.
        plot : bool, optional
            Ploting the heatmap. The default is True.
        return_fig : bool, optional
            Returning axis with the figure. The default is False.

        Returns
        -------
        fig : plt.Figure
            if 'return_fig' is True, returning plt.Figure of the heatmap.
        """
        
        exceptions.check_mat_scale(mat_scale)
        
        random_aa_value = self.random_aa_value
        
        if title is None:
            title = self.name
        elif title is False:
            title = None
            
        if ax is None:
            fig,ax = plt.subplots()
        else:
            plot = False
            if return_fig:
                raise ValueError('When Axes provided, \'return_fig\' must be False.')
        
        if label is True:
            label = f'Favorability ({mat_scale}-scale)'
        else:
            label = None
        
        if cmap is None:
            cmap = mcol.LinearSegmentedColormap.from_list("Kinase",['darkblue','blue','white','IndianRed','red'])
            
        if mat_scale == 'log':
            plot_matrix = self.log2_matrix
            vmin = None
            vcenter = 0
        elif mat_scale == 'linear':
            plot_matrix = self.norm_matrix
            vmin = 0
            vcenter = random_aa_value
        cnorm = mcol.TwoSlopeNorm(vmin=vmin, vcenter=vcenter)
        
        plot_matrix = plot_matrix.drop(drop_aa, errors='ignore').drop(drop_pos, axis=1, errors='ignore').rename(index = replace_aa_labels)
        
        if zero_pos:
            plot_matrix.insert(int((plot_matrix.columns < 0).sum()), 0, random_aa_value)
        sns.heatmap(plot_matrix, cmap=cmap, norm=cnorm, vmin=vmin, xticklabels=True, yticklabels=True, square=square, linewidths=0.25+0.25*(square==False), linecolor='gray', ax=ax, cbar=cbar, cbar_ax=cbar_ax)
        ax.collections[-1].colorbar.ax.set_yscale('linear')
        ax.set_ylabel(label, fontsize=8)
        if zero_pos:
            ax.add_patch(
                patches.Rectangle(
                    xy=(int((plot_matrix.columns < 0).sum()), 0),
                    width=1.0, height=len(plot_matrix),
                    edgecolor='gray',
                    fill=True, facecolor='white',
                    lw=0.25+0.25*(square==False)))
        else:
            ax.axvline(int((plot_matrix.columns < 0).sum()), color='black', lw=2+1*(square==False))
        
        ax.add_patch(patches.Rectangle((0, 0), plot_matrix.shape[1], plot_matrix.shape[0], fill=False, edgecolor='black', lw=1))
        
        if xticks_fontsize is None:
            if square:
                xticks_fontsize = 9
            else:
                xticks_fontsize = 14
        if yticks_fontsize is None:
            if square:
                yticks_fontsize = 9
            else:
                yticks_fontsize = 12
        
        ax.set_xticklabels(labels=[str(x) if x<=0 else '+'+str(x) for x in sorted(plot_matrix.columns)], fontsize=xticks_fontsize)
        ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=yticks_fontsize)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, left=True, labelleft=True, rotation=0)
        ax.set_title(title, fontsize=14)
        
        try:
            fig.tight_layout()
        except:
            pass
        
        if not plot:
            try:
                plt.close(fig)
            except:
                pass
        
        if return_fig:
            return(fig)
    
    
    def seq_logo(self, logo_type='ratio_to_median', zero_pos=None, zero_pos_format='upper',
                 drop_aa=['s'], drop_pos=[], replace_aa_labels={'t' :'pS/pT', 'y':'pY'},
                 title=None, xlabel='Position', ylabel=None,
                 save_fig=False, return_fig=False, plot=True, ax=None,
                 **seq_logo_kwargs):
        """
        Generates a sequence logo based on the kinase's matrix.

        Parameters
        ----------
        logo_type : str, optional
            Type of sequence logo - 'ratio_to_random', 'ratio_to_median', or 'prob'. The default is 'ratio_to_median'.
            ratio_to_random: height is ratio to random amino acid value. 'ratio_to_random' must be specified.
            ratio_to_median: height is ratio to position-median.
            prob: probability, all values sum up to 1.
        zero_pos : dict, optional
            Dictionary with relative intensities of the zero position. The default is None.
            Intensities will be normalized to sum up to 1.
        zero_pos_format : str, optional
            Letter-case format of zero position letters (lower or upper). The default is 'lower'.
        drop_aa : list, optional
            Amino acids to drop. The default is ['s'].
        drop_pos : list, optional
            Position to drop. The default is [].
        replace_aa_labels : dict, optional
            Renaming amino acids for sequence logo. The default is {'t' :'pS/pT', 'y':'pY'}.
        title : str, optional
            Figure title. The default is None.
        xlabel : str, optional
            X-axis label. The default is 'Position'. If False, no label will be diplayed.
        ylabel : str, optional
            Y-axis label. If None, specified based on logo_type. If False, no label will be diplayed. The default is None.
        save_fig : str, optional
            Path to save the figure. The default is False.
        return_fig : bool, optional
            If True - return the figure object. The default is False.
        plot : bool, optional
            If False - sequence logo will not be displayed. The default is True.
        ax : plt.axes, optional
            Provided axis on which to plot the sequence logo.
        **seq_logo_kwargs : args
            Optional keyword arguments passed into the seq_logo function.

        Returns
        -------
        fig : plt.figure()
            If specified, returning the figure object with the sequence logo.
        """
        
        plot_matrix = self.norm_matrix.drop(drop_aa, errors='ignore').drop(drop_pos, axis=1, errors='ignore')
        plot_matrix = plot_matrix.rename(index = replace_aa_labels)
        
        if zero_pos is None:
            if self.kin_type is None:
                raise Exception('Either \'kin_type\' or \'zero_pos\' must be specified.')
            if self.kin_type == 'tyrosine':
                zero_pos = {'y': 1}
            elif self.kin_type == 'ser_thr':
                zero_pos = self.st_fav
            
            if zero_pos_format == 'lower':
                zero_pos = {k.lower():v for k,v in zero_pos.items()}
            elif zero_pos_format == 'upper':
                zero_pos = {k.upper():v for k,v in zero_pos.items()}
            else:
                raise Exception('\'zero_pos_format\' must be either \'lower\' or \'upper\'.')
        
        return utils.make_seq_logo(plot_matrix, logo_type=logo_type, zero_pos=zero_pos,
                                   random_aa_value=self.random_aa_value,
                                   title=title, xlabel=xlabel, ylabel=ylabel,
                                   save_fig=save_fig, return_fig=return_fig, plot=plot, ax=ax,
                                   **seq_logo_kwargs)
    
    
    def plot_data(self, seq_logo=True, heatmap=True, st_fav=True, title=None,
                  fig_aspect=1.5, plot=True, save_fig=False, return_fig=False,
                  mat_scale='log', seq_logo_kwargs={}, heatmap_kwargs={}):
        """
        Generates a kinase ID figure including a sequence logo, heatmap, and s/t favorability plot.

        Parameters
        ----------
        seq_logo : bool, optional
            If True, a sequence logo is included in the figure. The default is True.
        heatmap : bool, optional
            If True, a heatmap is included in the figure. The default is True.
        st_fav : bool, optional
            If True, a s/t favorability plot is included in the figure. The default is True.
        title : str, optional
            Title for the subplots. The default is None, meaning the subplots will use self.name.
        fig_aspect : float, optional
            The height to width ratio of the figure. The default is 1.5.
        plot : bool, optional
            Whether or not to plot the produced figure. The default is True.
        save_fig : str, optional
            Path to file for saving the figure. The default is False.
        return_fig : bool, optional
            If true, the figure will be returned as a plt.figure object. The default is False.
        mat_scale : string, optional
            Scale of matrix - linear-scale ('linear') or log-scale ('log').
        seq_logo_kwargs : dict/args
            Optional keyword arguments passed into the seq_logo function.
        heatmap_kwargs : dict/args
            Optional keyword arguments passed into the heatmap function.

        Returns
        -------
        plt.figure
            If return_fig, the kinase figure will be returned.
        """
        
        w,h = plt.figaspect(fig_aspect)
        fig,ax = plt.subplots(3, 2, figsize=(w,h), gridspec_kw={'height_ratios': [0.4, 1, 0.1], 'width_ratios': [1, 0.05]})
        ax[0,1].axis('off')
        ax[2,1].axis('off')
        
        if seq_logo:
            self.seq_logo(ax=ax[0,0], xlabel_size=8, ylabel_size=8, yticks_fontsize=6, xlabel=False, xticks=False, **seq_logo_kwargs)
        else:
            ax[0,0].axis('off')
        if heatmap:
            self.heatmap(ax=ax[1,0], cbar_ax=ax[1,1], square=False, zero_pos=True, title=False, xticks_fontsize=12, yticks_fontsize=9, mat_scale=mat_scale, **heatmap_kwargs)
        else:
            ax[1,0].axis('off')
            ax[1,1].axis('off')
        if self.kin_type == 'ser_thr' and st_fav:
            self._plot_st_fav(ax=ax[2,0], title=None, value_annot=False, x_text=0.5, y_text=1.1, va_text='bottom')
            if  heatmap:
                con1 = patches.ConnectionPatch(xyA=((self.norm_matrix.columns<0).sum(),len(self.norm_matrix)), xyB=(0,1),
                                               coordsA='data', coordsB='data', axesA=ax[1,0], axesB=ax[2,0], color='black')
                con2 = patches.ConnectionPatch(xyA=((self.norm_matrix.columns<0).sum()+1,len(self.norm_matrix)), xyB=(1,1),
                                               coordsA='data', coordsB='data', axesA=ax[1,0], axesB=ax[2,0], color='black')
                ax[1,0].add_artist(con1)
                ax[1,0].add_artist(con2)
        else:
            ax[2,0].axis('off')
        
        if title != False:
            if title is None:
                title = self.name
            if seq_logo:
                ax[0,0].set_title(title)
            elif heatmap:
                ax[1,0].set_title(title)
            elif st_fav:
                ax[2,0].set_title(title)
        fig.tight_layout()
        
        if save_fig:
            fig.savefig(save_fig, dpi=1000)
            
        if not plot:
            plt.close(fig)
                
        if return_fig:
            return fig
    
    
    def score(self, subs, pp=False, phos_pos=None, phos_acc_fav=None,
              log2_score=True, output_type = 'series',
              round_digits=2,
              validate_phos_res=True, validate_aa=True,
              **sub_args):
        """
        Calculate score of the given substrates for the kinase.
        
        Score is being computed in a vectorized way:
            1. Making binary matrix for the substrates.
            2. Converting kinase matrix (norm-scaled) to log2
            3. Performing dot-product (summing the corresponding log2 of the kinase matrix)

        Parameters
        ----------
        subs : str or list
            List of substrates to score.
        pp : bool, optional
            Phosphopriming. The default is None.
            If not specified, will be inferred from the substrate.
        phos_pos : int, optional
            Position of phosphoacceptor. The default is None.
        phos_acc_fav : dict, optional
            Central phosphoacceptor favorability (in normalized scaled format). The default is None.
        log2_score : bool, optional
            Return scores as log2. The default is True.
        output_type : str, optional
            Type of returned data. The default is 'series'.
            'series': pd.Series, kinases as index.
            'list': list of values (same order as input kinase list).
            'dict': dictionary (kinase -> value).
        round_digits : int, optional
            Number of decimal digits. The default is 2.
        validate_phos_res : bool, optional
            validating phosphoacceptor. The default is True.
        validate_aa : bool, optional
            Validating amino acids. The default is True.
        **sub_args : args
            Arguments for the Substrate.score function.

        Returns
        -------
        score_output : pd.Series, list, or dictionary
            Scores of the specified substrates for the kinase.
        """
        
        exceptions.check_score_output_type(output_type)
        
        if isinstance(subs, str):
            subs = [subs]
        subs_list = pd.Series([utils.sequence_to_substrate(s, pp=pp, phos_pos=phos_pos, validate_phos_res=validate_phos_res, validate_aa=validate_aa, kin_type=self.kin_type) for s in subs])
        subs_bin_mat = utils.sub_binary_matrix(subs_list, aa=self.amino_acids, pos=self.positions)

        kin_mat_log2 = self.log2_matrix
        
        kin_vector = utils.flatten_matrix(kin_mat_log2)
        
        score_log2 = pd.Series(np.dot(subs_bin_mat,kin_vector.transpose()).transpose()[0], index=subs_bin_mat.index, name=self.name).round(round_digits)
        if phos_acc_fav:
            phos_acc_fav =  {k.lower(): v for k, v in phos_acc_fav.items()}
            if not set(score_log2.index.str[7].unique()) <= set(phos_acc_fav.keys()):
                raise Exception(f'Phosphoacceptor favorability must be provided for all phosphoacceptors (missing for {[x for x in score_log2.index.str[7].unique() if x not in phos_acc_fav.keys()]}).')
            phos_acc_fav_scores = score_log2.index.str[7].map(phos_acc_fav).to_series()
            phos_acc_fav_scores.index = score_log2.index
            phos_acc_fav_scores_log2 = np.log2(phos_acc_fav_scores)
            score_log2 = score_log2 + phos_acc_fav_scores_log2
        score = np.power(2,score_log2)
        
        if log2_score:
            score_output = score_log2.round(round_digits)
        else:
            score_output = score.round(round_digits)
    
        if output_type == 'series':
            return(score_output)
        elif output_type == 'dict':
            return(dict(score_output))
    
    
    def percentile(self, subs, pp=False, phos_pos=None, phos_acc_fav=None,
                   output_type='series', customized_scored_phosprot=None,
                   round_digits=2, validate_phos_res=True, validate_aa=True,
                   **sub_args):
        """
        Calculate the percentile score of the given substrate for the kinase.
        
        After score is being computed, the percentile of that score is being
        computed based on a basal scored phosphoproteome.
        Default: PhosphoSitePlus phosphorylation sites data base (07-2021)
                 High confidence sites:
                     * At least one low-throughput literateure report (LT_LIT)
                     OR
                     * At least 5 high-throughput (mass spectrometry) reports (MS_LIT + MS_CST)

        Parameters
        ----------
        subs : str or list
            List of substrates to score.
        pp : bool, optional
            Phosphopriming. The default is False.
            If not specified, will be inferred from the substrate.
        phos_pos : int, optional
            Position of phosphoacceptor. The default is None.
        phos_acc_fav : dict, optional
            Central phosphoacceptor favorability (in normalized scaled format). The default is None.
        output_type : str, optional
            Type of returned data. The default is 'series'.
            'series': pd.Series, kinases as index.
            'list': list of values (same order as input kinase list).
            'dict': dictionary (kinase -> value).
        customized_scored_phosprot : kl.ScoredPhosphoProteome, optional
            Customized phosphoproteome object. The default is None.
        round_digits : int, optional
            Number of decimal digits. The default is 2.
        validate_phos_res : bool, optional
            validating phosphoacceptor. The default is True.
        validate_aa : bool, optional
            Validating amino acids. The default is True.
        **sub_args : args
            Arguments for kl.Substrate class.

        Returns
        -------
        percent_output : pd.Series, list, or dictionary
            Percentiles of the specified substrates for the kinase.
        """
        
        exceptions.check_score_output_type(output_type)
        if self.kin_type not in _global_vars.valid_kin_types:
            raise Exception('In order to calculate percentile, kin_type must be one of the following: {}. Please reload Kianse object.'.format(_global_vars.valid_kin_types))
        
        if isinstance(subs, str):
            subs = [subs]
        subs_list = pd.Series([utils.sequence_to_substrate(s, pp=pp, phos_pos=phos_pos, validate_phos_res=validate_phos_res, validate_aa=validate_aa, kin_type=self.kin_type) for s in subs])
        
        score = self.score(subs=subs_list, pp=pp, phos_pos=phos_pos, phos_acc_fav=phos_acc_fav, log2_score=True, round_digits=round_digits, validate_phos_res=validate_phos_res, validate_aa=validate_aa, **sub_args)
        
        phosprot = data.get_phosphoproteome(kin_type=self.kin_type)
        phosprot_subs = phosprot[_global_vars.default_seq_col].to_list()
        scored_phosprot = self.score(subs=phosprot_subs, pp=pp, log2_score=True, round_digits=round_digits, validate_phos_res=validate_phos_res, validate_aa=validate_aa)
        
        percent_values = scored_phosprot.sort_values().searchsorted(score, side='right')/len(scored_phosprot)*100
        percent_values = np.round(percent_values, round_digits)
        
        if output_type == 'series':
            percentile_output = pd.Series(percent_values, index=score.index, name=self.name)
        elif output_type == 'dict':
            percentile_output = dict(zip(score.index, percent_values))
        
        return(percentile_output)
    

#%%

class ScoredPhosphoProteome(object):
    """
    Class of scored phosphoproteomes for calculating percentile

    Parameters
    ----------
    phosprot_name : str
        Name of phosphoproteome database.
    kin_type : list or str, optional
        Kinase types to read the scored phosphoproteome for. The default is None.
    phosprot_file : str, optional
        Customized scored phosphoproteome file suffix. The default is None.
        Files must be in format *kinase type*_*file suffix*
    phosprot_path : str, optional
        Path to scored phosphoproteome files. The default is './../databases/substrates/scored_phosprots'.
        Files must be in format *kinase type*_*file suffix*
    log2_values : bool, optional
        Score values are log2 transformed. The default is True.
    file_type : str, optional
        File type ('parquet', 'txt', 'tsv', or 'csv'). The default is 'parquet'.

    Examples
    -------
    >>> spp = kl.ScoredPhosphoProteome()
    >>> spp.ser_thr
                           AAK1  ACVR2A  ACVR2B    AKT1  ...   YANK3    YSK1    YSK4     ZAK
        SITE_+/-7_AA                                     ...                                
        __MtMDksELVQkAk -3.6796  1.8284  1.9022 -5.8160  ... -0.1768 -7.1092 -2.2866 -3.0714
        NEERNLLsVAykNVV -6.6282 -0.4899 -0.4783  0.4972  ... -0.0312 -0.9182 -0.9721  0.9999
        VVGARRssWRVISsI -6.4300 -1.9120 -2.5666  6.0576  ...  3.0207  1.4549  1.3652 -0.3073
        FyYEILNsPEKACSL -3.1932 -3.3275 -2.8763 -4.6189  ... -0.8332 -6.6627 -4.3845 -3.3388
        VEERNLLsVAykNVI -6.6282 -0.4899 -0.4783  0.4972  ... -0.0312 -0.9182 -0.9721  0.9999
                        ...     ...     ...     ...  ...     ...     ...     ...     ...
        RRQTEPVsPVLKRIK -6.1655 -2.6935 -2.5030 -6.5537  ... -2.0809 -6.9858 -4.0662 -3.3815
        LRSEAPNssEEDsPI -8.2035 -0.9307 -0.9038 -5.5443  ...  0.0879 -7.3131 -2.7316 -4.1168
        RSEAPNssEEDsPIK -7.1162  1.3564  2.0206 -7.3346  ... -0.6860 -9.1800 -3.2365 -4.6484
        PNssEEDsPIKSDKE -6.8779  1.4739  1.1953 -7.4137  ... -2.0387 -7.3567 -2.3573 -1.9678
        GLPARPksPLDPKKD -4.3794 -5.1915 -5.2657  0.5140  ...  1.0866 -6.2913 -3.9802 -4.7197
    """
    
    def __init__(self, phosprot_name, kin_type=None, phosprot_file=None, phosprot_path='./../databases/substrates',
                 phosprot_data_file=None, phosprot_data_path='./../databases/substrates',
                 log2_values=True, file_type='parquet'):
        
        current_dir = os.path.dirname(__file__)
        phosprot_path = os.path.join(current_dir, phosprot_path)
        phosprot_data_path = os.path.join(current_dir, phosprot_data_path)

        if phosprot_name not in data.get_phosphoproteomes_list():
            raise ValueError(f'Phosphoproteome named \'{phosprot_name}\' was not found. Use kl.get_phosphoproteomes_list() to get a list of available phosphoproteomes.')

        if kin_type is None:
            kin_type = utils._global_vars.valid_kin_types
        if isinstance(kin_type, str):
            kin_type = [kin_type]
        
        exceptions.check_phosprot_file_type(file_type)

        for kt in kin_type:
            if phosprot_file is None:
                scored_phosprot_file = phosprot_path + '/' + phosprot_name + '/scored_phosprots/' + kt + '_phosphoproteome_scored.' + file_type
            if file_type == 'parquet':
                setattr(self, kt+'_scores', pq.read_table(scored_phosprot_file).to_pandas())
            elif file_type == 'csv':
                setattr(self, kt+'_scores', pd.read_csv(scored_phosprot_file, index_col = 0))
            else:
                setattr(self, kt+'_scores', pd.read_csv(scored_phosprot_file, sep = '\t', index_col = 0))
            
            if phosprot_data_file is None:
                scored_phosprot_data_file = phosprot_path + '/' + phosprot_name + '/' + 'phosphoproteome_' + kt + '.txt'
            setattr(self, kt+'_data', pd.read_csv(scored_phosprot_data_file, sep = '\t'))
        
        self.log2_values = log2_values
    
    def merge_data_scores(self, kin_type, data_seq_col=None):
        """
        Merging phosphoproteome data and scores

        Parameters
        ----------
        kin_type : str
            Kinase type (ser_thr or tyrosine).
        data_seq_col : str, optional
            Sequence column name in data file. The default is None (will be set as _global_vars.default_seq_col).

        Returns
        -------
        merged_data : dataframe
            Merged dataframe of the phosphoproteome data and scores.
        """
        
        if data_seq_col is None:
            data_seq_col = _global_vars.default_seq_col
        
        data = getattr(self, kin_type+'_data').set_index(data_seq_col, drop=False)
        scores = getattr(self, kin_type+'_scores')
        try:
            merged_data = pd.concat([data,scores], axis=1)
        except:
            raise Exception('Sequence column in data file is not identical to the index in the scored phosphoproteome file.')
        
        return(merged_data)


#%%

class Kinome(object):
    """
    Kinase Library object for a kinome.

    Parameters
    ----------
    name : str
        Kinase name.
    matrix : np.ndarray or pd.DataFrame
        Kinase matrix.
    kin_type : str, optional
        Kinase type. The default is None.
    family : str, optional
        Kinase family. The default is None.
    pp : bool, optional
        Phospho-residues in the matrix (phospho-priming). The default is True.
    k_mod : bool, optional
        Modified lysine (acetylation and tri-methylation). The default is False.
    mat_type : str, optional
        Matrix type ('densitometry', 'raw', 'norm', 'log2', or customized). The default is 'log2'.
    cols : list, optional
        Matrix columns. Must fit the shape of the matrix. The default is None.
    rows : list, optional
        Matrix rows. Must fit the shape of the matrix. The default is None.

    Examples
    -------
    >>> matrix = kl.get_matrix('SRPK1')
    >>> kinase = kl.Kinase('SRPK1', matrix)
    >>> kinase.name
        'SRPK1'
    >>> kinase.matrix
               -5      -4      -3      -2      -1       1       2       3       4
        P  0.0594  0.0812  0.0353  0.0604  0.1116  0.2793  0.0786  0.0996  0.0798
        G  0.0753  0.0672  0.0373  0.0790  0.0564  0.0554  0.0609  0.0536  0.0661
        A  0.0889  0.0713  0.0477  0.0970  0.0661  0.0500  0.0724  0.0401  0.0611
        C  0.0814  0.0588  0.0384  0.0673  0.0462  0.0504  0.0666  0.0393  0.0635
        S  0.0525  0.0484  0.0255  0.0577  0.0436  0.0364  0.0609  0.0294  0.0581
        T  0.0525  0.0484  0.0255  0.0577  0.0436  0.0364  0.0609  0.0294  0.0581
        V  0.0517  0.0388  0.0202  0.0317  0.0411  0.0424  0.0421  0.0269  0.0414
        I  0.0468  0.0433  0.0219  0.0300  0.0326  0.0357  0.0346  0.0270  0.0484
        L  0.0464  0.0426  0.0192  0.0253  0.0436  0.0411  0.0367  0.0270  0.0460
        M  0.0525  0.0484  0.0178  0.0294  0.0425  0.0364  0.0421  0.0294  0.0455
        F  0.0588  0.0420  0.0175  0.0375  0.0280  0.0228  0.0368  0.0250  0.0436
        Y  0.0488  0.0454  0.0189  0.0349  0.0336  0.0222  0.0357  0.0258  0.0438
        W  0.0611  0.0432  0.0183  0.0363  0.0247  0.0245  0.0342  0.0266  0.0521
        H  0.0561  0.0562  0.0267  0.1012  0.0709  0.0331  0.0947  0.0410  0.0581
        K  0.0927  0.0945  0.0599  0.0787  0.1291  0.0910  0.0673  0.0843  0.0866
        R  0.0851  0.1347  0.5477  0.0944  0.1279  0.1368  0.0746  0.3521  0.0822
        Q  0.0397  0.0512  0.0350  0.0407  0.0685  0.0548  0.0577  0.0411  0.0645
        N  0.0510  0.0566  0.0255  0.0701  0.0602  0.0360  0.0677  0.0468  0.0620
        D  0.0453  0.0401  0.0298  0.0957  0.0286  0.0161  0.0859  0.0293  0.0667
        E  0.0403  0.0431  0.0214  0.0577  0.0347  0.0222  0.0781  0.0242  0.0521
        s  0.0450  0.0454  0.0300  0.0401  0.0322  0.0287  0.0740  0.0344  0.0701
        t  0.0450  0.0454  0.0300  0.0401  0.0322  0.0287  0.0740  0.0344  0.0701
        y  0.0557  0.0838  0.0408  0.0820  0.0443  0.0272  0.0559  0.0384  0.0622
    >>> kinase.kin_type
        'ser_thr'
    >>> kinase.family
        'CMGC'
    >>> kinase.info
        MATRIX_NAME                                                           SRPK1
        GENENAME                                                              SRPK1
        TYPE                                                                ser_thr
        SUBTYPE                                                                 STK
        FAM                                                                    CMGC
        UNIPROT_ID                                                           Q96SB4
        UNIPROT_ENTRY_NAME                                              SRPK1_HUMAN
        EMBL_ID                                                              U09564
        EMBL                                                             AAA20530.1
        ENSEMBL_GENE_ID                                             ENSG00000096063
        ENSEMBL_TRS_ID            ENST00000373825;ENST00000361690;ENST0000034616...
        ENSEMBL_PRO_ID                                              ENSP00000362931
        P_ENTREZGENEID                                                       6732.0
        P_GI                                                               82407376
        PIR                                                                  S45337
        REFSEQ_NT_ID                                                    NM_003137.4
        P_REFSEQ_AC                                                     NP_003128.3
        PDB_ID                                                                 1WAK
        BIOGRID_ID                                                         112610.0
        DIP_ID                                                           DIP-33888N
        STRING_ID                                              9606.ENSP00000362931
        CHEMBL_ID                                                        CHEMBL4375
        DRUGBANK_ID                                                             NaN
        GUIDETOPHARMACOLOGY_ID                                               2208.0
        BIOMUTA_ID                                                            SRPK1
        DMDM_ID                                                         209572680.0
        CPTAC_ID                                                                NaN
        PROTEOMICSDB_ID                                                       78098
        DNASU_ID                                                             6732.0
        REACTOME_ID                                                             NaN
        CHITARS_ID                                                            SRPK1
        GENEWIKI_ID                                                           SRPK1
        GENOMERNAI_ID                                                        6732.0
        HGNC_ID                                                          HGNC:11305
        Name: SRPK1, dtype: object
        """
    
    def __init__(self, kin_type=None, family=None):
        
        if kin_type is None:
            kin_type = utils._global_vars.valid_kin_types
        if isinstance(kin_type, str):
            kin_type = [kin_type]
        
    
    def _get_kinome_matrices(self, kin_type, mat_type, excld_kins=[], aa=None, pos=None, as_dict=False):
        """
        Making a data frame with all the matrices of a kinase type except for excluded kinases
    
        Parameters
        ----------
        kin_type : str
            Kinase type (ser_thr or tyrosine).
        mat_type : str
            Matrix type (raw, normalized, or normalized-scaled).
        excld_kins : list, optional
            List of kinases to exclude. The default is [].
        aa : list
            List of the amino acid labels to use.
        pos : list, optional
            List of specific positions to use.
        as_dict : bool, optional
            If True, return values as dictionary. The default is False.
    
        Returns
        -------
        df_kin_full_mat : dataframe
            Dataframe with all the wanted matrices as rows.
        kin_mat_dict : dictionary
            if 'as_dict' is True, returns a dictionary of kinases and their matrices.
        """
        
        current_dir = os.path.dirname(__file__)
        mat_dir = os.path.join(current_dir, _global_vars.mat_dir)
        
        exceptions.check_kin_type(kin_type)
        exceptions.check_mat_type(mat_type)
            
        if aa is None:
            aa = ['P','G','A','C','S','T','V','I','L','M','F','Y','W','H','K','R','Q','N','D','E','s','t','y']
        if pos is None:
            pos = _global_vars.ser_thr_pos*(kin_type == 'ser_thr') + _global_vars.tyrosine_pos*(kin_type == 'tyrosine')
        
        excld_kins = [x.upper() for x in excld_kins]
        
        aa_pos = []
        for p in pos:
            for a in aa:
                aa_pos.append(str(p) + a)
        
        kin_path = mat_dir + '/' + kin_type + '/' + mat_type
        kin_list = [name.split('.')[0].split('_')[0] for name in os.listdir(kin_path) if (name.startswith(".") == False)]
        kin_list.sort()
        
        wanted_kins = [x for x in kin_list if x not in excld_kins]
        
        kin_full_mat = []
        kin_mat_dict = {}
        
        for kin in tqdm(wanted_kins):
            kin_mat = data.get_matrix(kin, kin_type, aa=aa, pos=pos)
            kin_mat_dict[kin] = kin_mat
            kin_full_mat.append(kin_mat.values.reshape(kin_mat.shape[0]*kin_mat.shape[1],1))
    
        kin_full_mat = np.hstack(kin_full_mat)
        df_kin_full_mat = pd.DataFrame(kin_full_mat, columns = wanted_kins, index = aa_pos).transpose()
        
        if as_dict:
            return(kin_mat_dict)
        
        return(df_kin_full_mat)
    