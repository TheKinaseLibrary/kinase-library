"""
################################
# The Kinase Library - Scoring #
################################
"""

from ..objects import phosphoproteomics
from ..utils import _global_vars, exceptions, utils
from ..modules import data

#%%
"""
#################
# Score Protein #
#################
"""

def score_protein(seq, kinases=None, phosphoacceptor=['S', 'T', 'Y'],
                   pp=False, st_fav=True, non_canonical=False,
                   score_promiscuity_threshold=1,
                   percentile_promiscuity_threshold=90,
                   score_round_digits=3, percentile_round_digits=2):
    """
    Score all potential phosphosites in a given protein sequence.


    seq : str
        Protein sequence to analyze for phosphorylation sites.
    kinases : list, optional
        List of specific kinases to consider in the analysis.
        If None, all available kinases are used. The default is None.
    phosphoacceptor : list, optional
        Amino acid residues that can be phosphorylated. The default is ['S', 'T', 'Y'].
    pp : bool, optional
        Phospho-residues (s/t/y). The default is False.
    st_fav : bool, optional
        S/T favorability. The default is True.
    non_canonical : bool, optional
        Return also non-canonical kinases. For tyrosine kinases only. The default is False.
    score_promiscuity_threshold : float, optional
        Score threshold above which kinases are considered predicted.
    percentile_promiscuity_threshold : float, optional
        Percentile threshold above which kinases are considered predicted.
    score_round_digits : int, optional
        Number of decimal digits for score. The default is 3.
    percentile_round_digits : int, optional
        Number of decimal digits for percentile. The default is 2.

    Raises
    ------
    ValueError
        If phosphoacceptor residues are not a subset of valid phosphorylatable residues.

    Returns
    -------
    results : dict
        Dictionary containing prediction results with two keys:
        - 'ser_thr': Predictions for serine/threonine kinases
        - 'tyrosine': Predictions for tyrosine kinases
        Each prediction includes kinase scores, percentiles, ranks, and promiscuity indices.

    """

    if not set(phosphoacceptor).issubset(_global_vars.valid_phos_res):
        raise ValueError(f'Phosphoacceptor(s) must be a subset of {_global_vars.valid_phos_res}.')
    if kinases is not None:
        kinases = [kin.upper() for kin in kinases]
        exceptions.check_kin_name(kinases)
        ser_thr_kins = [kin for kin in kinases if kin in data.get_kinase_list('ser_thr')]
        tyrosine_kins = [kin for kin in kinases if kin in data.get_kinase_list('tyrosine')]
    else:
        ser_thr_kins = None
        tyrosine_kins = None

    phos_sites = utils.parse_phosphosites(seq, phosphoacceptor=phosphoacceptor, pp=pp)

    pps = phosphoproteomics.PhosphoProteomics(data=phos_sites, seq_col='Sequence', pp=pp, new_seq_phos_res_cols=False, suppress_warnings=False)

    results = {'ser_thr': pps.predict(kin_type='ser_thr', kinases=ser_thr_kins, st_fav=st_fav,
                                      score_promiscuity_threshold=score_promiscuity_threshold,
                                      percentile_promiscuity_threshold=percentile_promiscuity_threshold,
                                      score_round_digits=score_round_digits,
                                      percentile_round_digits=percentile_round_digits),
               'tyrosine': pps.predict(kin_type='tyrosine', kinases=tyrosine_kins, non_canonical=non_canonical,
                                       score_promiscuity_threshold=score_promiscuity_threshold,
                                       percentile_promiscuity_threshold=percentile_promiscuity_threshold,
                                       score_round_digits=score_round_digits,
                                       percentile_round_digits=percentile_round_digits)
               }

    return(results)