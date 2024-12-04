"""
###################################
# The Kinase Library - Exceptions #
###################################
"""

from ..utils import _global_vars, utils
from ..modules import data

#%%
"""
Kinases
"""

def check_kin_name(kinases, kin_type=None, valid_kin_list=None):
    if isinstance(kinases, str):
            kinases = [kinases]
    if valid_kin_list is None:
        if kin_type is None:
            valid_kin_list = data.get_kinase_list('ser_thr') + data.get_kinase_list('tyrosine', non_canonical=True)
        else:
            valid_kin_list = data.get_kinase_list(kin_type=kin_type, non_canonical=True)
    for kin in kinases:
        if kin.upper() not in valid_kin_list:
            raise Exception('Matrix was not found for kinase \'{}\'. Use \'get_kinase_list\' for available kinases.'.format(kin))
            
def check_kin_type(kin_type):
    if kin_type not in _global_vars.valid_kin_types:
        raise ValueError('kin_type must be one of the following: {}'.format(_global_vars.valid_kin_types))

def check_kin_list_type(kin_list, kin_type=None):
    if isinstance(kin_list, str):
        kin_list = [kin_list]
    check_kin_name(kin_list, kin_type=kin_type)
    if kin_type is None:
        kin_type = data.get_kinase_type(kin_list[0])
    kin_type_list = data.get_kinase_list(kin_type, non_canonical=True)
    for kin in kin_list:
        if kin.upper() not in kin_type_list:
            raise Exception('All kinases must be the same type (\'{}\' is not a \'{}\' kinase).'.format(kin,kin_type))

def check_mat_type(mat_type):
    if mat_type not in _global_vars.valid_mat_type:
        raise ValueError('mat_type must be one of the following: {}'.format(_global_vars.valid_mat_type))

def check_mat_scale(mat_scale):
    if mat_scale not in _global_vars.valid_mat_scale:
        raise ValueError('mat_scale must be one of the following: {}'.format(_global_vars.valid_mat_scale))

def check_name_type(name_type):
    if name_type not in _global_vars.valid_name_types:
        raise ValueError('name_type must be one of the following: {}'.format(_global_vars.valid_name_type))

#%%
"""
Substrates
"""

def check_sub_kin_type(sub, kin, sub_type=None, kin_type=None):
    if sub_type is None:
        sub_type = utils.substrate_type(sub)
    if kin_type is None:
        kin_type = data.get_kinase_type(kin)
    if sub_type != kin_type:
        raise Exception('Substrate type ({}) does not match kinase type ({}).'.format(sub_type,kin_type))

def check_sub(sub, validate_length=True, validate_phos_res=True, validate_aa=True):
    if validate_length:
        if len(sub) != 15:
            raise Exception('Substrate length must be 15 amino acids.')
    if validate_phos_res:
        if sub[7].lower() not in ['s','t','y']:
            raise Exception('Central residue must be a phosphoacceptor (s/t/y).')
    if validate_aa:
        if not _global_vars.valid_aa.issuperset(sub.upper()):
            raise Exception('Substrate contains invalid amino acids or characters.\nAllowed characters are: {}'.format(_global_vars.valid_aa))

#%%
"""
Scoring
"""

def check_scoring_metric(scoring_metric):
    if scoring_metric not in _global_vars.valid_scoring_metric:
        raise ValueError('Scoring metric must be one of the following: {}'.format(_global_vars.valid_scoring_metric))

def check_score_output_type(output_type):
    if output_type not in _global_vars.valid_score_output_type:
        raise ValueError('Output type must be one of the following: {}'.format(_global_vars.valid_score_output_type))

def check_output_sort_by(sort_by):
    if sort_by not in _global_vars.valid_output_sort_by:
        raise ValueError('Output sorting method must be one of the following: {}'.format(_global_vars.valid_output_sort_by))

def check_phosprot_file_type(file_type):
    if file_type not in _global_vars.phosprot_file_type:
        raise ValueError('Phosphoproteome files type must be one of the following: {}'.format(_global_vars.phosprot_file_type))
        
def check_phosprot_name(phosprot_name):
    valid_phosprot_name = data.get_phosphoproteomes_list()
    if phosprot_name not in valid_phosprot_name:
        raise ValueError('Phosphoproteome named \'{}\' was not found. Use kl.get_phosphoproteomes_list() to get a list of available phosphoproteomes.'.format(phosprot_name))
      
#%%
"""
Phosphoproteomics
"""

def check_score_type(score_type):
    if score_type not in _global_vars.valid_score_type:
        raise ValueError('score_type must be one of the following: {}'.format(_global_vars.valid_score_type))

#%%
"""
Enrichment
"""

def check_enrichment_data_type(data_type):
    if data_type not in _global_vars.valid_enrichment_data_type:
        raise ValueError('data_type must be one of the following: {}'.format(_global_vars.valid_enrichment_data_type))

def check_dp_enrichment_type(enrich_type):
    if enrich_type not in _global_vars.valid_dp_enrichment_type:
        raise ValueError('enrichment_type must be one of the following: {}'.format(_global_vars.valid_dp_enrichment_type))

def check_dp_sites_type(sites_type):
    if sites_type not in _global_vars.valid_dp_sites_type:
        raise ValueError('sites_type must be one of the following: {}'.format(_global_vars.valid_dp_sites_type))

def check_kl_method(kl_method):
    if kl_method not in _global_vars.valid_kl_method:
        raise ValueError('kl_method must be one of the following: {}'.format(_global_vars.valid_kl_method))

def check_enrichment_type(enrichment_type):
    if enrichment_type not in _global_vars.valid_enrichment_type:
        raise ValueError('enrichment_type must be one of the following: {}'.format(_global_vars.valid_enrichment_type))

#%%
"""
Plotting
"""

def check_color_kins_method(color_kins_by):
    if color_kins_by not in _global_vars.valid_color_kins_method:
        raise ValueError('color_kins_by must be one of the followings: {}'.format(_global_vars.valid_color_kins_method))

def check_cluster_method(cluster_by):
    if cluster_by not in _global_vars.valid_cluster_method:
        raise ValueError('cluster_by must be one of the followings: {}'.format(_global_vars.valid_cluster_method))
