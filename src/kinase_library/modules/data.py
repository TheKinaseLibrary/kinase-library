"""
####################################
# The Kinase Library - Import Data #
####################################
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq
from natsort import natsorted
import datetime
import shutil

from ..objects import core, phosphoproteomics
from ..utils import _global_vars, exceptions

#%%
"""
##############
# Parameters #
##############
"""

def get_positions(kin_type):
    """
    Get the positions of the library.

    Parameters
    ----------
    kin_type : str
        Kinase type (ser_thr, tyrosine).

    Returns
    -------
    Positions of the library.

    """
    exceptions.check_kin_type(kin_type)
    if kin_type == 'ser_thr':
        return(_global_vars.ser_thr_pos)
    elif kin_type == 'tyrosine':
        return(_global_vars.tyrosine_pos)


def get_aa(pp=True):
    """
    Get amino acids of the library.

    Parameters
    ----------
    pp : bool, optional
        Phospho-priming amino acids. The default is True.

    Returns
    -------
    Amino acids fo the library.

    """
    if pp:
        return(_global_vars.aa_phos)
    else:
        return(_global_vars.aa_unmod)
    

#%%
"""
############
# Matrices #
############
"""

def get_kinase_list(kin_type, family=None, subtype=None, non_canonical=False):
    """
    Get a list of all available kinases (matrices) of a certain kinase type and family.

    Parameters
    ----------
    kin_type : str
        Kinase type (ser_thr, tyrosine).
    family : str or list, optional
        Kinase family.
    subtype : str or list, optional
        Kinase subtype (relevant only for tyrosine kinases).
    non_canonical : bool, optional
        Return also non-canonical kinases. For tyrosine kinases only. The default is False.

    Returns
    -------
    kin_list : list
        List of available kinases from that type.
    """
    current_dir = os.path.dirname(__file__)
    mat_dir = os.path.join(current_dir, _global_vars.mat_dir)

    exceptions.check_kin_type(kin_type)
    
    if isinstance(family, str):
        family = [family]
        family = [x.upper() for x in family]
    if isinstance(subtype, str):
        subtype = [subtype]
        subtype = [x.upper() for x in subtype]
    kinome_info = get_kinome_info(kin_type)

    full_kin_list = natsorted([name.split('.')[0] for name in os.listdir(mat_dir + '/' + kin_type + '/norm/') if (name.startswith(".") == False)])
    if not non_canonical:
        full_kin_list = [x for x in full_kin_list if '_TYR' not in x]
    if family is not None:
        family_kins = kinome_info[kinome_info['FAMILY'].str.upper().isin(family)]['MATRIX_NAME'].to_list()
        kin_list = [x for x in full_kin_list if x in family_kins]
    else:
        kin_list = full_kin_list
    
    if subtype is not None:
        subtype_kins = kinome_info[kinome_info['SUBTYPE'].str.upper().isin(subtype)]['MATRIX_NAME'].to_list()
        kin_list = [x for x in kin_list if x in subtype_kins]
    
    return(kin_list)
    
    
def get_densitometry(kinase, kin_type=None, rotate=False):
    """
    Get densitometry matrix from database.

    Parameters
    ----------
    kinase : str
        Kinase name.
    kin_type : str, optional
        Kinase type (ser_thr, tyrosine). The default is None. 
    rotate : bool, optional
        Rotating the matrix 90 degrees clockwise. The default is False.

    Returns
    -------
    dens_mat : pd.DataFrame
        Kinase densitometry matrix.
    """
    
    current_dir = os.path.dirname(__file__)
    mat_dir = os.path.join(current_dir, _global_vars.mat_dir)
    
    if kin_type is None:
        kin_type = get_kinase_type(kinase)
    else:
        exceptions.check_kin_type(kin_type)
    
    kinase = kinase.upper()
    
    kin_list = [name.split('.')[0].split('_')[0] for name in os.listdir(mat_dir + '/' + kin_type + '/densitometry/') if (name.startswith(".") == False)]
    kin_list.sort()

    exceptions.check_kin_name(kinase, kin_type=kin_type, valid_kin_list=kin_list)
    
    dens_mat = pd.read_csv(mat_dir + '/' + kin_type + '/densitometry/' + kinase + '_densitometry.txt', sep = '\t', index_col = 0)
    
    if rotate:
        dens_mat = dens_mat.transpose().loc[:,::-1]
    
    return(dens_mat)


def get_matrix(kinase, kin_type=None, mat_type='log2', aa=None, pos=None, pp=True, transpose=False):
    """
    Read kinase matrix from file as pd.DataFrame.

    Parameters
    ----------
    kinase : str
        Kinase name.
    kin_type : str
        Kinase type (ser_thr, tyrosine).
    mat_type : str, optional
        Matrix type (raw, normalized, or log-scaled).
    aa : list, optional
        List of amino acids to select. The default is None.
    pos : list, optional
        Positions to select. The default is None.
    pp : bool, optional
        Phospho-priming columns. The default is True.
    transpose : bool, optional
        If True, return transposed matrix. The default is False.

    Returns
    -------
    kin_matrix : pd.DataFrame
        Kinase matrix as dataframe.
    """

    current_dir = os.path.dirname(__file__)
    mat_dir = os.path.join(current_dir, _global_vars.mat_dir)
    
    exceptions.check_kin_name(kinase, kin_type=kin_type)
    if kin_type is None:
        kin_type = get_kinase_type(kinase)
    else:
        exceptions.check_kin_type(kin_type)
    exceptions.check_mat_type(mat_type)
    
    kinase = kinase.upper()
    
    if pos is None:
        pos = get_positions(kin_type)
    if aa is None:
        aa = get_aa(pp=pp)
    
    kin_list = [name.split('.')[0] for name in os.listdir(mat_dir + '/' + kin_type + '/' + mat_type + '/') if (name.startswith(".") == False)]
    kin_list.sort()

    exceptions.check_kin_name(kinase, kin_type=kin_type, valid_kin_list=kin_list)
    
    full_matrix = pd.read_csv(mat_dir + '/' + kin_type + '/' + mat_type + '/' + kinase + '.tsv', sep = '\t', index_col = 0)
    kin_matrix = full_matrix.loc[pos, aa]
    
    # Matrices are saved with amino acids as columns and positions as rows
    # However default presentation is amino acids as rows and positions as columns
    if not transpose:
        kin_matrix = kin_matrix.transpose()
    
    return(kin_matrix)


def get_multiple_matrices(kinases, kin_type=None, mat_type='log2',
                          aa=None, pos=None, pp=True, as_dict=False):
    """
    Making a data frame with the matrices of all the wanted kinases as rows

    Parameters
    ----------
    kinases : list (or str)
        List of kinases to retrieve. Optional to pass only one kinase as string.
    kin_type : str
        Kinase type (ser_thr or tyrosine).
    mat_type : str, optional
        Matrix type (raw, normalized, or log-scaled).
    aa : list
        List of the amino acid labels to use.
    pos : list, optional
        List of specific positions to use.
    pp : bool, optional
        Phospho-priming columns. The default is True.
    as_dict : bool, optional
        If True, return values as dictionary. The default is False.

    Returns
    -------
    df_kin_full_mat : dataframe
        Dataframe with all the wanted matrices as rows.
    kin_mat_dict : dictionary
        if 'as_dict' is True, returns a dictionary of kinases and their matrices.
    """
    
    if isinstance(kinases, str):
        kinases = [kinases]
    kinases = [x.upper() for x in kinases]

    current_dir = os.path.dirname(__file__)
    mat_dir = os.path.join(current_dir, _global_vars.mat_dir)
    
    exceptions.check_kin_name(kinases, kin_type=kin_type)
    exceptions.check_kin_list_type(kinases, kin_type=kin_type)
    if kin_type is None:
        kin_type = get_kinase_type(kinases[0])
    else:
        exceptions.check_kin_type(kin_type)
    exceptions.check_mat_type(mat_type)
        
    if aa is None:
        if pp:
            aa = _global_vars.aa_phos
        else:
            aa = _global_vars.aa_unmod
    if pos is None:
        pos = _global_vars.ser_thr_pos*(kin_type == 'ser_thr') + _global_vars.tyrosine_pos*(kin_type == 'tyrosine')
        
    aa_pos = []
    for p in pos:
        for a in aa:
            aa_pos.append(str(p) + a)
    
    if as_dict:
        kin_mat_dict = {}
        for kin in tqdm(kinases):
            kin_mat = get_matrix(kin, mat_type=mat_type, aa=aa, pos=pos)
            kin_mat_dict[kin] = kin_mat
        return(kin_mat_dict)
    
    all_kin_matrices = pd.read_csv(mat_dir + '/' + kin_type + '/' + kin_type + '_all_' + mat_type + '_matrices.tsv', sep = '\t', index_col = 0)
    kin_matrices = all_kin_matrices.loc[kinases,aa_pos]
    
    return(kin_matrices)


def get_all_matrices(kin_type, mat_type='log2', excld_kins=[],
                     aa=None, pos=None, pp=True, non_canonical=False, as_dict=False):
    """
    Making a data frame with all the matrices of a kinase type except for excluded kinases

    Parameters
    ----------
    kin_type : str
        Kinase type (ser_thr or tyrosine).
    mat_type : str, optional
        Matrix type (raw, normalized, or log-scaled).
    excld_kins : list, optional
        List of kinases to exclude. The default is [].
    aa : list
        List of the amino acid labels to use.
    pos : list, optional
        List of specific positions to use.
    pp : bool, optional
        Phospho-priming columns. The default is True.
    non_canonical : bool, optional
        Return also non-canonical kinases. For tyrosine kinases only. The default is False.
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
        if pp:
            aa = _global_vars.aa_phos
        else:
            aa = _global_vars.aa_unmod
    if pos is None:
        pos = _global_vars.ser_thr_pos*(kin_type == 'ser_thr') + _global_vars.tyrosine_pos*(kin_type == 'tyrosine')
    
    excld_kins = [x.upper() for x in excld_kins]
    
    aa_pos = []
    for p in pos:
        for a in aa:
            aa_pos.append(str(p) + a)
    
    kin_list = get_kinase_list(kin_type, non_canonical=non_canonical)
    
    wanted_kins = [x for x in kin_list if x not in excld_kins]
    
    kin_full_mat = []
    kin_mat_dict = {}
    
    for kin in tqdm(wanted_kins):
        kin_mat = get_matrix(kin, kin_type, aa=aa, pos=pos, mat_type=mat_type)
        kin_mat_dict[kin] = kin_mat
        kin_full_mat.append(kin_mat.T.values.reshape(kin_mat.shape[0]*kin_mat.shape[1],1))

    kin_full_mat = np.hstack(kin_full_mat)
    df_kin_full_mat = pd.DataFrame(kin_full_mat, columns = wanted_kins, index = aa_pos).transpose()
    
    if as_dict:
        return(kin_mat_dict)
    
    return(df_kin_full_mat)


def get_matrix_from_file(file, name=None, random_aa_value=None, kin_type='undefined', family='undefined', mat_type='customized', pp=True, k_mod=False, transpose=False):
    """
    Get kl.Kinase object from matrix file

    Parameters
    ----------
    file : str
        File of matrix.
    name : str, optional
        Kinase name. The default is None.
        If None - name will be the file name.
    kin_type : str, optional
        Kinase type. The default is 'undefined'.
    family : str, optional
        Kinase family. The default is 'undefined'.
    mat_type : str, optional
        Matrix type. The default is 'undefined'.
    pp : bool, optional
        Phospho-priming columns. The default is True.
    k_mod : bool, optional
        Modified K columns. The default is False.
    transpose : bool, optional
        If True = matrix will be transposed. The default is False.

    Returns
    -------
    kin_obj : kl.Kinase
        KL Kinase object with the matrix.
    """

    matrix = pd.read_csv(file, sep = '\t', index_col = 0)
    if transpose:
        matrix = matrix.T
    if name is None:
        name = file.split('/')[-1].split('.')[0]
    if random_aa_value is None:
        random_aa_value = 1/len(matrix)
    
    kin_obj = core.Kinase(name=name, matrix=matrix, random_aa_value=random_aa_value, kin_type=kin_type, family=family, pp=pp, k_mod=k_mod, mat_type=mat_type)
    
    return(kin_obj)


def get_st_fav(kinases,
               st_fav_file='st_favorability.tsv',
               as_dict=False, lower_case=False):
    """
    Get serine/threonine favorability.

    Parameters
    ----------
    kinases : list (or str)
        List of kinases to retrieve. Optional to pass only one kinase as string.
    st_fav_file : str, optional
        Name of S/T favorability file in the matrices directory. The default is 'st_favorability.tsv'.
    as_dict : bool, optional
        If True, return values as dictionary. The default is False.
    lower_case : bool, optional
        If True, return amino acids as lowercase. The default is False.

    Returns
    -------
    st_fav : pd.DataFrame or dictionary
        S/T favorability information.
    """
    
    current_dir = os.path.dirname(__file__)
    mat_dir = os.path.join(current_dir, _global_vars.mat_dir)
    st_fav_file = os.path.join(current_dir, _global_vars.mat_dir, 'ser_thr', st_fav_file)
    
    if isinstance(kinases, str):
        kinases = [kinases]
    kinases = [x.upper() for x in kinases]
    
    exceptions.check_kin_list_type(kinases, kin_type='ser_thr')

    all_st_fav = pd.read_csv(st_fav_file, sep = '\t', index_col=0)
    st_fav = all_st_fav.loc[kinases]
    
    if lower_case:
        st_fav.columns = [x.lower() for x in st_fav.columns]
    else:
        st_fav.columns = [x.upper() for x in st_fav.columns]
    
    if as_dict:
        return(st_fav.to_dict(orient='index'))
    
    return(st_fav)


def get_random_aa_value(kinase):
    """
    Get value of random amino acid according to profiled peptide library.

    Parameters
    ----------
    kinase : str
        Kinase name.

    Returns
    -------
    random_aa_value : float
        Value of random amino acid in the used peptide library.

    """
    
    kinase_info = get_kinase_info(kinase)
    random_aa_value = _global_vars.random_aa_value[kinase_info['KL_LIBRARY']]
    
    return(random_aa_value)


def get_current_mat_dir():
    """
    Printing current directory of matrices.

    Parameters
    ----------
    None.

    Returns
    -------
    Print current phosphoproteome name.

    """
    
    print(f'Current matrices directory is set to: {_global_vars.mat_dir}')


def set_current_mat_dir(mat_dir):
    """
    Setting the directory of matrices.

    Parameters
    ----------
    mat_dir : str
        Matrices directory.

    Returns
    -------
    None.

    """
    
    _global_vars.mat_dir = mat_dir
    print(f'Matrices directory was set to: {_global_vars.mat_dir}')


def reset_current_mat_dir():
    """
    Reset matrices directory to default.

    Parameters
    ----------
    None.

    Returns
    -------
    None.

    """
    
    _global_vars.reset_mat_dir()
    print(f'Matrices directory was reset to: {_global_vars.mat_dir}')

#%%
"""
######################
# Kinase Information #
######################
"""

def get_kinome_info(kin_type=None, columns=None, info_file='./../databases/kinase_data/kinome_information.tsv'):
    """
    Get all the information about the kinome.

    Parameters
    ----------
    kin_type : str, optional
        Kinase type. The default is None.
    columns : list, optional
        Columns to filter. The default is None.
    info_file : str, optional
        Path to information file. The default is './../databases/kinase_data/kinome_information.tsv'.

    Returns
    -------
    all_kin_info : pd.DataFrame
        All available information on the requested kinases (default is all the data).
    """
    if not os.path.isfile(info_file):
        current_dir = os.path.dirname(__file__)
        info_file = os.path.join(current_dir, info_file)
    
    all_kin_info = pd.read_csv(info_file, sep = '\t')
    if kin_type is not None:
        exceptions.check_kin_type(kin_type)
        all_kin_info = all_kin_info[all_kin_info['TYPE'] == kin_type]
    if columns is not None:
        all_kin_info = all_kin_info[columns]
    return(all_kin_info)


def get_kinase_info(kinase, name_type='matrix', info_file='./../databases/kinase_data/kinome_information.tsv'):
    """
    Get kinase information.

    Parameters
    ----------
    kinase : str
        Kinase name.
    name_type : str, optional
        Gene name or kinase name. The default is 'kinase'.
    info_file : str, optional
        Path to information file. The default is './../databases/kinase_data/kinome_information.tsv'.

    Returns
    -------
    kin_info : pd.Series
        Kinase information from the information file.
    """

    if not os.path.isfile(info_file):
        current_dir = os.path.dirname(__file__)
        info_file = os.path.join(current_dir, info_file)
    
    exceptions.check_name_type(name_type)
    if isinstance(kinase, str):
        kinase = kinase.upper()
    elif isinstance(kinase, list):
        kinase = [k.upper() for k in kinase]
    
    
    name_column = {'kinase': 'KINASE', 'gene': 'GENE_NAME', 'matrix': 'MATRIX_NAME', 'uniprot_id': 'UNIPROT_ID'}
    all_kin_info = get_kinome_info(info_file=info_file)
    all_kin_info = all_kin_info.set_index(name_column[name_type])
    all_kin_info.index = all_kin_info.index.str.upper()
    
    kin_info = all_kin_info.loc[kinase]
    
    return(kin_info)

def get_kinase_type(kinase, name_type='matrix', info_file='./../databases/kinase_data/kinome_information.tsv'):
    """
    Get kinase type.

    Parameters
    ----------
    kinase : str
        Kinase name.
    name_type : str, optional
        Gene name or kinase name. The default is 'kinase'.
    info_dir : str, optional
        Path to information folder. The default is './../databases/kinase_data/'.

    Returns
    -------
    kin_type : str
        Kinase type ('ser_thr' or 'tyrosine').
    """
    
    current_dir = os.path.dirname(__file__)
    info_file = os.path.join(current_dir, info_file)
    
    if isinstance(kinase, str):
        kinase = kinase.upper()
    elif isinstance(kinase, list):
        kinase = [k.upper() for k in kinase]
        
    kin_info = get_kinase_info(kinase=kinase, name_type=name_type, info_file=info_file)
    kin_type = kin_info['TYPE']
    return(kin_type)


def get_kinase_family(kinase, kin_type=None, family_col='FAMILY', name_type='matrix', info_file='./../databases/kinase_data/kinome_information.tsv'):
    """
    Get kinase family.

    Parameters
    ----------
    kinase : str
        Kinase name.
    kin_type : str, optional
        Kinase type. The default is None.
    family_col : str, optional
        Column in information file with family information. The default is 'FAMILY'.
    name_type : str, optional
        Gene name or kinase name. The default is 'kinase'.
    info_dir : str, optional
        Path to information folder. The default is './../databases/kinase_data/'.

    Returns
    -------
    Kinase family : str
        Kinase family from information file.
    """
    
    current_dir = os.path.dirname(__file__)
    info_file = os.path.join(current_dir, info_file)
    
    exceptions.check_kin_name(kinase, kin_type=kin_type)
    if isinstance(kinase, str):
        kinase = kinase.upper()
    elif isinstance(kinase, list):
        kinase = [k.upper() for k in kinase]
    exceptions.check_name_type(name_type)
    
    kin_info = get_kinase_info(kinase=kinase, name_type=name_type, info_file=info_file)
    
    return(kin_info[family_col])
    

def get_kinase(kinase, kin_type=None, mat_type='log2', aa=None, pos=None, pp=True, transpose=False):
    """
    Get kl.Kinase object by name.

    Parameters
    ----------
    kinase : str
        Kinase name.
    kin_type : str, optional
        Kinase type (ser_thr, tyrosine). The default is None.
    mat_type : str, optional
        Matrix type (raw, normalized, or log-scaled).
    aa : list, optional
        List of amino acids to select. The default is None.
    pos : list, optional
        Positions to select. The default is None.
    pp : bool, optional
        Phospho-priming columns. The default is True.
    transpose : bool, optional
        If True, matrix will be transposed. The default is False.

    Returns
    -------
    kin_obj : kl.Kinase
        Kinase object.
    """
    
    current_dir = os.path.dirname(__file__)
    mat_dir = os.path.join(current_dir, _global_vars.mat_dir)
    
    exceptions.check_kin_name(kinase, kin_type=kin_type)
    kinase = kinase.upper()
    if kin_type is None:
        kin_type = get_kinase_type(kinase)
    else:
        exceptions.check_kin_type(kin_type)
    exceptions.check_mat_type(mat_type)
        
    kin_matrix = get_matrix(kinase=kinase, kin_type=kin_type, mat_type=mat_type, aa=aa, pos=pos, pp=pp, transpose=transpose)
    random_aa_value = get_random_aa_value(kinase)
    if kin_type == 'ser_thr':
        st_fav = get_st_fav(kinase).to_dict('records')[0]
    family = get_kinase_family(kinase)
    kin_obj = core.Kinase(name=kinase, matrix=kin_matrix, random_aa_value=random_aa_value, kin_type=kin_type, mat_type=mat_type, pp=pp, phos_acc_fav=st_fav, family=family)
    
    return(kin_obj)


def get_families(kin_type):
    """
    Get families for kinase  type.

    Parameters
    ----------
    kin_type : str
        Kinase type (ser_thr, tyrosine).

    Returns
    -------
    families : list
        List of families.
    """
    
    exceptions.check_kin_type(kin_type)
    
    kinome_info = get_kinome_info(kin_type)
    families = list(kinome_info['FAMILY'].unique())
    
    return(families)
    

#%%
"""
##########################
# Scored phosphoproteome #
##########################
"""

def get_phosphoproteome(phosprot_name=None, kin_type=None):
    """
    Get phosphoproteome substrates for a kinase type.

    Parameters
    ----------
    phosprot_name : str, optional
        Phosphoproteomics database for calculating percentile-score. The default is defined in the global variables file.
    kin_type : str
        Kinase type (ser_thr, tyrosine).

    Returns
    -------
    phosprot : pd.DataFrame
        Data frame with the phosphoproteome data.
    """
    
    current_dir = os.path.dirname(__file__)
    phosprot_path = os.path.join(current_dir, _global_vars.phosprot_path)
    
    if phosprot_name is None:
        phosprot_name=_global_vars.phosprot_name
    
    phosprot_list = get_phosphoproteomes_list()
    if phosprot_name not in phosprot_list:
        raise ValueError(f'Phosphoproteome named \'{phosprot_name}\' was not found. Use kl.get_phosphoproteomes_list() to get a list of available phosphoproteomes.')
    
    if kin_type is not None:
        exceptions.check_kin_type(kin_type)
        phosprot = pd.read_csv(phosprot_path + '/' + phosprot_name + '/phosphoproteome_' + kin_type + '.txt', sep = '\t')
    else:
        phosprot = pd.read_csv(phosprot_path + '/' + phosprot_name + '/phosphoproteome.txt', sep = '\t')
    
    return(phosprot)


def get_scored_phosphoproteome(kin_type, phosprot_name=None,
                               file_type='parquet', with_info=False):
    """
    Get scored phosphoproteome substrates for a kinase type as pd.DataFrame.

    Parameters
    ----------
    kin_type : str
        Kinase type (ser_thr, tyrosine).
    phosprot_name : str, optional
        Phosphoproteomics database for calculating percentile-score. The default is defined in the global variables file.
    file_type : str, optional
        Scored phosphoproteome files type. The default is 'parquet'.
    with_info : bool, optional
        If True, phosphoproteome info will be added to the scores.

    Returns
    -------
    scored_phosprot : pd.DataFrame
        Data frame with the substrates as indices, and their corresponding scores in the columns.
    """
    
    current_dir = os.path.dirname(__file__)
    phosprot_path = os.path.join(current_dir, _global_vars.phosprot_path)
    
    if phosprot_name is None:
        phosprot_name=_global_vars.phosprot_name
    
    exceptions.check_kin_type(kin_type)
    if phosprot_name not in get_phosphoproteomes_list():
        raise ValueError(f'Phosphoproteome named \'{phosprot_name}\' was not found. Use kl.get_phosphoproteomes_list() to get a list of available phosphoproteomes.')
    exceptions.check_phosprot_file_type(file_type)
    
    all_scored_phosprot = core.ScoredPhosphoProteome(phosprot_name=phosprot_name, phosprot_path=phosprot_path, file_type=file_type)
    scored_phosprot = getattr(all_scored_phosprot, kin_type+'_scores')

    if with_info:
        phosprot_info = get_phosphoproteome(kin_type=kin_type, phosprot_name=phosprot_name)
        scored_phosprot = pd.merge(phosprot_info, scored_phosprot, left_on=_global_vars.default_seq_col, right_index=True, how='left')
    
    return(scored_phosprot)


def add_scored_phosphoproteome(phosprot_data, phosprot_name,
                               description=None, seq_col=None,
                               replace=False, new_seq_phos_res_cols=True,
                               **pps_args):
    
    current_dir = os.path.dirname(__file__)
    phosprot_path = os.path.join(current_dir, _global_vars.phosprot_path)
    files_path = os.path.join(phosprot_path, phosprot_name)
    
    if seq_col is None:
        seq_col = _global_vars.default_seq_col
        
    if phosprot_name in get_phosphoproteomes_list() and not replace:
        raise Exception(f'Phosphoproteome named \'{phosprot_name}\' already exists. Use replace=True to replace existing phosphoproteomes.')
    elif phosprot_name not in get_phosphoproteomes_list() and replace:
        raise Exception(f'Phosphoproteome named \'{phosprot_name}\' does not exist. Use replace=False to create a new phosphoproteome.')
    
    os.makedirs(files_path, exist_ok=True)
    os.makedirs(files_path+'/scored_phosprots', exist_ok=True)
    
    pps_data = phosphoproteomics.PhosphoProteomics(phosprot_data, seq_col=seq_col, new_seq_phos_res_cols=new_seq_phos_res_cols, **pps_args)
    ser_thr_scored_phosprot = pps_data.score('ser_thr', values_only=True)
    tyrosine_scored_phosprot = pps_data.score('tyrosine', values_only=True, non_canonical=True)
    
    print('Writing files...')
    pps_data.data.to_csv(files_path+'/phosphoproteome.txt', sep = '\t', index=False)
    pps_data.ser_thr_data.to_csv(files_path+'/phosphoproteome_ser_thr.txt', sep = '\t', index=False)
    pps_data.tyrosine_data.to_csv(files_path+'/phosphoproteome_tyrosine.txt', sep = '\t', index=False)
    ser_thr_scored_phosprot.to_parquet(files_path+'/scored_phosprots/ser_thr_phosphoproteome_scored.parquet')
    tyrosine_scored_phosprot.to_parquet(files_path+'/scored_phosprots/tyrosine_phosphoproteome_scored.parquet')
    print('Complete.')

    all_phosprot_info = get_phosphoproteomes_info()
    if replace:
        phosprot_info = get_phosphoproteomes_info(phosprot_name)
        all_phosprot_info[all_phosprot_info[_global_vars.phosprot_info_columns['name']] == phosprot_name] = [phosprot_info[_global_vars.phosprot_info_columns['date_added']], datetime.date.today().strftime("%Y-%m-%d"), phosprot_name, description, len(pps_data.ser_thr_data), len(pps_data.tyrosine_data)]
    else:
        all_phosprot_info.loc[len(all_phosprot_info.index)] = [datetime.date.today().strftime("%Y-%m-%d"), datetime.date.today().strftime("%Y-%m-%d"), phosprot_name, description, len(pps_data.ser_thr_data), len(pps_data.tyrosine_data)]
    all_phosprot_info.to_csv(phosprot_path+'/scored_phosphoproteomes_data.txt', sep = '\t', index=False)


def remove_scored_phosphoproteome(phosprot_name):
    
    exceptions.check_phosprot_name(phosprot_name)
    current_dir = os.path.dirname(__file__)
    phosprot_path = os.path.join(current_dir, _global_vars.phosprot_path)
    
    while True:
        answer = str(input(f'Delete \'{phosprot_name}\' phosphoproteome? (y/n): '))
        if answer in ('y', 'n'):
            break
        print('Invalid input.')
    if answer == 'y':
        files_path = phosprot_path+'/'+phosprot_name
        print('Deleting files...')
        shutil.rmtree(files_path)
        phosprot_info = pd.read_csv(phosprot_path+'/scored_phosphoproteomes_data.txt', sep = '\t')
        phosprot_info = phosprot_info[phosprot_info['Name'] != phosprot_name]
        phosprot_info.to_csv(phosprot_path+'/scored_phosphoproteomes_data.txt', sep = '\t', index=False)
        print('Complete.')
    elif answer == 'n':
        print('\nProcess cancelled.')
    

def get_phosphoproteomes_list():
    
    current_dir = os.path.dirname(__file__)
    phosprot_path = os.path.join(current_dir, _global_vars.phosprot_path)
    
    phosprot_info = get_phosphoproteomes_info()
    return(phosprot_info['Name'].to_list())


def get_phosphoproteomes_info(phosprot_name=None):
    
    current_dir = os.path.dirname(__file__)
    phosprot_path = os.path.join(current_dir, _global_vars.phosprot_path)
    
    phosprot_info = pd.read_csv(phosprot_path + '/scored_phosphoproteomes_data.txt', sep = '\t')
    
    if phosprot_name is None:
        return(phosprot_info)
    else:
        if phosprot_name not in get_phosphoproteomes_list():
            raise ValueError(f'Phosphoproteome named \'{phosprot_name}\' was not found. Use kl.get_phosphoproteomes_list() to get a list of available phosphoproteomes.')
        return(phosprot_info[phosprot_info['Name'] == phosprot_name].squeeze())


def update_scored_phosphoproteome(phosprot_name=None):
    
    if phosprot_name is None:
        phosprot_name = get_phosphoproteomes_list()
    else:
        if isinstance(phosprot_name, str):
            phosprot_name = [phosprot_name]
    
    for ppt in phosprot_name:
        print('\nUpdating phosphoproteome: ' + str(ppt))
        phosprot_data = get_phosphoproteome(phosprot_name=ppt)
        phosprot_info = get_phosphoproteomes_info(phosprot_name=ppt)
        description = phosprot_info[_global_vars.phosprot_info_columns['description']]
        add_scored_phosphoproteome(phosprot_data=phosprot_data, phosprot_name=ppt, description=description, seq_col=_global_vars.default_seq_col, replace=True, new_seq_phos_res_cols=False)


def get_current_phosphoproteome():
    """
    Printing the current phosphoproteome for percentile calculation.

    Parameters
    ----------
    None.

    Returns
    -------
    Print current phosphoproteome name.

    """
    
    print(f'Current phosphoproteome is set to: {_global_vars.phosprot_name}')


def set_current_phosphoproteome(phosprot_name):
    """
    Setting phosphoproteome for percentile calculation.

    Parameters
    ----------
    phosprot_name : str
        Phosphoproteome name.

    Returns
    -------
    None.

    """
    
    exceptions.check_phosprot_name(phosprot_name)
    _global_vars.phosprot_name = phosprot_name
    print(f'Phosphoproteome was set to: {_global_vars.phosprot_name}')


def reset_current_phosphoproteome():
    """
    Reset phosphoproteome to default.

    Parameters
    ----------
    None.

    Returns
    -------
    None.

    """
    
    _global_vars.reset_phosprot_name()
    print(f'Phosphoproteome was reset to: {_global_vars.phosprot_name}')

