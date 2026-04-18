"""
############################################
# The Kinase Library - Matrix Processing   #
############################################
"""

import os
import types
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils import _global_vars, exceptions
from ..modules import data
from ..objects import core


#%%
"""
######################
# Internal Utilities #
######################
"""

def _get_aa_labels(kin_type, k_mod=False):
    """
    Get amino acid labels for the densitometry-to-raw step (before 's' insertion).
    """
    aa_labels = ['P','G','A','C','S','T','V','I','L','M','F','Y','W','H','K','R','Q','N','D','E','t','y']
    if kin_type == 'tyrosine' and k_mod:
        aa_labels += ['kac','kmet']
    return aa_labels


def _get_full_aa_labels(kin_type, k_mod=False):
    """
    Get full amino acid labels (with 's' column inserted).
    """
    full_aa_labels = ['P','G','A','C','S','T','V','I','L','M','F','Y','W','H','K','R','Q','N','D','E','s','t','y']
    if kin_type == 'tyrosine' and k_mod:
        full_aa_labels += ['kac','kmet']
    return full_aa_labels


def _get_positions(kin_type):
    """
    Get position indices for the given kinase type.
    """
    if kin_type == 'ser_thr':
        return _global_vars.ser_thr_pos
    elif kin_type == 'tyrosine':
        return _global_vars.tyrosine_pos


#%%
"""
#####################
# Core Transforms   #
#####################
"""

def densitometry_to_raw(densitometry, kin_type, k_mod=False):
    """
    Convert a densitometry matrix to a raw matrix.

    Performs transposition, row inversion, slicing to relevant positions
    and amino acids, and inserts the 's' column as a copy of 't'.

    Parameters
    ----------
    densitometry : pd.DataFrame or str
        Densitometry matrix (rows=amino acid letters A-X, cols=position numbers 1-12/14),
        or a file path to a densitometry TSV file.
    kin_type : str
        Kinase type ('ser_thr' or 'tyrosine').
    k_mod : bool, optional
        Include lysine modifications (kac, kmet) for tyrosine kinases.
        The default is False.

    Returns
    -------
    raw_matrix : pd.DataFrame
        Raw matrix with positions as index and amino acids as columns.
    """
    exceptions.check_kin_type(kin_type)

    if isinstance(densitometry, str):
        densitometry = pd.read_csv(densitometry, sep='\t', index_col=0)

    aa_labels = _get_aa_labels(kin_type, k_mod)
    positions = _get_positions(kin_type)

    raw_matrix = densitometry.transpose()
    raw_matrix = raw_matrix.loc[:, ::-1]
    raw_matrix = raw_matrix.iloc[:len(positions), :len(aa_labels)]
    raw_matrix = raw_matrix.apply(pd.to_numeric)
    raw_matrix.columns = aa_labels
    raw_matrix.index = positions
    raw_matrix.insert(list(raw_matrix.columns).index('t'), 's', raw_matrix['t'])

    return raw_matrix


def raw_to_normalized(raw_matrix, kin_type, lib_type, dual_spec=False):
    """
    Normalize a raw matrix.

    Performs row normalization by randomized amino acid sums,
    phosphoacceptor correction, and cysteine correction.

    Parameters
    ----------
    raw_matrix : pd.DataFrame
        Raw matrix from densitometry_to_raw.
    kin_type : str
        Kinase type ('ser_thr' or 'tyrosine').
    lib_type : str
        Library type ('ser_thr', 'tyr', or 'ser_thr_tyr').
    dual_spec : bool, optional
        Treat kinase as dual specific. The default is False.

    Returns
    -------
    norm_matrix : pd.DataFrame
        Normalized matrix.
    """
    exceptions.check_kin_type(kin_type)
    exceptions.check_lib_type(lib_type)

    random_aa_labels = _global_vars.random_aa_dict[lib_type]

    norm_matrix = raw_matrix.divide(raw_matrix[random_aa_labels].sum(axis=1), axis=0)

    if kin_type == 'ser_thr' or dual_spec:
        norm_matrix['S'] = norm_matrix[random_aa_labels].median(axis=1)
        norm_matrix['T'] = norm_matrix[random_aa_labels].median(axis=1)
    if kin_type == 'tyrosine' or dual_spec:
        norm_matrix['Y'] = norm_matrix['F']

    norm_matrix['C'] = norm_matrix['C'] / norm_matrix['C'].median() * 1/len(random_aa_labels)

    return norm_matrix


def normalized_to_log2(norm_matrix, lib_type):
    """
    Convert a normalized matrix to log2 scale.

    Multiplies by the number of randomized amino acids and applies
    log2 transformation.

    Parameters
    ----------
    norm_matrix : pd.DataFrame
        Normalized matrix from raw_to_normalized.
    lib_type : str
        Library type ('ser_thr', 'tyr', or 'ser_thr_tyr').

    Returns
    -------
    log2_matrix : pd.DataFrame
        Log2-scaled matrix.
    """
    exceptions.check_lib_type(lib_type)

    random_aa_labels = _global_vars.random_aa_dict[lib_type]
    log2_matrix = np.log2(norm_matrix * len(random_aa_labels))

    return log2_matrix


#%%
"""
#####################
# S/T Favorability  #
#####################
"""

def compute_st_favorability(raw_matrix):
    """
    Compute serine/threonine favorability from a raw matrix.

    Parameters
    ----------
    raw_matrix : pd.DataFrame
        Raw matrix with 'S' and 'T' columns.

    Returns
    -------
    st_fav : pd.Series
        S/T favorability scores with index ['s', 't'].
    """
    s_sum = raw_matrix['S'].sum()
    t_sum = raw_matrix['T'].sum()

    s_ctrl = 0.75 * s_sum - 0.25 * t_sum
    t_ctrl = 0.75 * t_sum - 0.25 * s_sum

    s_fav = s_ctrl / max(s_ctrl, t_ctrl)
    t_fav = t_ctrl / max(s_ctrl, t_ctrl)

    return pd.Series([s_fav, t_fav], index=['s', 't'])


#%%
"""
######################
# Combined Matrices  #
######################
"""

def build_combined_matrices(kin_type, k_mod=False, round_digits=4):
    """
    Build combined matrices from all individual matrices in the database.

    Reads all individual raw, normalized, and log2 matrices for the given
    kinase type, flattens them, and assembles combined DataFrames.

    Parameters
    ----------
    kin_type : str
        Kinase type ('ser_thr' or 'tyrosine').
    k_mod : bool, optional
        Include lysine modifications. The default is False.
    round_digits : int, optional
        Number of decimal digits for rounding. The default is 4.

    Returns
    -------
    combined : dict
        Dictionary with keys 'raw', 'norm', 'log2', and optionally
        'st_favorability' (for ser_thr only), each containing a pd.DataFrame.
    """
    exceptions.check_kin_type(kin_type)

    current_dir = os.path.dirname(__file__)
    mat_dir = os.path.join(current_dir, _global_vars.mat_dir, kin_type)

    full_aa_labels = _get_full_aa_labels(kin_type, k_mod)
    positions = _get_positions(kin_type)
    combined_matrix_columns = [str(x) + str(y) for x in positions for y in full_aa_labels]

    densitometry_dir = os.path.join(mat_dir, 'densitometry')
    all_kinases = np.sort([
        '_'.join(name.split('.')[0].split('_')[:-1])
        for name in os.listdir(densitometry_dir)
        if not name.startswith(".")
    ])

    full_matrix_raw = pd.DataFrame(index=all_kinases, columns=combined_matrix_columns)
    full_matrix_norm = pd.DataFrame(index=all_kinases, columns=combined_matrix_columns)
    full_matrix_scaled = pd.DataFrame(index=all_kinases, columns=combined_matrix_columns)

    if kin_type == 'ser_thr':
        full_st_fav = pd.DataFrame(index=all_kinases, columns=['s', 't'])

    for kinase in all_kinases:
        raw_matrix = pd.read_csv(os.path.join(mat_dir, 'raw', kinase + '.tsv'), sep='\t', index_col=0)
        norm_matrix = pd.read_csv(os.path.join(mat_dir, 'norm', kinase + '.tsv'), sep='\t', index_col=0)
        log2_matrix = pd.read_csv(os.path.join(mat_dir, 'log2', kinase + '.tsv'), sep='\t', index_col=0)

        full_matrix_raw.loc[kinase] = raw_matrix.values.reshape(len(positions) * len(full_aa_labels))
        full_matrix_norm.loc[kinase] = norm_matrix.values.reshape(len(positions) * len(full_aa_labels))
        full_matrix_scaled.loc[kinase] = log2_matrix.values.reshape(len(positions) * len(full_aa_labels))

        if kin_type == 'ser_thr':
            full_st_fav.loc[kinase] = compute_st_favorability(raw_matrix).values

    combined = {
        'raw': full_matrix_raw.astype(np.float64).round(round_digits),
        'norm': full_matrix_norm.astype(np.float64).round(round_digits),
        'log2': full_matrix_scaled.astype(np.float64).round(round_digits),
    }

    if kin_type == 'ser_thr':
        combined['st_favorability'] = full_st_fav.astype(np.float64).round(round_digits)

    return combined


#%%
"""
############################
# High-Level Processing    #
############################
"""

def process_kinase(densitometry, kinase_name, kin_type, lib_type,
                   dual_spec=False, k_mod=False, round_digits=4):
    """
    Process a single kinase through the full matrix processing pipeline.

    Converts a densitometry matrix to raw, normalized, and log2-scaled
    matrices in a single call.

    Parameters
    ----------
    densitometry : pd.DataFrame or str
        Densitometry matrix or file path to a densitometry TSV file.
    kinase_name : str
        Kinase name.
    kin_type : str
        Kinase type ('ser_thr' or 'tyrosine').
    lib_type : str
        Library type ('ser_thr', 'tyr', or 'ser_thr_tyr').
    dual_spec : bool, optional
        Treat kinase as dual specific. The default is False.
    k_mod : bool, optional
        Include lysine modifications. The default is False.
    round_digits : int, optional
        Number of decimal digits for rounding. The default is 4.

    Returns
    -------
    result : dict
        Dictionary with keys 'raw', 'norm', 'log2' (pd.DataFrames),
        and 'st_favorability' (pd.Series, for ser_thr kinases only).
    """
    exceptions.check_kin_type(kin_type)
    exceptions.check_lib_type(lib_type)

    raw_matrix = densitometry_to_raw(densitometry, kin_type, k_mod=k_mod)
    norm_matrix = raw_to_normalized(raw_matrix, kin_type, lib_type, dual_spec=dual_spec)
    log2_matrix = normalized_to_log2(norm_matrix, lib_type)

    result = {
        'raw': raw_matrix.round(round_digits),
        'norm': norm_matrix.round(round_digits),
        'log2': log2_matrix.round(round_digits),
    }

    if kin_type == 'ser_thr':
        result['st_favorability'] = compute_st_favorability(raw_matrix)

    return result


def process_kinases(kin_type, lib_type, kinase_names=None, dual_spec=False,
                    k_mod=False, round_digits=4, densitometry_dir=None,
                    densitometry_data=None):
    """
    Process multiple kinases through the full matrix processing pipeline.

    Parameters
    ----------
    kin_type : str
        Kinase type ('ser_thr' or 'tyrosine').
    lib_type : str
        Library type ('ser_thr', 'tyr', or 'ser_thr_tyr').
    kinase_names : list, optional
        List of kinase names to process. If None, auto-discovers from the
        densitometry directory or densitometry_data keys. The default is None.
    dual_spec : bool, optional
        Treat kinases as dual specific. The default is False.
    k_mod : bool, optional
        Include lysine modifications. The default is False.
    round_digits : int, optional
        Number of decimal digits for rounding. The default is 4.
    densitometry_dir : str, optional
        Path to directory containing densitometry files. If None and
        densitometry_data is None, uses the package database directory.
        The default is None.
    densitometry_data : dict, optional
        Dictionary mapping kinase names to densitometry DataFrames.
        Alternative to densitometry_dir. The default is None.

    Returns
    -------
    results : dict
        Dictionary mapping kinase names to result dicts from process_kinase.
    """
    exceptions.check_kin_type(kin_type)
    exceptions.check_lib_type(lib_type)

    if densitometry_data is not None:
        # Input from dict of DataFrames
        if kinase_names is None:
            kinase_names = sorted(densitometry_data.keys())

        results = {}
        for kinase in tqdm(kinase_names):
            if kinase not in densitometry_data:
                print(f'ERROR: the kinase \'{kinase}\' was not found in densitometry_data')
                continue
            results[kinase] = process_kinase(
                densitometry_data[kinase], kinase, kin_type, lib_type,
                dual_spec=dual_spec, k_mod=k_mod, round_digits=round_digits
            )
        return results

    # Input from directory
    if densitometry_dir is None:
        current_dir = os.path.dirname(__file__)
        densitometry_dir = os.path.join(current_dir, _global_vars.mat_dir, kin_type, 'densitometry')

    if densitometry_dir[-1] != '/':
        densitometry_dir = densitometry_dir + '/'

    kin_list = [
        '_'.join(name.split('.')[0].split('_')[:-1])
        for name in os.listdir(densitometry_dir)
        if not name.startswith(".")
    ]

    if kinase_names is None:
        kinase_names = np.sort(kin_list)

    results = {}
    for kinase in tqdm(kinase_names):
        if kinase not in kin_list:
            print(f'ERROR: the kinase \'{kinase}\' was not found in the folder')
            continue
        kinase_file = [
            x for x in os.listdir(densitometry_dir)
            if '_'.join(x.split('.')[0].split('_')[:-1]) == kinase
        ][0]
        results[kinase] = process_kinase(
            densitometry_dir + kinase_file, kinase, kin_type, lib_type,
            dual_spec=dual_spec, k_mod=k_mod, round_digits=round_digits
        )

    return results


def save_matrices_to_database(processed_results, kin_type, round_digits=4,
                              output_dir=None, rebuild_combined=True,
                              update_phosphoproteome=False):
    """
    Save processed matrices to the database.

    Writes individual raw, normalized, and log2 matrix files, optionally
    rebuilds combined matrix files, and optionally updates the scored
    phosphoproteomes.

    Parameters
    ----------
    processed_results : dict
        Dictionary mapping kinase names to result dicts (output of
        process_kinase or process_kinases).
    kin_type : str
        Kinase type ('ser_thr' or 'tyrosine').
    round_digits : int, optional
        Number of decimal digits for rounding. The default is 4.
    output_dir : str, optional
        Custom output directory. If None, writes to the package's internal
        database. The default is None.
    rebuild_combined : bool, optional
        Rebuild combined matrix files after saving. The default is True.
    update_phosphoproteome : bool, optional
        Update scored phosphoproteomes after saving. The default is False.

    Returns
    -------
    None.
    """
    exceptions.check_kin_type(kin_type)

    if output_dir is None:
        current_dir = os.path.dirname(__file__)
        base_dir = os.path.join(current_dir, _global_vars.mat_dir, kin_type)
    else:
        base_dir = output_dir

    raw_dir = os.path.join(base_dir, 'raw')
    norm_dir = os.path.join(base_dir, 'norm')
    log2_dir = os.path.join(base_dir, 'log2')

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(norm_dir, exist_ok=True)
    os.makedirs(log2_dir, exist_ok=True)

    for kinase, result in processed_results.items():
        result['raw'].round(round_digits).to_csv(
            os.path.join(raw_dir, kinase + '.tsv'), header=True, index=True, sep='\t'
        )
        result['norm'].round(round_digits).to_csv(
            os.path.join(norm_dir, kinase + '.tsv'), sep='\t'
        )
        result['log2'].round(round_digits).to_csv(
            os.path.join(log2_dir, kinase + '.tsv'), sep='\t'
        )

    print(f'Saved {len(processed_results)} kinase matrices to {base_dir}')

    if rebuild_combined:
        if output_dir is not None:
            print('Skipping combined matrix rebuild (only supported for package database).')
        else:
            print('Rebuilding combined matrices...')
            combined = build_combined_matrices(kin_type, round_digits=round_digits)
            combined['raw'].to_csv(
                os.path.join(base_dir, kin_type + '_all_raw_matrices.tsv'),
                header=True, index=True, sep='\t'
            )
            combined['norm'].to_csv(
                os.path.join(base_dir, kin_type + '_all_norm_matrices.tsv'),
                header=True, index=True, sep='\t'
            )
            combined['log2'].to_csv(
                os.path.join(base_dir, kin_type + '_all_log2_matrices.tsv'),
                header=True, index=True, sep='\t'
            )
            if kin_type == 'ser_thr':
                combined['st_favorability'].to_csv(
                    os.path.join(base_dir, 'st_favorability.tsv'),
                    header=True, index=True, sep='\t'
                )
            print('Combined matrices saved.')

    if update_phosphoproteome:
        print('Updating scored phosphoproteomes...')
        data.update_scored_phosphoproteome()

    _print_database_update_reminder(update_phosphoproteome=update_phosphoproteome)


def _print_database_update_reminder(update_phosphoproteome=False):
    """
    Print a prominent reminder of manual steps after updating the database.
    """
    banner = '#' * 70
    lines = [
        '',
        banner,
        '# REMEMBER: manual steps to complete the database update'.ljust(69) + '#',
        banner,
    ]
    step = 1
    if not update_phosphoproteome:
        lines.append(f'# {step}. Update scored phosphoproteome:'.ljust(69) + '#')
        lines.append('#    >>> kl.update_scored_phosphoproteome()'.ljust(69) + '#')
        step += 1
    lines += [
        f'# {step}. Update databases/kinase_data/kinome_information.tsv'.ljust(69) + '#',
        f'# {step + 1}. Update test files in src/tests/ (if applicable)'.ljust(69) + '#',
        f'# {step + 2}. Update CHANGELOG.md'.ljust(69) + '#',
        f'# {step + 3}. Update README.md (kinase counts, new features)'.ljust(69) + '#',
        f'# {step + 4}. Bump __version__ in src/kinase_library/__init__.py'.ljust(69) + '#',
        banner,
        '',
    ]
    print('\n'.join(lines))


#%%
"""
############################
# Kinase Object API        #
############################
"""

def kinase_from_densitometry(densitometry, name, kin_type, lib_type,
                             dual_spec=False, k_mod=False,
                             family='undefined', round_digits=4,
                             update_database=False,
                             rebuild_combined=True,
                             update_phosphoproteome=False):
    """
    Read a densitometry matrix into a kl.Kinase object.

    Processes the densitometry through the full pipeline (raw, norm, log2)
    and returns a kl.Kinase object with all standard attributes and methods
    (plot_data, seq_logo, heatmap, score, percentile, etc.). The raw matrix
    is attached as an extra attribute, and a save_to_database() method is
    bound to the instance for optional persistence.

    Parameters
    ----------
    densitometry : pd.DataFrame or str
        Densitometry matrix or file path to a densitometry TSV file.
    name : str
        Kinase name.
    kin_type : str
        Kinase type ('ser_thr' or 'tyrosine').
    lib_type : str
        Library type ('ser_thr', 'tyr', or 'ser_thr_tyr').
    dual_spec : bool, optional
        Treat kinase as dual specific. The default is False.
    k_mod : bool, optional
        Include lysine modifications. The default is False.
    family : str, optional
        Kinase family. The default is 'undefined'.
    round_digits : int, optional
        Number of decimal digits for rounding. The default is 4.
    update_database : bool, optional
        If True, save the processed matrices to the package database
        immediately after creation. The default is False.
    rebuild_combined : bool, optional
        When update_database is True, rebuild combined matrix files.
        The default is True.
    update_phosphoproteome : bool, optional
        When update_database is True, re-score phosphoproteomes.
        The default is False.

    Returns
    -------
    kin : kl.Kinase
        Kinase object with raw_matrix attached and save_to_database() bound.
    """
    exceptions.check_kin_type(kin_type)
    exceptions.check_lib_type(lib_type)

    result = process_kinase(
        densitometry, kinase_name=name, kin_type=kin_type, lib_type=lib_type,
        dual_spec=dual_spec, k_mod=k_mod, round_digits=round_digits
    )

    # Kinase constructor expects amino acids as rows and positions as columns.
    # process_kinase returns positions as rows, so transpose.
    norm_for_kinase = result['norm'].transpose()

    random_aa_value = _global_vars.random_aa_value[lib_type]

    if kin_type == 'ser_thr':
        st_fav = result['st_favorability']
        phos_acc_fav = {'S': float(st_fav['s']), 'T': float(st_fav['t'])}
    else:
        phos_acc_fav = {'Y': 1.0}

    kin = core.Kinase(
        name=name, matrix=norm_for_kinase, random_aa_value=random_aa_value,
        mat_type='norm', kin_type=kin_type, family=family,
        pp=True, k_mod=k_mod, phos_acc_fav=phos_acc_fav
    )

    # Attach extras
    kin.raw_matrix = result['raw'].transpose()
    if kin_type == 'ser_thr':
        kin.st_fav_series = result['st_favorability']

    # Store processing context for later re-save
    kin._lib_type = lib_type
    kin._dual_spec = dual_spec
    kin._round_digits = round_digits

    # Bind save_to_database method to the instance
    def save_to_database(self, rebuild_combined=True, update_phosphoproteome=False,
                         output_dir=None):
        save_kinase_to_database(
            self, rebuild_combined=rebuild_combined,
            update_phosphoproteome=update_phosphoproteome,
            output_dir=output_dir, round_digits=self._round_digits
        )
    kin.save_to_database = types.MethodType(save_to_database, kin)

    if update_database:
        kin.save_to_database(
            rebuild_combined=rebuild_combined,
            update_phosphoproteome=update_phosphoproteome
        )

    return kin


def save_kinase_to_database(kinase_obj, rebuild_combined=True,
                            update_phosphoproteome=False,
                            output_dir=None, round_digits=4):
    """
    Save a Kinase object's matrices to the package database.

    Parameters
    ----------
    kinase_obj : kl.Kinase
        Kinase object (must have raw_matrix attribute, as created by
        kinase_from_densitometry).
    rebuild_combined : bool, optional
        Rebuild combined matrix files after saving. The default is True.
    update_phosphoproteome : bool, optional
        Re-score phosphoproteomes after saving. The default is False.
    output_dir : str, optional
        Custom output directory. If None, writes to the package database.
        The default is None.
    round_digits : int, optional
        Number of decimal digits for rounding. The default is 4.

    Returns
    -------
    None.
    """
    if not hasattr(kinase_obj, 'raw_matrix'):
        raise AttributeError(
            'Kinase object has no raw_matrix attribute. '
            'Use kinase_from_densitometry() to create a savable Kinase object.'
        )

    # Matrices are saved with positions as rows and amino acids as columns
    # (transpose back from Kinase convention)
    result = {
        'raw': kinase_obj.raw_matrix.transpose(),
        'norm': kinase_obj.norm_matrix.transpose(),
        'log2': kinase_obj.log2_matrix.transpose(),
    }
    if kinase_obj.kin_type == 'ser_thr' and hasattr(kinase_obj, 'st_fav_series'):
        result['st_favorability'] = kinase_obj.st_fav_series

    save_matrices_to_database(
        {kinase_obj.name: result},
        kin_type=kinase_obj.kin_type,
        round_digits=round_digits,
        output_dir=output_dir,
        rebuild_combined=rebuild_combined,
        update_phosphoproteome=update_phosphoproteome
    )
