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

    # Enumerate from raw/ — that's the canonical source for the combined
    # matrices (every kinase in the database has its raw matrix written there)
    raw_dir = os.path.join(mat_dir, 'raw')
    all_kinases = np.sort([
        name.split('.')[0]
        for name in os.listdir(raw_dir)
        if not name.startswith('.') and name.endswith('.tsv')
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


_KINOME_INFO_COLUMNS = [
    'MATRIX_NAME', 'KINASE', 'GENE_NAME', 'TYPE', 'SUBTYPE', 'FAMILY',
    'UNIPROT_ID', 'UNIPROT_ENTRY_NAME', 'PDB_ID', 'KL_LIBRARY',
    'DUAL_SPECIFICITY', 'DISPLAY_NAME'
]


def _write_individual_matrices(processed_results, output_dir, round_digits):
    """Write raw/norm/log2 TSV files for each kinase under output_dir."""
    raw_dir = os.path.join(output_dir, 'raw')
    norm_dir = os.path.join(output_dir, 'norm')
    log2_dir = os.path.join(output_dir, 'log2')

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

    print(f'Saved {len(processed_results)} kinase matrices to {output_dir}')


def _write_combined_matrices(kin_type, base_dir, round_digits):
    """Rebuild and write the combined *_all_*.tsv files for the package database."""
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


def _confirm(message):
    """Prompt y/n. Returns True on yes, False on no."""
    while True:
        ans = input(f'{message} (y/n): ').strip().lower()
        if ans in ('y', 'yes'):
            return True
        if ans in ('n', 'no'):
            return False
        print("Please enter 'y' or 'n'.")


def _kinase_already_in_kinome_info(name):
    """Return True if the kinase has a row in kinome_information.tsv."""
    try:
        all_info = data.get_kinome_info()
        return name in all_info['MATRIX_NAME'].astype(str).tolist()
    except Exception:
        return False


def _prompt_field(label, default=None):
    """Prompt for a field; pressing enter accepts the default if given."""
    if default is not None:
        prompt = f'  {label} [{default}]: '
    else:
        prompt = f'  {label}: '
    val = input(prompt).strip()
    if val == '' and default is not None:
        return default
    return val


def _prompt_kinome_info_for_kinase(name, kin_type, lib_type, dual_spec, family=None):
    """
    Prompt the user for kinome_information columns for a single kinase.

    Returns a dict of column->value, or None if the user declined to update
    an existing row.
    """
    if _kinase_already_in_kinome_info(name):
        if not _confirm(f"Kinase '{name}' already exists in kinome_information.tsv. Update its row?"):
            return None

    print(f"Filling kinome_information.tsv row for '{name}':")

    kinase_val = _prompt_field('KINASE', default=name)
    gene_val = _prompt_field('GENE_NAME', default=name)
    subtype_val = _prompt_field('SUBTYPE (e.g. STK, RTK, ncTK)')
    family_val = _prompt_field('FAMILY', default=family) if family else _prompt_field('FAMILY')
    uniprot_id = _prompt_field('UNIPROT_ID')
    uniprot_entry = _prompt_field('UNIPROT_ENTRY_NAME')
    pdb_id = _prompt_field('PDB_ID (optional, blank to skip)', default='')
    display_name = _prompt_field('DISPLAY_NAME', default=name)

    return {
        'MATRIX_NAME': name,
        'KINASE': kinase_val,
        'GENE_NAME': gene_val,
        'TYPE': kin_type,
        'SUBTYPE': subtype_val,
        'FAMILY': family_val,
        'UNIPROT_ID': uniprot_id,
        'UNIPROT_ENTRY_NAME': uniprot_entry,
        'PDB_ID': pdb_id,
        'KL_LIBRARY': lib_type,
        'DUAL_SPECIFICITY': bool(dual_spec),
        'DISPLAY_NAME': display_name,
    }


def _resolve_kinome_info_rows(kinome_info, kinase_metadata, confirm):
    """
    Resolve the kinome_info rows to apply.

    Parameters
    ----------
    kinome_info : str, pd.DataFrame, dict, or None
        - str: path to a TSV/CSV file with one row per kinase
        - pd.DataFrame: rows already prepared
        - dict: {matrix_name: {column: value, ...}}
        - None: prompt interactively (requires confirm=True)
    kinase_metadata : dict
        {name: (kin_type, lib_type, dual_spec, family)} for each kinase being saved.
    confirm : bool
        If False and kinome_info is None, raise an error.

    Returns
    -------
    pd.DataFrame or None
        DataFrame ready for data.update_kinome_info, or None if the user
        declined all updates.
    """
    if kinome_info is None:
        if not confirm:
            raise ValueError(
                'kinome_info must be provided when confirm=False and '
                'update_database=True (no interactive prompts available).'
            )
        rows = []
        for name, (kin_type, lib_type, dual_spec, family) in kinase_metadata.items():
            row = _prompt_kinome_info_for_kinase(name, kin_type, lib_type, dual_spec, family)
            if row is not None:
                rows.append(row)
        if not rows:
            return None
        return pd.DataFrame(rows)

    if isinstance(kinome_info, str):
        sep = '\t' if kinome_info.endswith('.tsv') else ','
        return pd.read_csv(kinome_info, sep=sep)
    if isinstance(kinome_info, dict):
        df = pd.DataFrame.from_dict(kinome_info, orient='index')
        df.index.name = 'MATRIX_NAME'
        return df.reset_index()
    if isinstance(kinome_info, pd.DataFrame):
        return kinome_info.copy()
    raise TypeError(f'Unsupported type for kinome_info: {type(kinome_info)}')


def save_matrices(processed_results, kin_type, output_dir=None,
                  update_database=False, confirm=True, kinome_info=None,
                  kinase_metadata=None, round_digits=4):
    """
    Save processed matrices to a directory and/or update the package database.

    Two independent save targets:
      * `output_dir`  - write matrices to this directory only
      * `update_database` - perform full package update (matrices + combined +
        kinome_information.tsv + phosphoproteomes), with confirmation prompt

    Both can be set in the same call.

    Parameters
    ----------
    processed_results : dict
        {kinase_name: result_dict} from process_kinase / process_kinases.
    kin_type : str
        Kinase type ('ser_thr' or 'tyrosine').
    output_dir : str, optional
        Directory to write matrices to. The default is None.
    update_database : bool, optional
        If True, perform the full package update. The default is False.
    confirm : bool, optional
        If True, prompt before updating the package database. The default is True.
    kinome_info : str, dict, or pd.DataFrame, optional
        Path to a TSV/CSV file, dict, or DataFrame with kinome_information rows
        for the kinases being saved. Only used when update_database is True.
        If None and confirm=True, fields are prompted interactively. If None and
        confirm=False, an error is raised. The default is None.
    kinase_metadata : dict, optional
        {name: (kin_type, lib_type, dual_spec, family)} for each kinase. Used
        to drive the kinome_info prompts. Defaults inferred from kin_type and
        ('ser_thr', False, None) if not provided.
    round_digits : int, optional
        Number of decimal digits for rounding. The default is 4.

    Returns
    -------
    None.
    """
    exceptions.check_kin_type(kin_type)

    if not output_dir and not update_database:
        print('Nothing to do. Pass output_dir or update_database=True.')
        return

    # Path-only save (independent of update_database)
    if output_dir:
        _write_individual_matrices(processed_results, output_dir, round_digits)

    if not update_database:
        return

    # Confirmation prompt for the package update
    if confirm:
        names = list(processed_results.keys())
        if len(names) <= 5:
            label = ', '.join(f"'{n}'" for n in names)
        else:
            label = f"{len(names)} kinases ({', '.join(names[:3])}, ...)"
        if not _confirm(f'Update package database with {label}?'):
            print('Database update cancelled.')
            return

    # Resolve kinome_info rows BEFORE writing anything to the database
    if kinase_metadata is None:
        kinase_metadata = {n: (kin_type, kin_type, False, None) for n in processed_results}
    kinome_rows = _resolve_kinome_info_rows(kinome_info, kinase_metadata, confirm)

    # Write to package
    current_dir = os.path.dirname(__file__)
    base_dir = os.path.join(current_dir, _global_vars.mat_dir, kin_type)
    _write_individual_matrices(processed_results, base_dir, round_digits)

    # Save densitometry source files to the package, where available, so the
    # kinase can be re-processed in the future from the package alone.
    densitometry_dir = os.path.join(base_dir, 'densitometry')
    os.makedirs(densitometry_dir, exist_ok=True)
    for kinase, result in processed_results.items():
        if 'densitometry' in result and result['densitometry'] is not None:
            result['densitometry'].to_csv(
                os.path.join(densitometry_dir, kinase + '_densitometry.txt'),
                sep='\t', header=True, index=True
            )

    _write_combined_matrices(kin_type, base_dir, round_digits)

    if kinome_rows is not None and len(kinome_rows) > 0:
        print('Updating kinome_information.tsv...')
        data.update_kinome_info(kinome_rows)

    print('Updating scored phosphoproteomes...')
    data.update_scored_phosphoproteome()

    _print_database_update_reminder()


def _print_database_update_reminder():
    """Print a prominent reminder of remaining manual steps."""
    banner = '#' * 70
    lines = [
        '',
        banner,
        '# REMEMBER: manual steps to complete the database update'.ljust(69) + '#',
        banner,
        '# 1. Update CHANGELOG.md'.ljust(69) + '#',
        '# 2. Update README.md (kinase counts, new features)'.ljust(69) + '#',
        '# 3. Bump __version__ in src/kinase_library/__init__.py'.ljust(69) + '#',
        '# 4. Update test files in src/tests/ (if applicable)'.ljust(69) + '#',
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

def _kinase_to_results_dict(kinase_obj):
    """Build a processed_results-style dict from a Kinase object's attributes."""
    if not hasattr(kinase_obj, 'raw_matrix'):
        raise AttributeError(
            'Kinase object has no raw_matrix attribute. '
            'Use kinase_from_densitometry() to create a savable Kinase object.'
        )

    # Matrices are saved with positions as rows and amino acids as columns
    # (transpose back from Kinase convention of amino acids as rows)
    result = {
        'raw': kinase_obj.raw_matrix.transpose(),
        'norm': kinase_obj.norm_matrix.transpose(),
        'log2': kinase_obj.log2_matrix.transpose(),
    }
    if kinase_obj.kin_type == 'ser_thr' and hasattr(kinase_obj, 'st_fav_series'):
        result['st_favorability'] = kinase_obj.st_fav_series
    if hasattr(kinase_obj, '_densitometry'):
        result['densitometry'] = kinase_obj._densitometry

    return {kinase_obj.name: result}


def kinase_from_densitometry(densitometry, name, kin_type, lib_type,
                             dual_spec=False, k_mod=False,
                             family='undefined', round_digits=4,
                             output_dir=None, update_database=False,
                             confirm=True, kinome_info=None):
    """
    Read a densitometry matrix into a kl.Kinase object.

    Processes the densitometry through the full pipeline (raw, norm, log2)
    and returns a kl.Kinase object with all standard attributes and methods
    (plot_data, seq_logo, heatmap, score, percentile, etc.). The raw matrix
    is attached as an extra attribute, and a save() method is bound to the
    instance for optional persistence.

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
    output_dir : str, optional
        Save matrices to this directory. The default is None (no save).
    update_database : bool, optional
        If True, perform the full package update. The default is False.
    confirm : bool, optional
        If True, prompt before updating the package database. The default is True.
    kinome_info : str, dict, or pd.DataFrame, optional
        Pre-built kinome_information row(s). Only used when update_database is
        True. If None and confirm=True, fields are prompted interactively.

    Returns
    -------
    kin : kl.Kinase
        Kinase object with raw_matrix attached and save() bound.
    """
    exceptions.check_kin_type(kin_type)
    exceptions.check_lib_type(lib_type)

    # Resolve densitometry to a DataFrame so we can stash it on the Kinase
    # for later persistence to the package's densitometry/ folder.
    if isinstance(densitometry, str):
        densitometry_df = pd.read_csv(densitometry, sep='\t', index_col=0)
    else:
        densitometry_df = densitometry.copy()

    result = process_kinase(
        densitometry_df, kinase_name=name, kin_type=kin_type, lib_type=lib_type,
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

    # Store processing context for later save
    kin._densitometry = densitometry_df
    kin._lib_type = lib_type
    kin._dual_spec = dual_spec
    kin._round_digits = round_digits

    # Bind unified save method to the instance
    def save(self, output_dir=None, update_database=False, confirm=True,
             kinome_info=None):
        save_matrices(
            _kinase_to_results_dict(self),
            kin_type=self.kin_type,
            output_dir=output_dir,
            update_database=update_database,
            confirm=confirm,
            kinome_info=kinome_info,
            kinase_metadata={self.name: (self.kin_type, self._lib_type,
                                         self._dual_spec, self.family)},
            round_digits=self._round_digits,
        )

    kin.save = types.MethodType(save, kin)

    # If save flags were passed at creation, run the save now
    if output_dir is not None or update_database:
        kin.save(output_dir=output_dir, update_database=update_database,
                 confirm=confirm, kinome_info=kinome_info)

    return kin


def kinases_from_densitometry_dir(densitometry_dir, kin_type, lib_type,
                                  kinase_names=None, dual_spec=False,
                                  k_mod=False, family='undefined',
                                  round_digits=4, output_dir=None,
                                  update_database=False, confirm=True,
                                  kinome_info=None):
    """
    Process all densitometry files in a directory into kl.Kinase objects.

    Mirrors the original matrix_processing.py command-line behavior.

    Parameters
    ----------
    densitometry_dir : str
        Directory containing `{KINASE}_densitometry.txt` files.
    kin_type : str
        Kinase type ('ser_thr' or 'tyrosine').
    lib_type : str
        Library type ('ser_thr', 'tyr', or 'ser_thr_tyr').
    kinase_names : list, optional
        Specific kinases to process. The default is None (all files in the dir).
    dual_spec, k_mod, family, round_digits : see process_kinases.
    output_dir : str, optional
        Save matrices to this directory. The default is None.
    update_database : bool, optional
        Perform full package update. The default is False.
    confirm : bool, optional
        Prompt before updating the package database. The default is True.
    kinome_info : str, dict, or pd.DataFrame, optional
        Pre-built kinome_information rows for all kinases. Required when
        update_database=True and confirm=False.

    Returns
    -------
    kinases : dict
        {kinase_name: kl.Kinase} dictionary.
    """
    exceptions.check_kin_type(kin_type)
    exceptions.check_lib_type(lib_type)

    if not densitometry_dir.endswith('/'):
        densitometry_dir = densitometry_dir + '/'

    available = [
        '_'.join(name.split('.')[0].split('_')[:-1])
        for name in os.listdir(densitometry_dir)
        if not name.startswith('.')
    ]

    if kinase_names is None:
        kinase_names = sorted(available)

    print(f'Processing {len(kinase_names)} kinase(s) from {densitometry_dir}')
    kinases = {}
    for kinase in tqdm(kinase_names):
        if kinase not in available:
            print(f"ERROR: kinase '{kinase}' not found in {densitometry_dir}")
            continue
        kinase_file = [
            x for x in os.listdir(densitometry_dir)
            if '_'.join(x.split('.')[0].split('_')[:-1]) == kinase
        ][0]
        kinases[kinase] = kinase_from_densitometry(
            densitometry_dir + kinase_file,
            name=kinase, kin_type=kin_type, lib_type=lib_type,
            dual_spec=dual_spec, k_mod=k_mod, family=family,
            round_digits=round_digits,
            # don't save individually — we'll save the batch below
        )

    if output_dir is None and not update_database:
        return kinases

    # Build processed_results dict and run a single batch save (one combined
    # rebuild, one phosphoproteome update at the end)
    processed_results = {}
    kinase_metadata = {}
    for name, kin in kinases.items():
        processed_results[name] = _kinase_to_results_dict(kin)[name]
        kinase_metadata[name] = (kin.kin_type, kin._lib_type, kin._dual_spec, kin.family)

    save_matrices(
        processed_results, kin_type=kin_type,
        output_dir=output_dir, update_database=update_database,
        confirm=confirm, kinome_info=kinome_info,
        kinase_metadata=kinase_metadata, round_digits=round_digits
    )

    return kinases
