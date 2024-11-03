"""
##################################
# The Kinase Library - Utilities #
##################################
"""
import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .. import logomaker
from tqdm import tqdm
from Bio import SeqIO
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

from ..utils import _global_vars, exceptions
from ..modules import data

#%%
"""
############
# Matrices #
############
"""

def matrix_to_df(mat, kin_type=None, pp=True, k_mod=False, mat_type='log2', cols=None, rows=None):
    """
    Converting NumPy.ndarray to Pandas.DataFrame.

    Parameters
    ----------
    mat : np.ndarray
        Matrix values with appropriate shape:
            Rows:
                * Serine/Threonine: 9 (-5 to +4)
                * Tyrosine: 11 (-5 to +6)
            Columns:
                * Only unmodified residues: 20
                * Unmodified residues and phospho-residues: 23
                * Unmodified residues, phospho-residues and modified lysine: 25
    kin_type : str, optional
        Kinase type. If ser_thr or tyrosine, annotation conventions will be used.
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

    Returns
    -------
    df_mat : pd.DataFrame
        Matrix as Pandas dataframe.
    """
    
    if cols is None:
        if mat_type == 'densitometry':
            cols = [str(i) for i in range(mat.shape[1])]
        elif mat_type in _global_vars.valid_mat_types: #raw, norm, log2
            if not pp:
                cols = _global_vars.aa_unmod
            elif not k_mod:
                cols = _global_vars.aa_phos
            else:
                cols = _global_vars.aa_all_ptm
                
    if rows is None:
        if mat_type == 'densitometry':
            rows = list(string.ascii_uppercase[:mat.shape[1]])
        elif (mat_type in _global_vars.valid_mat_types) and (kin_type in _global_vars.valid_kin_types):
                rows = data.get_positions(kin_type)
    
    if len(rows) != mat.shape[0]:
        if rows is None:
            raise Exception('As default, {} matrices must be of shape of {}'.format(kin_type,(len(rows),len(cols))))
        else:
             raise Exception('Length of provided rows ({}) does not match the number of rows in the provided matrix ({})'.format(len(rows),mat.shape[0]))
    if len(cols) != mat.shape[1]:
        if cols is None:
            raise Exception('As default, {} matrices must be of shape of {}'.format(kin_type,(len(rows),len(cols))))
        else:
            raise Exception('Length of provided columns ({}) does not match the number of columns in the provided matrix ({})'.format(len(cols),mat.shape[1]))
    
    df_mat = pd.DataFrame(mat, index=rows, columns=cols)
    
    return(df_mat)


def flatten_matrix(matrix):
    """
    Flattening matrix (concatenating positions)

    Parameters
    ----------
    matrix : pd.DataFrame
        Matrix.

    Returns
    -------
    flat_matrix : np.ndarray
        Flatten matrix.
    """
    
    flat_matrix = matrix.transpose().values.reshape(matrix.shape[0] * matrix.shape[1],1).transpose()
    return (flat_matrix)


def make_seq_logo(matrix, logo_type='ratio_to_median', random_aa_value=0.05,
                  zero_pos=None, color_dict=None, zero_pos_color='darkmagenta',
                  title=None, xlabel='Position', ylabel=None, xlabel_size=24, ylabel_size=18,
                  xticks=True, yticks=True, xticks_fontsize=24, yticks_fontsize=24,
                  save_fig=False, return_fig=False, plot=True, ax=None,
                  flip_below=False, symmetric_y_axis=False, font_name='Arial Rounded MT Bold'):
    """
    Making sequence logo from a linear kinase matrix.

    Parameters
    ----------
    matrix : pd.DataFrame
        Kinase matrix (linear scale). Amino acids as rows, positions as columns.
    method : str
        Method for normalizing matrix. 'ratio_to_random' or 'ratio_to_median'.
    logo_type : str, optional
        Type of sequence logo - 'ratio_to_random', 'ratio_to_median', or 'prob'. The default is 'ratio_to_median'.
        ratio_to_random: height is ratio to random amino acid value. 'ratio_to_random' must be specified.
        ratio_to_median: height is ratio to position-median.
        prob: probability, all values sum up to 1.
    random_aa_value : float, optional
        Value of random amino acid to normalize for. The default is 0.05.
    zero_pos : dict, optional
        Dictionary with relative intensities of the zero position. The default is None.
        Intensities will be normalized to sum up to 1.
    color_dict : dict, optional
        Dictionary with the color scheme for the amino acids. The default is None.
    zero_pos_color : string, optional
        Color for the zero position. The default is 'darkmagenta'.
    title : str, optional
        Figure title. The default is None.
    xlabel : str, optional
        X-axis label. The default is 'Position'. If False, no label will be diplayed.
    ylabel : str, optional
        Y-axis label. If None, specified based on logo_type. If False, no label will be diplayed. The default is None.
    xlabel_size : float, optional
        Size of x-axis label. The Default is 24.
    ylabel_size : float, optional
        Size of y-axis label. The Default is 18.
    xticks : bool, optional
        Plotting x-ticks labels. The Default is True.
    yticks : bool, optional
        Plotting y-ticks labels. The Default is True.
    xticks_fontsize : float, optional
        Size of x-tick labels. The Default is 36.
    yticks_fontsize : float, optional
        Size of y-tick labels. The Default is 20.
    save_fig : str, optional
        Path to save the figure. The default is False.
    return_fig : bool, optional
        If True - return the figure object. The default is False.
    plot : bool, optional
        If False - sequence logo will not be displayed. The default is True.
    ax : axes, optional
        plt.Axes for the sequence logo.
    flip_below : bool, optional
        Flip amino acids below the axis. The default is False.
    symmetric_y_axis : bool, optional
        Symmetric range of the y-axis. The default is False.
    font_name : str, optional
        Font of letters. The default is 'Arial Rounded MT Bold'.
        If 'Arial Rounded MT Bold' does not exist in system - will use 'sans'.
        For a list of valid font names, run kl.list_font_names().

    Returns
    -------
    fig : plt.figure()
        If specified, returning the figure object with the sequence logo.
    """
    
    if font_name == 'Arial Rounded MT Bold' and font_name not in logomaker.list_font_names():
        font_name = 'sans'
    
    if ax is None:
        zero_lw = 3
        fig,ax = plt.subplots(figsize = [10,5])
    else:
        zero_lw = 1
        plot = False
        if save_fig or return_fig:
            raise ValueError('When Axes provided, \'save_fig\', and \'return_fig\' must be False.')
    
    if color_dict is None: 
        color_dict = _global_vars.aa_colors
    
    if logo_type == 'ratio_to_random':
        if random_aa_value is None:
            raise Exception('If method is specified to be ratio to random, \'random_aa_value\' must be provided.')
        height_matrix = np.log2(matrix/random_aa_value).transpose()
    elif logo_type == 'ratio_to_median':
        height_matrix = np.log2(matrix.divide(matrix.median())).transpose()
    elif logo_type == 'prob':
        height_matrix = matrix.divide(matrix.sum()).transpose()

    logomaker.Logo(height_matrix, ax=ax, color_scheme=color_dict, flip_below=flip_below, font_name=font_name) # adding allow_nan=True lead to an error in logomaker package
    ax.axhline(0, color='k', linewidth=zero_lw)
    height_matrix.loc[0] = 0
    if zero_pos is not None:
        max_height = height_matrix[height_matrix>0].sum(axis=1).max()
        norm_zero_pos = {key: zero_pos[key]/sum(zero_pos.values())*max_height for key in zero_pos.keys()}
        for aa in zero_pos.keys():
            height_matrix.loc[0,aa] = norm_zero_pos[aa]
    
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        logomaker.Logo(pd.DataFrame(height_matrix.loc[0]).transpose(), ax=ax, color_scheme=zero_pos_color, flip_below=flip_below, font_name=font_name)
        ax.set_xlim(xlims)
        if symmetric_y_axis:
            ax.set_ylim((-max(np.abs(ylims)),max(np.abs(ylims))))
        else:
            ax.set_ylim(ylims)
    else:
        ax.axvline(0, color='k', linewidth=3, linestyle=':')
    
    ax.set_xticks(ticks=sorted(height_matrix.index))
    ax.set_xticklabels(labels=['' if not xticks else str(x) if x<=0 else '+'+str(x) for x in sorted(height_matrix.index)], fontsize=xticks_fontsize, weight = 'bold')
    if yticks:
        ax.tick_params(axis='y', labelsize=yticks_fontsize)
    else:
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    if title is not None:
        ax.set_title(title, fontsize=24)
    
    if xlabel != False:
        ax.set_xlabel(xlabel, fontsize=xlabel_size)
    if ylabel != False:
        if ylabel is None:
            if logo_type == 'ratio_to_random':
                ax.set_ylabel('log2(Ratio to Random)', fontsize=ylabel_size)
            elif logo_type == 'ratio_to_median':
                ax.set_ylabel('log2(Ratio to Median)', fontsize=ylabel_size)
            elif logo_type == 'prob':
                ax.set_ylabel('Probability',fontsize = ylabel_size)
        else:
            ax.set_ylabel(ylabel, fontsize=ylabel_size)
    
    if save_fig:
        fig.savefig(save_fig)
    
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
    

#%%
"""
##############
# Substrates #
##############
"""

def filter_invalid_subs(data, seq_col, suppress_warnings=True):
    """
    Filtering enteries with invalid sequences:
        1. NaN values
        2. Invalid amino acids or characters
        3. Even length (cannot define central position)
        4. Invalid central phosphoacceptor

    Parameters
    ----------
    data : pd.DataFrame
        Input data with sequences.
    seq_col : str
        Sequence column.
    suppress_warnings : bool, optional
        Do not print warnings. The default is False.

    Returns
    -------
    data_no_invalid_phos_acc : pd.DataFrame
        Filtered data without invalid sequences.
    omitted_enteries : pd.DataFrame
        Dropped enteries due to invalid sequence.
    """
    
    data_no_na = data[data[seq_col].notna()]
    if not suppress_warnings and ((len(data)-len(data_no_na))>0):
        print(str(len(data)-len(data_no_na)) + ' entries were omitted due to empty value in the substrates column.')
    
    escaped_aa_list = [re.escape(c) for c in _global_vars.valid_aa]
    data_no_invalid_aa = data_no_na[data_no_na[seq_col].astype(str).str.contains('^['+''.join(escaped_aa_list)+']+$')]
    if not suppress_warnings and ((len(data_no_na)-len(data_no_invalid_aa))>0):
        print(str(len(data_no_na)-len(data_no_invalid_aa)) + ' entries were omitted due to invalid amino acids or characters.')
    
    data_no_even = data_no_invalid_aa[(data_no_invalid_aa[seq_col].str.len())%2 != 0]
    if not suppress_warnings and ((len(data_no_invalid_aa)-len(data_no_even))>0):
        print(str(len(data_no_invalid_aa)-len(data_no_even)) + ' entries were omitted due to even length (no central position).')
    
    data_no_invalid_phos_acc = data_no_even[data_no_even[seq_col].apply(lambda x: x[len(x)//2]).isin(_global_vars.valid_phos_res)]
    if not suppress_warnings and ((len(data_no_even)-len(data_no_invalid_phos_acc))>0):
        print(str(len(data_no_even)-len(data_no_invalid_phos_acc)) + ' entries were omitted due to invalid central phosphoacceptor.')

    omitted_enteries = pd.concat([data, data_no_invalid_phos_acc])
    omitted_enteries = omitted_enteries.loc[omitted_enteries.astype(str).drop_duplicates(keep=False).index] #In order to deal with lists in the DataFrame
    
    return(data_no_invalid_phos_acc, omitted_enteries)


def sequence_to_substrate(seq, pp=False, phos_pos=None, kin_type=None, validate_phos_res=True, validate_aa=True):
    """
    Converting general sequence to a substrate (15-mer).

    Parameters
    ----------
    seq : str
        Sequence with phosphorylation site.
    pp : bool, optional
        Phospho-priming (phospho-residues in the sequence). The default is False.
    phos_pos : int, optional
        Position of phosphoacceptor. The default is None.
    kin_type : str, optional
        Kinase type. If ser_thr or tyrosine, annotation conventions will be used.
    validate_phos_res : bool, optional
        validating phosphoacceptor. The default is True.
    validate_aa : bool, optional
        Validating amino acids. The default is True.

    Returns
    -------
    substrate : str
        15-mer with phosphoacceptor at the center.
    """
    
    if validate_phos_res:
        if phos_pos is None:
            if (seq[len(seq)//2].lower() not in ['s','t','y']) | (len(seq) % 2 == 0):
                raise Exception('Central residue must be a phosphoacceptor (s/t/y). Otherwise, please identify the position of the phosphoacceptor.')
        elif seq[phos_pos-1].lower() not in ['s','t','y']:
            raise Exception('Invalid phosphoacceptor: {} (must be s/t/y). Use \x1B[3mvalidate_phos_res=False\x1B[0m to ignore.'.format(seq[phos_pos-1].lower()))
    if validate_aa:
        if not _global_vars.valid_aa.issuperset(seq.upper()):
            raise Exception('Sequence contains invalid amino acids or characters ({}).\nAllowed characters are: {}'.format(seq.upper(),_global_vars.valid_aa))
    
    if seq is np.nan:
        return(None)

    if phos_pos is None:
        phos_pos = len(seq)//2 + 1
    
    if kin_type is not None:
        exceptions.check_kin_type(kin_type)
        phos_acc = seq[phos_pos-1]
        if phos_acc not in _global_vars.kin_type_phos_acc[kin_type]:
            raise Exception(f'Mismatch beteween kinase type ({kin_type}) and phosphoacceptor ({phos_acc}): {seq}')
        
    pad_seq = '_'*7 + seq + '_'*7
    substrate = pad_seq[phos_pos-1:phos_pos+14]
    
    if pp:
        substrate = ''.join([x.upper() if x not in ['s','t','y'] else x for x in substrate[:7]]) + substrate[7].lower() + ''.join([x.upper() if x not in ['s','t','y'] else x for x in substrate[8:]])
    else:
        substrate = substrate[:7].upper() + substrate[7].lower() + substrate[8:].upper()
    
    return(substrate)


def sub_binary_matrix(substrates, pp=True, aa=None, pos=None, sub_pos=None, seq_col=None, as_dict=False):
    """
    Making a binary matrix for a substrate.

    Parameters
    ----------
    substrates : str, list, pd.Series, or pd.DataFame
        Substrates list (15-mer, central residue is phosphoacceptor).
        If input format is dataframe, sequence column must be specified.
    pp : bool, optional
        Phospho-priming residues (s/t/y). The default is True.
    aa : list, optional
        List of amino acids to use in the matrix columns. The default is None.
    pos : list, optional
        List of positions to use in the matrix rows. The default is None.
    sub_pos : list, optional
        List of positions indicating the substrate positions. The default is None.
    seq_col: str, optional
        Sequence column. Must be specified if input format is dataframe.
    as_dict : bool, optional
        If True, return values as dictionary. The default is False.

    Returns
    -------
    sub_mat : pd.DataFrame
        Binary matrix of the substrate.
        If input is one substrate - returns single binary martix.
        If input is list of substrates - returns dataframe with flattened binary matrices.
    """
    
    if isinstance(substrates, str):
        subs_list = pd.Series([substrates], name='Sequence')
    elif isinstance(substrates, list):
        subs_list = pd.Series(substrates, name='Sequence')
    elif isinstance(substrates, pd.Series):
        subs_list = substrates
    elif isinstance(substrates, pd.DataFrame):
        if seq_col is None:
            raise Exception('If input format is DataFrame, sequence column must be specified.')
        subs_list = substrates[seq_col]
    else:
        raise Exception('Invalid substrates list.')
        
    if not pp:
        subs_list = subs_list.apply(lambda x: unprime_substrate(x))
    
    if pos is None:
        pos = list(range(1,max([len(x) for x in subs_list])+1))
    if sub_pos is None:
        sub_pos = range(-7,8)
    if aa is None:
        aa = data.get_aa()
    
    subs_pos_aa = [[str(p)+a for a,p in zip(sub,sub_pos)] for sub in subs_list]
    
    bin_mat_classes = [str(pos)+a for pos in pos for a in aa]
    mlb = MultiLabelBinarizer(classes=bin_mat_classes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bin_mat = mlb.fit_transform(subs_pos_aa)

    if isinstance(substrates, str):
        return(pd.DataFrame(bin_mat.reshape(len(pos), len(aa)), index=pos, columns=aa).T)
    
    if as_dict:
        bin_df = [pd.DataFrame(bin_mat.reshape(len(pos), len(aa)), index=pos, columns=aa).T for x in bin_mat]
        return(dict(zip(subs_list,bin_df)))
    
    sub_mat = pd.DataFrame(bin_mat, index=subs_list, columns=bin_mat_classes)
    
    return(sub_mat)


def substrate_type(sub):
    """
    Returning substrate type based on the central phosphoacceptor.

    Parameters
    ----------
    sub : str
        Substrate (15-mer, central residue is phosphoacceptor).

    Returns
    -------
    sub_type : str
        Substrate type ('ser_thr', 'tyrosine', or 'unknown').
    """
    
    if sub[7].lower() in ['s','t']:
        sub_type = 'ser_thr'
    elif sub[7].lower() == 'y':
        sub_type = 'tyrosine'
    else:
        sub_type = 'unknown'
    return(sub_type)


def unprime_substrate(sub):
    """
    Convert substrate to unprimed version

    Parameters
    ----------
    sub : str
        Substrate (15-mer, central residue is phosphoacceptor).

    Returns
    -------
    un_primed_sub : str
        The input substrate without phospho-residues.
    """
        
    un_primed_sub = sub[:7].upper() + sub[7] + sub[8:].upper()
    return(un_primed_sub)


#%%
"""
###################
# Other utilities #
###################
"""

def list_font_names():
    """
    Returns a list of valid font_name options for use in sequence logo.

    parameters
    ----------
    None.

    returns
    -------
    fontname : list
        List of valid font_name names from logomaker. This will vary from system to system.
    """
    
    return logomaker.list_font_names()


def list_series_to_df(subs_list, col_name=None):
    """
    Convert list or pd.Series into a one-column pd.DataFrame

    Parameters
    ----------
    subs_list : list or pd.Series
        List or Series.
    col_name : str, optional
        Column name. The default is None.
        If None - will use the name from the pd.Series object or '0' for a list.

    Returns
    -------
    pd.DataFrame with one column containing the list or pd.Series.
    """
    
    if isinstance(subs_list, pd.Series):
            return(subs_list.to_frame(name=col_name))
    if isinstance(subs_list, list):
            return(pd.Series(subs_list).to_frame(name=col_name))
