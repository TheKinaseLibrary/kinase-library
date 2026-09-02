"""
##########################################
# The Kinase Library - Sequence Retrieval #
##########################################

Turning a protein identifier plus a site position into the 15-mer window the
rest of the package scores.

A collaborator sends a table with a UniProt accession (or a gene symbol) and a
site position, and no sequence column. `retrieve_sequences` turns that into a
table `PhosphoProteomics` can score: it appends `SITE_+/-7_AA` and reports, per
row, every site it could not resolve and why.

Provenance
----------
This is `retrieve_sequences` from the legacy `py_kinase_library`, restored with
its defects fixed. Numerical behaviour is preserved exactly: seven underscores
of padding at each end of every protein, the `[pos-1 : pos+14]` slice, the
central-residue check against `s/t/y`, and the `SITE_+/-7_AA` output column
name. The window is returned in the case the FASTA has it in (upper), and
`utils.sequence_to_substrate` lowercases the central residue downstream.

Four copies of the legacy function exist on disk and they disagree about
`id_format`. The bundled copy branched on `'gene_name'`, so the documented
`'gene_symbol'` fell through to accession matching and silently matched
nothing; the published copy branched on `'gene_symbol'` but extracted the
UniProt entry-name prefix (`P53`) rather than the `GN=` gene symbol (`TP53`).
Here `gene_symbol` means the `GN=` field, `protein_name` means the entry-name
prefix, and an unrecognised `id_format` raises rather than falling through.

The reference proteomes are NOT bundled with this package - they are hundreds
of megabytes. See `set_current_proteome_dir` and `download_proteome`.
"""

import argparse
import datetime
import gzip
import os
import re
import shutil
import sys
import warnings
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from Bio import SeqIO

from ..utils import _global_vars, exceptions

__all__ = ['retrieve_sequences', 'load_proteome', 'parse_position',
           'download_proteome', 'SiteResolutionError']

#%%

WINDOW_COL = _global_vars.site_window_column
PAD = '_'*7

# Why a row did not resolve. A fixed enumeration: every miss carries exactly one
# of these, so a caller reports counts per reason rather than a single "n
# missing" that hides four different problems.
REASON_NOT_IN_FASTA = 'accession_not_in_fasta'
REASON_UNPARSEABLE = 'position_unparseable'
REASON_OUT_OF_RANGE = 'position_out_of_range'
REASON_NOT_STY = 'central_residue_not_STY'
REASON_MISMATCH = 'residue_mismatch'
REASONS = (REASON_NOT_IN_FASTA, REASON_UNPARSEABLE, REASON_OUT_OF_RANGE,
           REASON_NOT_STY, REASON_MISMATCH)


class SiteResolutionError(ValueError):
    """Raised when a site cannot be resolved and the caller asked to be told.

    A subclass of ValueError, so `except ValueError` around legacy code that
    expected the old `raise ValueError('No proteins were found...')` still
    catches it.
    """

#%%
"""
####################
# Reference proteome #
####################
"""

def _proteome_file(species=None, ref_prot_file=None):
    """
    Resolve which FASTA file to read.

    An explicit `ref_prot_file` always wins. Otherwise the species is looked up
    in the current reference proteome directory, trying a current download
    before the July-2021 file that shipped with the legacy package.

    Parameters
    ----------
    species : str, optional
        One of `kl._global_vars.valid_species`. The default is 'human'.
    ref_prot_file : str, optional
        Explicit path to a FASTA file. The default is None.

    Returns
    -------
    str
        Path to an existing FASTA file.

    Raises
    ------
    ValueError
        If `species` is not recognised.
    FileNotFoundError
        If no proteome file can be found, naming every way to fix it.

    """

    if ref_prot_file:
        if not os.path.isfile(ref_prot_file):
            raise FileNotFoundError(f'Reference proteome file was not found: {ref_prot_file}')
        return(ref_prot_file)

    if species is None:
        species = _global_vars.default_species
    exceptions.check_species(species)

    proteome_dir = _global_vars.proteome_dir
    candidates = _global_vars.proteome_files[species]
    for file_name in candidates:
        path = os.path.join(proteome_dir, file_name)
        if os.path.isfile(path):
            return(path)

    raise FileNotFoundError(
        f'Reference proteome for species \'{species}\' was not found in: {proteome_dir}\n'
        f'Looked for: {", ".join(candidates)}\n'
        'The UniProt FASTA files are not bundled with this package. Fix by any of:\n'
        f'  - set the {_global_vars.proteome_dir_env_var} environment variable to the directory holding them\n'
        '  - call kl.set_current_proteome_dir(directory)\n'
        '  - pass ref_prot_file=path to point at one file directly\n'
        f'  - call kl.download_proteome(\'{species}\') to fetch a current proteome from UniProt')


UNIPROT_STREAM_URL = 'https://rest.uniprot.org/uniprotkb/stream'


def download_proteome(species='human', dest_dir=None, overwrite=False, timeout=600):
    """
    Fetch a current UniProt reference proteome, isoforms included.

    The package ships no FASTA files, so this is how a fresh install gets one.
    The download is written to a temporary name and moved into place only once
    it is complete, so an interrupted download never leaves a half file under
    the real name. A `.release.txt` sidecar records which UniProt release the
    windows will have come from, which is what a methods section needs.

    Parameters
    ----------
    species : str, optional
        'human', 'mouse' or 'rat'. The default is 'human'. 'all' has no single
        UniProt reference proteome and is not downloadable.
    dest_dir : str, optional
        Where to write it. The default is None, meaning the current reference
        proteome directory (`kl.get_current_proteome_dir()`). Created if absent.
    overwrite : bool, optional
        Replace an existing file of the same name. The default is False.
    timeout : int, optional
        Socket timeout in seconds. The default is 600.

    Returns
    -------
    str
        Path to the downloaded FASTA file.

    Raises
    ------
    ValueError
        If `species` is not recognised, or is 'all'.
    FileExistsError
        If the target exists and `overwrite` is False.
    urllib.error.URLError
        If the download fails. No partial file is left behind.

    Examples
    --------
    >>> import kinase_library as kl
    >>> kl.set_current_proteome_dir('~/uniprot')            # doctest: +SKIP
    >>> kl.download_proteome('human')                       # doctest: +SKIP

    """

    exceptions.check_species(species)
    if species not in _global_vars.uniprot_proteome_ids:
        raise ValueError(
            f'species \'{species}\' cannot be downloaded: it is not a single UniProt '
            f'reference proteome. Downloadable: {sorted(_global_vars.uniprot_proteome_ids)}. '
            'For anything else, download it yourself and pass ref_prot_file, or put it in '
            'the reference proteome directory.')

    proteome_id = _global_vars.uniprot_proteome_ids[species]
    if dest_dir is None:
        dest_dir = _global_vars.proteome_dir
    os.makedirs(dest_dir, exist_ok=True)

    file_name = _global_vars.proteome_files[species][0]
    target = os.path.join(dest_dir, file_name)
    if os.path.isfile(target) and not overwrite:
        raise FileExistsError(f'{target} already exists. Pass overwrite=True to replace it.')

    url = (f'{UNIPROT_STREAM_URL}?format=fasta&includeIsoform=true&compressed=true'
           f'&query=%28proteome%3A{proteome_id}%29')
    partial = target + '.part'
    print(f'Downloading the {species} reference proteome ({proteome_id}) from UniProt ...')
    try:
        request = Request(url, headers={'User-Agent': 'kinase-library'})
        with urlopen(request, timeout=timeout) as response:
            release = response.headers.get('X-UniProt-Release', 'unknown')
            release_date = response.headers.get('X-UniProt-Release-Date', 'unknown')
            with gzip.GzipFile(fileobj=response) as stream, open(partial, 'wb') as out_file:
                shutil.copyfileobj(stream, out_file)
        os.replace(partial, target)
    except BaseException:
        if os.path.isfile(partial):
            os.remove(partial)
        raise

    with open(target) as handle:
        records = sum(1 for line in handle if line.startswith('>'))

    sidecar = os.path.splitext(target)[0] + '.release.txt'
    downloaded = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    with open(sidecar, 'w') as handle:
        handle.write(f'file: {file_name}\n'
                     f'species: {species}\n'
                     f'proteome_id: {proteome_id}\n'
                     f'source_url: {url}\n'
                     f'uniprot_release: {release}\n'
                     f'uniprot_release_date: {release_date}\n'
                     f'downloaded_utc: {downloaded}\n'
                     f'records: {records}\n')

    print(f'  {records} records (UniProt release {release})  ->  {target}')
    print(f'  provenance  ->  {sidecar}')
    return(target)


def _record_id(record, id_format):
    """
    The key a FASTA record is indexed under, or None if it has none.

    `uniprot_acc` keeps the FULL accession including any isoform suffix, so
    `Q00266-2` indexes the isoform and the bare `Q00266` indexes the canonical
    record. Collapsing them would silently score the wrong sequence.

    Parameters
    ----------
    record : Bio.SeqRecord.SeqRecord
        One parsed FASTA record.
    id_format : str
        One of `kl._global_vars.valid_id_format`.

    Returns
    -------
    str or None
        The index key, or None when the record carries no such identifier.

    """

    if id_format == 'uniprot_acc':
        parts = record.id.split('|')
        return(parts[1] if len(parts) > 1 else record.id)
    if id_format == 'protein_name':
        parts = record.id.split('|')
        return(parts[2].split('_')[0] if len(parts) > 2 else None)
    if id_format == 'gene_symbol':
        match = re.search(r'GN=([^\s]+)', record.description)
        return(match.group(1) if match else None)
    exceptions.check_id_format(id_format)


def _description(record):
    """
    The protein name, between the identifier and the first `XX=` field.

    Parameters
    ----------
    record : Bio.SeqRecord.SeqRecord
        One parsed FASTA record.

    Returns
    -------
    str
        The free-text description, or an empty string.

    """

    text = record.description.split(' ', 1)[1] if ' ' in record.description else ''
    return(re.split(r'\s+[A-Z]{2}=', text, maxsplit=1)[0].strip())


def load_proteome(fasta_path, id_format='uniprot_acc'):
    """
    Parse a FASTA file into a dataframe indexed by identifier.

    Duplicate keys are resolved rather than taken in file order. Accessions are
    unique, but many records share a `GN=`, so for `gene_symbol` the canonical
    record wins over an isoform, and a Swiss-Prot record wins over TrEMBL.

    Parameters
    ----------
    fasta_path : str
        Path to a UniProt FASTA file.
    id_format : str, optional
        How to index the records: 'uniprot_acc' (the full accession, isoform
        suffix included), 'gene_symbol' (the `GN=` field) or 'protein_name'
        (the entry-name prefix). The default is 'uniprot_acc'.

    Returns
    -------
    pd.DataFrame
        Indexed by identifier, with columns `NAME` (the full FASTA id),
        `DESCRIPTION`, `SEQUENCE` (padded with seven underscores at each end)
        and `LENGTH` (the unpadded residue count).

    Raises
    ------
    ValueError
        If `id_format` is not recognised.
    FileNotFoundError
        If `fasta_path` does not exist.

    """

    exceptions.check_id_format(id_format)
    if not os.path.isfile(fasta_path):
        raise FileNotFoundError(f'Reference proteome file was not found: {fasta_path}')

    keys, names, descriptions, sequences, lengths = [], [], [], [], []
    canonical, swissprot = [], []

    with open(fasta_path) as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            key = _record_id(record, id_format)
            if key is None:
                continue
            parts = record.id.split('|')
            accession = parts[1] if len(parts) > 1 else record.id
            raw = str(record.seq)

            keys.append(key)
            names.append(record.id)
            descriptions.append(_description(record))
            sequences.append(PAD + raw + PAD)
            lengths.append(len(raw))
            canonical.append('-' not in accession)
            swissprot.append(record.id.startswith('sp|'))

    proteome = pd.DataFrame(
        {'NAME': names, 'DESCRIPTION': descriptions, 'SEQUENCE': sequences,
         'LENGTH': lengths, '_canonical': canonical, '_swissprot': swissprot},
        index=pd.Index(keys, name=id_format))
    # Stable sort: canonical before isoform, Swiss-Prot before TrEMBL, then file
    # order - so keep='first' picks the record a human would pick.
    proteome = proteome.sort_values(['_canonical', '_swissprot'],
                                    ascending=[False, False], kind='stable')
    proteome = proteome[~proteome.index.duplicated(keep='first')]
    return(proteome.drop(columns=['_canonical', '_swissprot']))

#%%
"""
###################
# Site positions  #
###################
"""

# A site cell is rarely a bare integer. Collaborator tables carry 'S33',
# 'Ser33', 'pY1068', 'S-33', and - whenever the column has a single NaN -
# pandas turns the whole column to float64 and every position arrives as '33.0'.
# All of those mean the same site.
#
# The residue letter, when present, is not decoration: it is checked against the
# residue actually at that position, and that check is the only thing standing
# between an off-by-one and a silently-scored wrong window when the neighbouring
# residue also happens to be a phosphoacceptor.
_SITE_RE = re.compile(
    r"""^\s*
        (?:p(?=[A-Za-z]))?                        # phospho prefix, only before a letter
        \s*
        (?:(Ser|Thr|Tyr|[A-Za-z])\s*[-_ ]?\s*)?   # optional residue
        (\d+)                                     # the position
        (?:\.0+)?                                 # tolerate 33.0 from a float column
        \s*$""",
    re.VERBOSE | re.IGNORECASE)

_THREE_LETTER = {'ser': 'S', 'thr': 'T', 'tyr': 'Y'}


def parse_position(value):
    """
    Read a site cell into a residue letter and a position.

    Parameters
    ----------
    value : int, float or str
        The cell as it arrived: 33, '33', 'S33', 'Ser33', 'pY1068', 'S-33',
        '  S33 ', or the 33.0 pandas produces once a column holds one NaN.

    Returns
    -------
    tuple
        `(residue_or_None, position_or_None)`. A None position means the cell
        could not be read at all - the caller turns that into
        `position_unparseable` rather than guessing.

    """

    if value is None:
        return(None, None)
    if isinstance(value, float) and np.isnan(value):
        return(None, None)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return(None, int(value))

    match = _SITE_RE.match(str(value))
    if not match:
        return(None, None)

    residue, position = match.group(1), int(match.group(2))
    if residue:
        residue = _THREE_LETTER.get(residue.lower(), residue).upper()
    return(residue, position)


def _resolve_one(key, position, proteome, expect_residue=None):
    """
    Resolve one site to a window, or to the reason it failed.

    `expect_residue`, when given, is checked against the residue actually at
    that position. That check is the only defence against an off-by-one where
    the neighbouring residue is also a phosphoacceptor and the central-residue
    test would pass anyway.

    Parameters
    ----------
    key : str
        The protein identifier, as indexed in `proteome`.
    position : int or None
        1-based position of the phosphoacceptor.
    proteome : pd.DataFrame
        As returned by `load_proteome`.
    expect_residue : str, optional
        The residue letter the caller stated, if any. The default is None.

    Returns
    -------
    tuple
        `(window, None)` on success, `(None, reason)` on failure.

    """

    if key not in proteome.index:
        return(None, REASON_NOT_IN_FASTA)

    try:
        pos = int(position)
    except (TypeError, ValueError):
        return(None, REASON_UNPARSEABLE)

    length = int(proteome.at[key, 'LENGTH'])
    if pos < 1 or pos > length:
        return(None, REASON_OUT_OF_RANGE)

    sequence = proteome.at[key, 'SEQUENCE']
    motif = sequence[pos-1:pos+14]
    if len(motif) < 15 or motif[7] not in _global_vars.valid_phos_res:
        return(None, REASON_NOT_STY)

    if expect_residue and motif[7].upper() != expect_residue.upper():
        return(None, REASON_MISMATCH)

    return(motif, None)


def _resolve_sites(site_data, protein_column, pos_column, proteome, strict=False):
    """
    Append the window column to a copy of `site_data`; return it and the misses.

    Parameters
    ----------
    site_data : pd.DataFrame
        The site table.
    protein_column : str
        Column holding the protein identifier.
    pos_column : str
        Column holding the position.
    proteome : pd.DataFrame
        As returned by `load_proteome`.
    strict : bool, optional
        Raise on the first unresolved site. The default is False.

    Returns
    -------
    tuple
        `(windowed, unresolved)`. `windowed` has every input row, in the input
        order, with an empty window where the site could not be resolved, so
        the table stays joinable against whatever the collaborator sent.
        `unresolved` holds only the failed rows with an added `reason`.
        The input frame is never modified.

    Raises
    ------
    SiteResolutionError
        Under `strict`, on the first unresolved site.

    """

    windowed = site_data.copy()
    windows, reasons = [], []
    for key, position in zip(site_data[protein_column], site_data[pos_column]):
        residue, parsed = parse_position(position)
        window, reason = _resolve_one(str(key).strip(), parsed, proteome,
                                      expect_residue=residue)
        if reason is not None and strict:
            raise SiteResolutionError(
                f'{protein_column}={key!r} {pos_column}={position!r}: {reason}')
        windows.append(window if window is not None else '')
        reasons.append(reason)

    windowed[WINDOW_COL] = windows
    failed = np.array([r is not None for r in reasons], dtype=bool)
    unresolved = site_data.loc[failed].copy()
    unresolved['reason'] = [r for r in reasons if r is not None]
    return(windowed, unresolved)

#%%
"""
######################
# Sequence retrieval #
######################
"""

def _reason_summary(unresolved):
    """
    '2 accession_not_in_fasta, 1 position_out_of_range' for a message.

    Parameters
    ----------
    unresolved : pd.DataFrame
        The failed rows, carrying a `reason` column.

    Returns
    -------
    str
        Counts per reason, commonest first.

    """

    counts = unresolved['reason'].value_counts()
    return(', '.join(f'{count} {reason}' for reason, count in counts.items()))


def _deprecated_alias(new_value, old_value, new_name, old_name):
    """
    Accept an older argument spelling, once, with a warning.

    Parameters
    ----------
    new_value : object
        Whatever was passed under the current name, or None.
    old_value : object
        Whatever was passed under the deprecated name, or None.
    new_name : str
        The current argument name.
    old_name : str
        The deprecated argument name.

    Returns
    -------
    object
        The value to use.

    Raises
    ------
    TypeError
        If both spellings were passed, which is ambiguous.

    """

    if old_value is None:
        return(new_value)
    if new_value is not None:
        raise TypeError(f'Pass either \'{new_name}\' or \'{old_name}\', not both. \'{old_name}\' is the deprecated spelling of \'{new_name}\'.')
    warnings.warn(f'\'{old_name}\' is deprecated, use \'{new_name}\' instead.',
                  DeprecationWarning, stacklevel=3)
    return(old_value)


def retrieve_sequences(site_data=None, protein_column=None, pos_column=None,
                       species=None, id_format='uniprot_acc', ref_prot_file=None,
                       strict=False, drop_unresolved=False, *,
                       phospho_data=None, id_column=None,
                       protein_position_column=None, ignore_missing=None,
                       peptide_column=None):
    """
    Build the `SITE_+/-7_AA` window for every site in a table.

    Parameters
    ----------
    site_data : pd.DataFrame
        Table of phosphosites. Never modified.
    protein_column : str
        Column holding the protein identifier (accession, gene symbol or entry
        name, matching `id_format`).
    pos_column : str
        Column holding the 1-based position. Any of 33, '33', 'S33', 'Ser33',
        'pY1068', 'S-33' or 33.0 is read; a stated residue letter is verified
        against the sequence.
    species : str, optional
        Which reference proteome to use: 'human', 'mouse', 'rat' or 'all'.
        Ignored when `ref_prot_file` is given. The default is 'human'.
    id_format : str, optional
        'uniprot_acc' (the full accession, isoform suffix included),
        'gene_symbol' (the `GN=` field) or 'protein_name' (the entry-name
        prefix). The default is 'uniprot_acc'.
    ref_prot_file : str, optional
        A FASTA file to use instead of the configured proteome directory.
        The default is None.
    strict : bool, optional
        Raise `SiteResolutionError` on the first unresolved site instead of
        reporting it. The default is False.
    drop_unresolved : bool, optional
        Return a single dataframe holding only the resolved rows, which is what
        the legacy `retrieve_sequences` did. Rows are never dropped silently: a
        partial loss warns with the count per reason, and losing every row
        raises. The default is False.

    Returns
    -------
    tuple or pd.DataFrame
        By default `(windowed, unresolved)`. `windowed` holds every input row,
        in the input order, with `SITE_+/-7_AA` empty where the site could not
        be resolved, so it stays joinable against the original table.
        `unresolved` holds only the failed rows plus a `reason` column, whose
        values are one of `accession_not_in_fasta`, `position_unparseable`,
        `position_out_of_range`, `central_residue_not_STY` or
        `residue_mismatch`.
        With `drop_unresolved`, a single dataframe of the resolved rows only,
        keeping the input's index.

    Raises
    ------
    TypeError
        If `site_data` is not a dataframe.
    KeyError
        If either column is not in the table.
    ValueError
        If `species` or `id_format` is not recognised.
    FileNotFoundError
        If no reference proteome can be found.
    SiteResolutionError
        Under `strict`, on the first unresolved site; under `drop_unresolved`,
        when a non-empty table resolved nothing at all.

    Warns
    -----
    UserWarning
        Under `drop_unresolved`, when some rows are dropped.
    DeprecationWarning
        When a legacy argument spelling is used.

    Notes
    -----
    Legacy scripts written against `py_kinase_library` used other spellings,
    all of which still work and warn:

    ==========================  =================================================
    Deprecated                  Meaning now
    ==========================  =================================================
    `phospho_data=`             `site_data=`
    `id_column=`                `protein_column=`
    `protein_position_column=`  `pos_column=`
    `ignore_missing=True`       `drop_unresolved=True` (the legacy single frame,
                                warning about whatever it dropped)
    `ignore_missing=False`      `strict=True`, which raises instead of the
                                legacy `print` followed by `return None`
    `id_format='gene_name'`     `id_format='gene_symbol'`
    ==========================  =================================================

    `peptide_column=` is not supported. Ten call sites pass it, but no released
    version of the function ever accepted it, so there is no behaviour to
    restore; it raises rather than being ignored.

    Examples
    --------
    >>> import pandas as pd, kinase_library as kl
    >>> sites = pd.DataFrame({'acc': ['P04637'], 'pos': ['S15']})
    >>> windowed, unresolved = kl.retrieve_sequences(sites, 'acc', 'pos')
    >>> windowed['SITE_+/-7_AA'][0]
    'PSVEPPLSQETFSDL'

    """

    if peptide_column is not None:
        raise TypeError(
            'peptide_column is not supported. Inferring a site position from a peptide '
            'sequence was never implemented in any released version of retrieve_sequences, '
            'so there is no behaviour to restore. Map the peptide onto a position first, '
            'then pass pos_column.')

    site_data = _deprecated_alias(site_data, phospho_data, 'site_data', 'phospho_data')
    protein_column = _deprecated_alias(protein_column, id_column, 'protein_column', 'id_column')
    pos_column = _deprecated_alias(pos_column, protein_position_column, 'pos_column', 'protein_position_column')

    if ignore_missing is not None:
        warnings.warn(
            '\'ignore_missing\' is deprecated. ignore_missing=True is now '
            'drop_unresolved=True, which returns only the resolved rows and warns about '
            'the rest; ignore_missing=False is now strict=True, which raises instead of '
            'returning None.', DeprecationWarning, stacklevel=2)
        strict = not ignore_missing
        drop_unresolved = True

    if id_format == 'gene_name':
        warnings.warn(
            '\'gene_name\' is deprecated, use id_format=\'gene_symbol\' instead. Both mean '
            'the FASTA\'s GN= field.', DeprecationWarning, stacklevel=2)
        id_format = 'gene_symbol'

    if not isinstance(site_data, pd.DataFrame):
        raise TypeError(f'site_data must be a pandas DataFrame, not {type(site_data).__name__}.')
    exceptions.check_id_format(id_format)

    # Everything cheap first: parsing the proteome is the slow step, and a
    # misconfiguration must not cost the user that wait.
    for column in (protein_column, pos_column):
        if column not in site_data.columns:
            raise KeyError(f'Column \'{column}\' is not in the table. Columns present: {list(site_data.columns)}.')

    fasta_path = _proteome_file(species=species, ref_prot_file=ref_prot_file)
    proteome = load_proteome(fasta_path, id_format)

    windowed, unresolved = _resolve_sites(site_data, protein_column, pos_column,
                                          proteome, strict=strict)
    if not drop_unresolved:
        return(windowed, unresolved)

    # The legacy single-frame contract. It cannot carry the reasons, so the
    # rows it drops are announced rather than left for the caller to notice
    # from a row count they were not watching.
    if len(unresolved):
        summary = _reason_summary(unresolved)
        if len(unresolved) == len(site_data):
            raise SiteResolutionError(
                f'None of the {len(site_data)} sites could be resolved: {summary}. '
                f'Check the values in \'{protein_column}\' and \'{pos_column}\', and whether id_format=\'{id_format}\' matches them.')
        warnings.warn(
            f'{len(unresolved)} of {len(site_data)} sites could not be resolved and were dropped: {summary}. '
            'Call without drop_unresolved to get every row back with its reason.',
            UserWarning, stacklevel=2)
    return(windowed[windowed[WINDOW_COL] != ''])

#%%
"""
################
# Command line #
################
"""

def _read_table(path, sep=None, header=0):
    """
    Read a csv/tsv/xlsx table, inferring the delimiter from the extension.

    Parameters
    ----------
    path : str
        Path to the table.
    sep : str, optional
        Delimiter. The default is None, meaning infer it.
    header : int, optional
        Header row index. The default is 0.

    Returns
    -------
    pd.DataFrame
        The table.

    """

    extension = os.path.splitext(path)[1].lower()
    if extension in ('.xlsx', '.xls'):
        return(pd.read_excel(path, header=header))
    if sep is None:
        sep = ',' if extension == '.csv' else '\t'
    return(pd.read_csv(path, sep=sep, header=header))


def main(argv=None):
    """
    Command line entry point: a site table in, a windowed table out.

    Run as `python -m kinase_library.modules.sequences TABLE --protein-col NAME
    --pos-col NAME`.

    Parameters
    ----------
    argv : list, optional
        Arguments to parse. The default is None, meaning `sys.argv[1:]`.

    Returns
    -------
    int
        0 when every row resolved (or under `--allow-missing`), 1 when some did
        not or `--strict` stopped, 2 when the invocation itself was wrong.

    """

    parser = argparse.ArgumentParser(
        prog='python -m kinase_library.modules.sequences',
        description='Build 15-mer windows from a protein identifier plus a position.',
        epilog='Note: this pays the package import (about 11 s), which loads the '
               'scored phosphoproteome. For several tables in a row, import '
               'kinase_library once and call kl.retrieve_sequences instead.')
    parser.add_argument('table', help='csv/tsv/xlsx of phosphosites')
    parser.add_argument('--protein-col', required=True,
                        help='column holding the accession, gene symbol or entry name')
    parser.add_argument('--pos-col', required=True,
                        help='column holding the position (33, S33, pY1068 ...)')
    parser.add_argument('--out', default=None,
                        help='output path (default: <table stem>_windows.tsv)')
    parser.add_argument('--fasta', default=None,
                        help='reference proteome to use instead of the configured one')
    parser.add_argument('--species', default=_global_vars.default_species,
                        choices=sorted(_global_vars.valid_species),
                        help='which reference proteome (ignored when --fasta given)')
    parser.add_argument('--id-format', default='uniprot_acc',
                        choices=sorted(_global_vars.valid_id_format),
                        help='how --protein-col matches the FASTA headers')
    parser.add_argument('--sep', default=None,
                        help='delimiter (default: from the file extension)')
    parser.add_argument('--header', type=int, default=0,
                        help='header row index (default 0)')
    parser.add_argument('--allow-missing', action='store_true',
                        help='exit 0 even when some rows are unresolved')
    parser.add_argument('--strict', action='store_true',
                        help='stop at the first unresolved row')
    args = parser.parse_args(argv)

    # Everything cheap first: parsing the proteome is the slow step, and a
    # misconfiguration must not cost the user that wait.
    try:
        data = _read_table(args.table, args.sep, args.header)
    except FileNotFoundError:
        print(f'ERROR  table not found: {args.table}', file=sys.stderr)
        return(2)
    except Exception as error:                     # unreadable, bad --header, ...
        print(f'ERROR  could not read {args.table}: {type(error).__name__}: {error}',
              file=sys.stderr)
        return(2)

    missing = [c for c in (args.protein_col, args.pos_col) if c not in data.columns]
    if missing:
        print(f'ERROR  column(s) not in the table: {", ".join(missing)}\n'
              f'       columns present: {", ".join(map(str, data.columns))}',
              file=sys.stderr)
        return(2)

    try:
        fasta_path = _proteome_file(species=args.species, ref_prot_file=args.fasta)
    except (FileNotFoundError, ValueError) as error:
        print(f'ERROR  {error}\n'
              '       pass --fasta PATH to point at a proteome directly, or --species '
              f'({", ".join(sorted(_global_vars.valid_species))}) to pick one from the '
              'configured directory.', file=sys.stderr)
        return(2)

    out_path = args.out or (os.path.splitext(args.table)[0] + '_windows.tsv')
    unresolved_path = os.path.splitext(out_path)[0] + '_unresolved.tsv'

    print(f'table    : {args.table}  ({len(data)} rows)')
    print(f'proteome : {fasta_path}')
    print('parsing reference proteome ...')
    proteome = load_proteome(fasta_path, args.id_format)
    print(f'           {len(proteome)} entries indexed by {args.id_format}')

    try:
        windowed, unresolved = _resolve_sites(data, args.protein_col, args.pos_col,
                                              proteome, strict=args.strict)
    except SiteResolutionError as error:
        print(f'ERROR  {error}\n'
              '       drop --strict to get every row back with its reason instead.',
              file=sys.stderr)
        return(1)

    windowed.to_csv(out_path, sep='\t', index=False)
    resolved = len(windowed) - len(unresolved)
    print(f'\nresolved   : {resolved} of {len(windowed)}  ->  {out_path}')
    if len(unresolved):
        unresolved.to_csv(unresolved_path, sep='\t', index=False)
        print(f'unresolved : {len(unresolved)}  ->  {unresolved_path}')
        for reason, count in unresolved['reason'].value_counts().items():
            print(f'    {count:>6}  {reason}')
        print('\nThose rows are a deliverable, not a nuisance: report the count '
              'and the reasons alongside the analysis.')
        return(0 if args.allow_missing else 1)
    print('unresolved : 0')
    return(0)


if __name__ == '__main__':
    raise SystemExit(main())
