"""Tests for `kinase_library.modules.sequences`.

Plan: `src/tests/TEST-PLAN-sequences.md`. Three categories, marked with
`logic`, `edge` and `expected_failure`.

The failure these tests exist to catch is a window that is off by one, or built
from the wrong isoform, being scored silently. Nothing downstream can detect
that - a wrong 15-mer is still a valid 15-mer - so the checks have to happen
here.

Two of them carry most of the weight:

* `test_legacy_equivalence_on_every_fixture_row` runs the ORIGINAL
  `retrieve_sequences` beside the package function. The analyses already
  published rest on the legacy values, so any divergence is a defect in the
  lift. The legacy function is exec'd from source rather than imported -
  importing the legacy package would pull in logomaker, sklearn and matplotlib
  and write bytecode into a git repository.
* `test_gene_symbol_prefers_the_canonical_record` guards the `id_format`
  regression: the bundled legacy copy branches on `'gene_name'`, so the
  documented `'gene_symbol'` fell through to accession matching and matched
  nothing. See `test_red_first_proof_gene_symbol_fix` for the proof it bites.

Everything runs against `data/inputs/uniprot_test_proteome.fasta`, a committed
subset of the July-2021 human proteome, so the suite needs no file outside the
repository. The one exception is the legacy-equivalence test, which needs the
legacy source and skips with a reason where it is absent.
"""
import os
import shutil

import pandas as pd
import pytest

import kinase_library as kl
from kinase_library.modules import sequences

LEGACY_UTILS = ("/Users/tomer/My Drive/PhD/Thesis/Kinome/packages/py_kinase_library/"
                "src/py_kinase_library/utils/utils.py")


#%% Fixtures and helpers


@pytest.fixture
def restore_proteome_dir():
    """Put the session's proteome directory back, whatever a test did to it.

    The value is captured and reassigned directly rather than by calling
    `reset_current_proteome_dir`, because the reset reads the environment and a
    test that monkeypatched it would otherwise leave the wrong value behind
    depending on fixture teardown order.
    """
    saved = kl._global_vars.proteome_dir
    yield
    kl._global_vars.proteome_dir = saved


@pytest.fixture(scope="module")
def rows(windows_fixture_path):
    """The oracle fixture: expected windows computed without this module."""
    return pd.read_csv(windows_fixture_path, sep="\t", dtype=str)


@pytest.fixture(scope="module")
def proteome_gene(test_proteome_path):
    return kl.load_proteome(test_proteome_path, "gene_symbol")


@pytest.fixture(scope="module")
def built(rows, test_proteome_path):
    """Every fixture row put through the module, once."""
    return kl.retrieve_sequences(rows, "acc", "pos",
                                 ref_prot_file=test_proteome_path)[0]


def _one(acc, site, proteome=None, id_format="uniprot_acc"):
    """(window, reason) for a single site."""
    frame = pd.DataFrame({"a": [acc], "p": [site]})
    out, unresolved = kl.retrieve_sequences(frame, "a", "p",
                                            ref_prot_file=proteome,
                                            id_format=id_format)
    return (out[sequences.WINDOW_COL].iloc[0],
            unresolved["reason"].iloc[0] if len(unresolved) else None)


#%% Logic


@pytest.mark.logic
def test_p53_golden_case(test_proteome_path):
    """P04637/15 is the ATM/ATR site PSVEPPLSQETFSDL."""
    window, reason = _one("P04637", 15, test_proteome_path)
    assert reason is None
    assert window == "PSVEPPLSQETFSDL"


@pytest.mark.logic
def test_every_fixture_row_matches_its_independent_oracle(rows, built):
    """The module agrees with a window computed by a different route."""
    mismatches = [
        (a, p, want, got)
        for a, p, want, got in zip(rows["acc"], rows["pos"],
                                   rows["expected_15mer"],
                                   built[sequences.WINDOW_COL])
        if want != got
    ]
    assert not mismatches, f"{len(mismatches)} of {len(rows)} differ: {mismatches[:5]}"


@pytest.mark.logic
@pytest.mark.skipif(not os.path.isfile(LEGACY_UTILS),
                    reason=f"legacy source not on this machine: {LEGACY_UTILS}")
def test_legacy_equivalence_on_every_fixture_row(rows, built, test_proteome_path):
    """The original `retrieve_sequences` produces the same windows.

    Published analyses rest on the legacy values, so a divergence is a defect
    in the lift, not a difference of opinion.
    """
    legacy = _load_legacy_retrieve_sequences()
    site_data = rows[["acc", "pos"]].copy()
    site_data["pos"] = site_data["pos"].astype(int)

    legacy_out = legacy(site_data, protein_column="acc", pos_column="pos",
                        ref_prot_file=test_proteome_path, ignore_missing=True)

    assert legacy_out is not None and len(legacy_out) == len(rows), (
        "the legacy function dropped rows the package resolved")

    ours = built[sequences.WINDOW_COL].tolist()
    theirs = legacy_out["SITE_+/-7_AA"].tolist()
    mismatches = [(a, p, t, o) for a, p, t, o
                  in zip(rows["acc"], rows["pos"], theirs, ours) if t != o]
    assert not mismatches, f"{len(mismatches)} rows differ from legacy: {mismatches[:5]}"


@pytest.mark.logic
def test_n_terminus_pads_on_the_left(rows, built):
    """A site at or below position 7 gets leading underscores, acceptor at 7."""
    subset = built[rows["kind"].values == "n_terminus"]
    assert len(subset) >= 5
    for window in subset[sequences.WINDOW_COL]:
        assert window.startswith("_")
        assert len(window) == 15 and window[7] in kl._global_vars.valid_phos_res


@pytest.mark.logic
def test_c_terminus_pads_on_the_right(rows, built):
    """A site within 7 of the end gets trailing underscores, same centring."""
    subset = built[rows["kind"].values == "c_terminus"]
    assert len(subset) >= 5
    for window in subset[sequences.WINDOW_COL]:
        assert window.endswith("_")
        assert len(window) == 15 and window[7] in kl._global_vars.valid_phos_res


@pytest.mark.logic
def test_isoform_suffix_resolves_to_the_isoform(rows, built):
    """`ACC-2` gives the isoform's window; the bare accession gives another."""
    pairs = rows[rows["kind"].isin(["isoform", "isoform_canonical_twin"])]
    assert len(pairs) >= 4
    windows = built.loc[pairs.index, sequences.WINDOW_COL].tolist()
    accessions = pairs["acc"].tolist()
    for i in range(0, len(pairs) - 1, 2):
        assert "-" in accessions[i] and "-" not in accessions[i + 1]
        assert windows[i] != windows[i + 1], (
            f"{accessions[i]} and {accessions[i+1]} gave the same window - the "
            "isoform suffix is being collapsed")


@pytest.mark.logic
def test_gene_symbol_matches_the_accession_route(test_proteome_path):
    """The same site by gene symbol and by accession gives one answer."""
    by_acc, _ = _one("P04637", 33, test_proteome_path)
    by_gene, _ = _one("TP53", 33, test_proteome_path, id_format="gene_symbol")
    assert by_gene == by_acc == "LPENNVLSPLPSQAM"


@pytest.mark.logic
def test_gene_symbol_prefers_the_canonical_record(proteome_gene):
    """A gene with several isoform records resolves to the canonical one.

    Regression: the bundled legacy copy branched on `'gene_name'`, so the
    documented `'gene_symbol'` matched nothing at all.
    """
    for gene in ("MAPK1", "TP53", "EGFR"):
        name = proteome_gene.at[gene, "NAME"]
        accession = name.split("|")[1]
        assert "-" not in accession, f"{gene} resolved to isoform {name}"
        assert name.startswith("sp|")


@pytest.mark.logic
def test_gene_symbol_prefers_swissprot_over_trembl(tmp_path):
    """Swiss-Prot canonical wins even when TrEMBL comes first in the file.

    The records are written in a deliberately adverse order, so a `keep='first'`
    de-duplication - what the legacy code does - picks the wrong one.
    """
    fasta = tmp_path / "custom.fasta"
    fasta.write_text(
        ">tr|Q9AAA1|Q9AAA1_HUMAN Fake trembl OS=Homo sapiens OX=9606 GN=FAKEGENE PE=1 SV=1\n"
        "MSTAAAAAAAAAAAAAAAAAAAA\n"
        ">sp|Q9BBB2-2|FAKE_HUMAN Isoform 2 of Fake OS=Homo sapiens OX=9606 GN=FAKEGENE\n"
        "MSTBBBBBBBBBBBBBBBBBBBB\n"
        ">sp|Q9BBB2|FAKE_HUMAN Fake protein OS=Homo sapiens OX=9606 GN=FAKEGENE PE=1 SV=1\n"
        "MSTCCCCCCCCCCCCCCCCCCCC\n"
        ">tr|Q9AAA1-2|Q9AAA1_HUMAN Isoform of fake trembl OS=Homo sapiens OX=9606 GN=FAKEGENE\n"
        "MSTDDDDDDDDDDDDDDDDDDDD\n")
    proteome = kl.load_proteome(str(fasta), "gene_symbol")
    assert proteome.at["FAKEGENE", "NAME"] == "sp|Q9BBB2|FAKE_HUMAN"


@pytest.mark.logic
def test_protein_name_id_format_resolves(test_proteome_path):
    """`id_format='protein_name'` matches the UniProt entry-name prefix."""
    window, reason = _one("P53", 15, test_proteome_path, id_format="protein_name")
    assert reason is None
    assert window == "PSVEPPLSQETFSDL"


@pytest.mark.logic
@pytest.mark.parametrize("raw,expected", [
    (33, (None, 33)), ("33", (None, 33)), ("S33", ("S", 33)),
    ("s33", ("S", 33)), ("Ser33", ("S", 33)), ("SER33", ("S", 33)),
    ("Y1068", ("Y", 1068)), ("pS33", ("S", 33)), ("pY1068", ("Y", 1068)),
    ("S-33", ("S", 33)), ("T_33", ("T", 33)), ("  S33 ", ("S", 33)),
    ("33.0", (None, 33)),
])
def test_parse_position_across_every_real_form(raw, expected):
    """Every site spelling seen in collaborator tables parses identically."""
    assert kl.parse_position(raw) == expected


@pytest.mark.logic
def test_input_frame_is_never_mutated(test_proteome_path):
    """The caller's frame is unchanged and gains no window column."""
    frame = pd.DataFrame({"a": ["P04637"], "p": [15]})
    snapshot = frame.copy()
    kl.retrieve_sequences(frame, "a", "p", ref_prot_file=test_proteome_path)
    assert frame.equals(snapshot)
    assert sequences.WINDOW_COL not in frame.columns


@pytest.mark.logic
def test_output_preserves_input_row_order_and_row_count(test_proteome_path):
    """Unresolved rows are kept in place so the table stays joinable."""
    frame = pd.DataFrame({"a": ["P04637", "XXNOTREAL9", "P28482"],
                          "p": ["S15", "S33", "Y187"]})
    out, unresolved = kl.retrieve_sequences(frame, "a", "p",
                                            ref_prot_file=test_proteome_path)
    assert len(out) == 3 and list(out["a"]) == list(frame["a"])
    assert out[sequences.WINDOW_COL].iloc[0] == "PSVEPPLSQETFSDL"
    assert out[sequences.WINDOW_COL].iloc[1] == ""
    assert out[sequences.WINDOW_COL].iloc[2] == "HTGFLTEYVATRWYR"
    assert len(unresolved) == 1


@pytest.mark.logic
def test_window_is_always_fifteen_with_the_acceptor_at_index_seven(built):
    """The invariant that makes a window scoreable at all."""
    for window in built[sequences.WINDOW_COL]:
        assert len(window) == 15
        assert window[7] in kl._global_vars.valid_phos_res


#%% Proteome directory


@pytest.mark.logic
def test_proteome_dir_accessors_round_trip(restore_proteome_dir, tmp_path):
    """set_current_proteome_dir then get returns it; reset restores the default."""
    default = kl.get_current_proteome_dir()
    kl.set_current_proteome_dir(str(tmp_path))
    assert kl.get_current_proteome_dir() == str(tmp_path)
    kl.reset_current_proteome_dir()
    assert kl.get_current_proteome_dir() == default


@pytest.mark.logic
def test_proteome_dir_defaults_to_the_legacy_directory(restore_proteome_dir,
                                                       monkeypatch):
    """With KL_PROTEOME_DIR unset, the bundled legacy path is the default."""
    monkeypatch.delenv("KL_PROTEOME_DIR", raising=False)
    kl.reset_current_proteome_dir()
    assert kl.get_current_proteome_dir() == kl._global_vars._legacy_proteome_dir


@pytest.mark.logic
def test_proteome_dir_reads_the_environment_variable(restore_proteome_dir,
                                                     monkeypatch, tmp_path):
    """KL_PROTEOME_DIR set then reset makes that directory current."""
    monkeypatch.setenv("KL_PROTEOME_DIR", str(tmp_path))
    kl.reset_current_proteome_dir()
    assert kl.get_current_proteome_dir() == str(tmp_path)

    monkeypatch.delenv("KL_PROTEOME_DIR")
    kl.reset_current_proteome_dir()
    assert kl.get_current_proteome_dir() == kl._global_vars._legacy_proteome_dir


@pytest.mark.logic
def test_species_resolves_dateless_before_dated(restore_proteome_dir, tmp_path,
                                                test_proteome_path):
    """A current download shadows the July-2021 file of the same species."""
    dated = tmp_path / "human_uniprot_proteome_with_isoforms_07_2021.fasta"
    dateless = tmp_path / "human_uniprot_proteome_with_isoforms.fasta"
    kl.set_current_proteome_dir(str(tmp_path))

    shutil.copy(test_proteome_path, dated)
    window, reason = _one("P04637", 15)
    assert window == "PSVEPPLSQETFSDL", "the dated file alone must be found"

    # A decoy under the dateless name: if it is chosen, p53 stops resolving.
    # That is a sharper proof of precedence than comparing two working files.
    dateless.write_text(">sp|Q0DECOY|DECOY_HUMAN Decoy OS=Homo sapiens OX=9606 GN=DECOY\n"
                        "MSTAAAAAAAAAA\n")
    window, reason = _one("P04637", 15)
    assert window == "" and reason == sequences.REASON_NOT_IN_FASTA, (
        "the dateless file must take precedence over the dated one")

    # And with only the dateless name present, holding the real records.
    dated.unlink()
    shutil.copy(test_proteome_path, dateless)
    window, reason = _one("P04637", 15)
    assert window == "PSVEPPLSQETFSDL"


#%% Edge cases


@pytest.mark.edge
def test_position_one_is_all_left_padding(test_proteome_path):
    """The first residue resolves when it is an acceptor."""
    window, reason = _one("P31749", 2, test_proteome_path)      # AKT1 S2
    assert reason is None
    assert window.startswith("______") and window[7] == "S"


@pytest.mark.edge
def test_position_equals_protein_length_resolves(rows, built):
    """The last residue is reachable, with seven underscores after it."""
    subset = built[rows["kind"].values == "c_terminus"]
    assert any(w.endswith("_______") for w in subset[sequences.WINDOW_COL]), (
        "no fixture row sits exactly at the C-terminus")


@pytest.mark.edge
def test_float_position_from_a_nan_column(test_proteome_path):
    """`33.0` is position 33 - pandas floats a column with one NaN in it."""
    frame = pd.DataFrame({"a": ["P04637", "P04637"], "p": [15.0, float("nan")]})
    out, unresolved = kl.retrieve_sequences(frame, "a", "p",
                                            ref_prot_file=test_proteome_path)
    assert out[sequences.WINDOW_COL].iloc[0] == "PSVEPPLSQETFSDL"
    assert unresolved["reason"].iloc[0] == sequences.REASON_UNPARSEABLE


@pytest.mark.edge
def test_whitespace_around_the_site_string(test_proteome_path):
    """A padded cell resolves rather than failing to parse."""
    window, reason = _one("P04637", "  S15 ", test_proteome_path)
    assert reason is None and window == "PSVEPPLSQETFSDL"


@pytest.mark.edge
def test_duplicate_rows_resolve_identically(test_proteome_path):
    """The same site twice gives the same window twice, both rows kept."""
    frame = pd.DataFrame({"a": ["P04637"] * 2, "p": ["S15", "S15"]})
    out, unresolved = kl.retrieve_sequences(frame, "a", "p",
                                            ref_prot_file=test_proteome_path)
    assert len(out) == 2 and len(unresolved) == 0
    assert out[sequences.WINDOW_COL].nunique() == 1


@pytest.mark.edge
def test_empty_table_gives_empty_output_and_no_misses(test_proteome_path):
    """A header-only table is not an error."""
    frame = pd.DataFrame({"a": [], "p": []})
    out, unresolved = kl.retrieve_sequences(frame, "a", "p",
                                            ref_prot_file=test_proteome_path)
    assert len(out) == 0 and len(unresolved) == 0
    assert sequences.WINDOW_COL in out.columns


@pytest.mark.edge
def test_lowercase_accession_is_reported_not_silently_matched(test_proteome_path):
    """UniProt accessions are case-sensitive; a miss is better than a guess."""
    window, reason = _one("p04637", 15, test_proteome_path)
    assert window == "" and reason == sequences.REASON_NOT_IN_FASTA


@pytest.mark.edge
def test_records_without_a_gene_symbol_are_skipped_not_indexed_as_nan(
        test_proteome_path, proteome_gene):
    """845 of the 100,100 human headers carry no `GN=`; they must drop out."""
    all_records = kl.load_proteome(test_proteome_path, "uniprot_acc")
    assert len(proteome_gene) < len(all_records)
    assert proteome_gene.index.notna().all()
    assert "Q4G0T1" not in proteome_gene.index      # a header with no GN=


#%% Expected failures


@pytest.mark.expected_failure
def test_accession_absent_from_fasta_is_reported(test_proteome_path):
    """Reason `accession_not_in_fasta`, row kept with an empty window."""
    window, reason = _one("XXNOTREAL9", 33, test_proteome_path)
    assert window == "" and reason == sequences.REASON_NOT_IN_FASTA


@pytest.mark.expected_failure
def test_position_beyond_protein_length_is_reported(test_proteome_path):
    """Reason `position_out_of_range`, not a short or padded window."""
    _, reason = _one("P04637", "S9999", test_proteome_path)
    assert reason == sequences.REASON_OUT_OF_RANGE


@pytest.mark.expected_failure
def test_unparseable_site_string_is_reported(test_proteome_path):
    """Reason `position_unparseable` rather than a guessed position."""
    for value in ("abc", "", "S", None):
        _, reason = _one("P04637", value, test_proteome_path)
        assert reason == sequences.REASON_UNPARSEABLE, f"{value!r} -> {reason}"


@pytest.mark.expected_failure
def test_non_phosphoacceptor_centre_is_reported(test_proteome_path):
    """Reason `central_residue_not_STY` - p53 position 1 is methionine."""
    _, reason = _one("P04637", 1, test_proteome_path)
    assert reason == sequences.REASON_NOT_STY


@pytest.mark.expected_failure
def test_stated_residue_disagreeing_with_the_sequence_is_reported(test_proteome_path):
    """`T33` where position 33 is `S`: the centre IS an acceptor, so only the
    residue check catches it. This is the off-by-one guard."""
    _, reason = _one("P04637", "T33", test_proteome_path)
    assert reason == sequences.REASON_MISMATCH


@pytest.mark.expected_failure
def test_strict_raises_on_the_first_miss(test_proteome_path):
    """`SiteResolutionError` naming the identifier and the reason."""
    frame = pd.DataFrame({"a": ["XXNOTREAL9"], "p": [33]})
    with pytest.raises(kl.SiteResolutionError) as exc:
        kl.retrieve_sequences(frame, "a", "p", ref_prot_file=test_proteome_path,
                              strict=True)
    assert "XXNOTREAL9" in str(exc.value)
    assert sequences.REASON_NOT_IN_FASTA in str(exc.value)


@pytest.mark.expected_failure
def test_retrieve_sequences_raises_KeyError_naming_an_absent_column(test_proteome_path):
    """The library-level counterpart of the CLI's column check."""
    frame = pd.DataFrame({"a": ["P04637"], "p": [15]})
    with pytest.raises(KeyError) as exc:
        kl.retrieve_sequences(frame, "nope", "p", ref_prot_file=test_proteome_path)
    assert "nope" in str(exc.value)


@pytest.mark.expected_failure
def test_retrieve_sequences_raises_TypeError_on_a_non_dataframe(test_proteome_path):
    """A list of dicts is a plausible mistake and must be named."""
    with pytest.raises(TypeError, match="DataFrame"):
        kl.retrieve_sequences([{"a": "P04637", "p": 15}], "a", "p",
                              ref_prot_file=test_proteome_path)


@pytest.mark.expected_failure
def test_invalid_species_raises_ValueError_listing_the_valid_set():
    """An unknown species is named, not silently fallen through to all-species."""
    frame = pd.DataFrame({"a": ["P04637"], "p": [15]})
    with pytest.raises(ValueError, match="species") as exc:
        kl.retrieve_sequences(frame, "a", "p", species="cat")
    assert "human" in str(exc.value)


@pytest.mark.expected_failure
def test_invalid_id_format_raises_ValueError_listing_the_valid_set(test_proteome_path):
    """The exact fall-through that made the legacy gene_symbol match nothing."""
    frame = pd.DataFrame({"a": ["P04637"], "p": [15]})
    with pytest.raises(ValueError, match="id_format") as exc:
        kl.retrieve_sequences(frame, "a", "p", ref_prot_file=test_proteome_path,
                              id_format="gene")
    message = str(exc.value)
    assert "uniprot_acc" in message
    assert "gene_symbol" in message
    assert "protein_name" in message


@pytest.mark.expected_failure
def test_missing_proteome_file_raises_FileNotFoundError_naming_the_recovery(
        restore_proteome_dir, tmp_path):
    """The message must say where it looked and every way to fix it."""
    kl.set_current_proteome_dir(str(tmp_path))
    frame = pd.DataFrame({"a": ["P04637"], "p": [15]})
    with pytest.raises(FileNotFoundError) as exc:
        kl.retrieve_sequences(frame, "a", "p")
    message = str(exc.value)
    assert str(tmp_path) in message
    assert "human_uniprot_proteome_with_isoforms.fasta" in message
    assert "human_uniprot_proteome_with_isoforms_07_2021.fasta" in message
    assert "KL_PROTEOME_DIR" in message
    assert "set_current_proteome_dir" in message
    assert "ref_prot_file" in message


#%% drop_unresolved, and the guards that stop rows vanishing quietly


@pytest.mark.logic
def test_drop_unresolved_returns_the_legacy_single_frame(test_proteome_path):
    """One frame of resolved rows only, with the input's index preserved."""
    frame = pd.DataFrame({"a": ["P04637", "XXNOTREAL9", "P28482"],
                          "p": ["S15", "S33", "Y187"]})
    tuple_out, _ = kl.retrieve_sequences(frame, "a", "p",
                                         ref_prot_file=test_proteome_path)
    with pytest.warns(UserWarning):
        single = kl.retrieve_sequences(frame, "a", "p",
                                       ref_prot_file=test_proteome_path,
                                       drop_unresolved=True)

    assert isinstance(single, pd.DataFrame)
    assert len(single) == 2
    assert list(single.index) == [0, 2], "the original index must survive"
    assert single[sequences.WINDOW_COL].tolist() == [
        tuple_out[sequences.WINDOW_COL].iloc[0],
        tuple_out[sequences.WINDOW_COL].iloc[2]]


@pytest.mark.logic
def test_partial_miss_warns_with_counts_per_reason(test_proteome_path):
    """Dropping rows is never silent: the warning names how many and why."""
    frame = pd.DataFrame({"a": ["P04637", "XXNOTREAL9", "P04637"],
                          "p": ["S15", "S33", "S9999"]})
    with pytest.warns(UserWarning) as record:
        kl.retrieve_sequences(frame, "a", "p", ref_prot_file=test_proteome_path,
                              drop_unresolved=True)
    message = str(record[0].message)
    assert "2" in message
    assert sequences.REASON_NOT_IN_FASTA in message
    assert sequences.REASON_OUT_OF_RANGE in message


@pytest.mark.edge
def test_drop_unresolved_on_an_empty_table_is_not_an_error(test_proteome_path):
    """An empty input yields an empty frame, no warning and no exception."""
    import warnings

    frame = pd.DataFrame({"a": [], "p": []})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = kl.retrieve_sequences(frame, "a", "p",
                                    ref_prot_file=test_proteome_path,
                                    drop_unresolved=True)
    assert len(out) == 0 and sequences.WINDOW_COL in out.columns


@pytest.mark.expected_failure
def test_drop_unresolved_with_nothing_resolved_raises(test_proteome_path):
    """A frame where every row failed is a mistake, not a result."""
    frame = pd.DataFrame({"a": ["XXNOTREAL9", "XXALSOFAKE"], "p": ["S33", "S1"]})
    with pytest.raises(kl.SiteResolutionError) as exc:
        kl.retrieve_sequences(frame, "a", "p", ref_prot_file=test_proteome_path,
                              drop_unresolved=True)
    message = str(exc.value)
    assert "2" in message
    assert sequences.REASON_NOT_IN_FASTA in message


@pytest.mark.expected_failure
def test_strict_wins_over_drop_unresolved(test_proteome_path):
    """`strict` raises at the first miss, before any warning is issued."""
    import warnings

    frame = pd.DataFrame({"a": ["P04637", "XXNOTREAL9"], "p": ["S15", "S33"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(kl.SiteResolutionError, match="XXNOTREAL9"):
            kl.retrieve_sequences(frame, "a", "p",
                                  ref_prot_file=test_proteome_path,
                                  strict=True, drop_unresolved=True)


#%% Deprecated argument spellings from the legacy call sites


@pytest.mark.logic
def test_id_column_alias_matches_the_modern_spelling(test_proteome_path):
    """`id_column` is the older name for `protein_column`."""
    frame = pd.DataFrame({"a": ["P04637"], "p": ["S15"]})
    expected, _ = kl.retrieve_sequences(frame, "a", "p",
                                        ref_prot_file=test_proteome_path)
    with pytest.warns(DeprecationWarning, match="id_column"):
        out, _ = kl.retrieve_sequences(frame, id_column="a", pos_column="p",
                                       ref_prot_file=test_proteome_path)
    assert out[sequences.WINDOW_COL].tolist() == expected[sequences.WINDOW_COL].tolist()


@pytest.mark.logic
def test_protein_position_column_alias_matches_the_modern_spelling(test_proteome_path):
    """`protein_position_column` is the older name for `pos_column`."""
    frame = pd.DataFrame({"a": ["P04637"], "p": ["S15"]})
    with pytest.warns(DeprecationWarning, match="protein_position_column"):
        out, _ = kl.retrieve_sequences(frame, protein_column="a",
                                       protein_position_column="p",
                                       ref_prot_file=test_proteome_path)
    assert out[sequences.WINDOW_COL].iloc[0] == "PSVEPPLSQETFSDL"


@pytest.mark.logic
def test_legacy_corpus_call_with_phospho_data_keyword(test_proteome_path):
    """A real corpus call shape, verbatim apart from the proteome path.

    `kl.retrieve_sequences(phospho_data=..., species=..., id_column=...,
    protein_position_column=...)` appears six times in the published Ser/Thr
    analysis scripts and names the frame by a third spelling again.
    """
    data_sumi = pd.DataFrame({"accession": ["P04637"],
                              "Positions within proteins": ["S15"]})
    with pytest.warns(DeprecationWarning):
        out, unresolved = kl.retrieve_sequences(
            phospho_data=data_sumi, species="human", id_column="accession",
            protein_position_column="Positions within proteins",
            ref_prot_file=test_proteome_path)
    assert out[sequences.WINDOW_COL].iloc[0] == "PSVEPPLSQETFSDL"
    assert len(unresolved) == 0


@pytest.mark.logic
def test_ignore_missing_true_restores_the_legacy_single_frame(test_proteome_path):
    """114 corpus calls assign one variable; they must keep working."""
    frame = pd.DataFrame({"a": ["P04637", "XXNOTREAL9"], "p": ["S15", "S33"]})
    with pytest.warns(DeprecationWarning, match="ignore_missing"):
        out = kl.retrieve_sequences(frame, "a", "p", ignore_missing=True,
                                    ref_prot_file=test_proteome_path)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 1
    assert out[sequences.WINDOW_COL].iloc[0] == "PSVEPPLSQETFSDL"


@pytest.mark.logic
def test_id_format_gene_name_is_accepted_as_a_deprecated_alias(test_proteome_path):
    """Six live corpus calls pass it; it means `gene_symbol`."""
    frame = pd.DataFrame({"g": ["TP53"], "p": ["S33"]})
    with pytest.warns(DeprecationWarning, match="gene_name"):
        out, _ = kl.retrieve_sequences(frame, "g", "p", id_format="gene_name",
                                       ref_prot_file=test_proteome_path)
    assert out[sequences.WINDOW_COL].iloc[0] == "LPENNVLSPLPSQAM"


@pytest.mark.logic
def test_legacy_corpus_call_with_gene_name_and_ignore_missing(test_proteome_path):
    """The other real corpus shape, with both deprecated spellings at once."""
    data_zhuang = pd.DataFrame({"gene": ["TP53", "NOTAGENE"], "pos": ["S33", "S1"]})
    with pytest.warns(DeprecationWarning):
        out = kl.retrieve_sequences(data_zhuang, species="human",
                                    protein_column="gene", pos_column="pos",
                                    ignore_missing=True, id_format="gene_name",
                                    ref_prot_file=test_proteome_path)
    assert isinstance(out, pd.DataFrame)
    assert out[sequences.WINDOW_COL].tolist() == ["LPENNVLSPLPSQAM"]


@pytest.mark.expected_failure
def test_ignore_missing_false_raises_on_the_first_miss(test_proteome_path):
    """The legacy `ignore_missing=False` printed and returned None; now it raises."""
    frame = pd.DataFrame({"a": ["XXNOTREAL9"], "p": ["S33"]})
    with pytest.warns(DeprecationWarning):
        with pytest.raises(kl.SiteResolutionError, match="XXNOTREAL9"):
            kl.retrieve_sequences(frame, "a", "p", ignore_missing=False,
                                  ref_prot_file=test_proteome_path)


@pytest.mark.expected_failure
@pytest.mark.parametrize("deprecated,modern", [
    ("id_column", "protein_column"),
    ("protein_position_column", "pos_column"),
])
def test_deprecated_alias_with_its_replacement_raises_TypeError(
        deprecated, modern, test_proteome_path):
    """Passing both spellings is ambiguous and must be refused by name."""
    frame = pd.DataFrame({"a": ["P04637"], "p": ["S15"]})
    kwargs = {deprecated: "a", modern: "a",
              "ref_prot_file": test_proteome_path}
    kwargs.setdefault("pos_column", "p")
    with pytest.raises(TypeError) as exc:
        kl.retrieve_sequences(frame, **kwargs)
    assert deprecated in str(exc.value) and modern in str(exc.value)


@pytest.mark.expected_failure
def test_phospho_data_with_site_data_raises_TypeError(test_proteome_path):
    """The same ambiguity for the frame itself."""
    frame = pd.DataFrame({"a": ["P04637"], "p": ["S15"]})
    with pytest.raises(TypeError) as exc:
        kl.retrieve_sequences(frame, "a", "p", phospho_data=frame,
                              ref_prot_file=test_proteome_path)
    assert "phospho_data" in str(exc.value) and "site_data" in str(exc.value)


@pytest.mark.expected_failure
def test_peptide_column_raises_TypeError_naming_the_unsupported_mode(test_proteome_path):
    """Ten corpus calls pass it; no released implementation ever accepted it."""
    frame = pd.DataFrame({"a": ["P04637"], "peptide": ["SQETFSDL"]})
    with pytest.raises(TypeError) as exc:
        kl.retrieve_sequences(frame, protein_column="a", pos_column=None,
                              peptide_column="peptide",
                              ref_prot_file=test_proteome_path)
    assert "peptide_column" in str(exc.value)


@pytest.mark.expected_failure
def test_an_unknown_keyword_still_raises_the_normal_TypeError(test_proteome_path):
    """Accepting aliases must not turn every typo into a silent no-op."""
    frame = pd.DataFrame({"a": ["P04637"], "p": ["S15"]})
    with pytest.raises(TypeError, match="nonsense_column"):
        kl.retrieve_sequences(frame, "a", "p", nonsense_column="x",
                              ref_prot_file=test_proteome_path)


#%% Command line


def _table(tmp_path, text="acc\tpos\nP04637\t15\n"):
    path = tmp_path / "t.tsv"
    path.write_text(text)
    return path


@pytest.mark.logic
def test_cli_resolves_everything_and_exits_zero(tmp_path, capsys, test_proteome_path):
    """Exit 0, the windowed table written with the window column."""
    table = _table(tmp_path, "acc\tpos\nP04637\tS15\nP28482\tY187\n")
    code = sequences.main([str(table), "--protein-col", "acc", "--pos-col", "pos",
                           "--fasta", test_proteome_path])
    captured = capsys.readouterr()
    assert code == 0
    out_path = tmp_path / "t_windows.tsv"
    assert out_path.is_file()
    written = pd.read_csv(out_path, sep="\t")
    assert written[sequences.WINDOW_COL].tolist() == ["PSVEPPLSQETFSDL",
                                                      "HTGFLTEYVATRWYR"]
    assert "unresolved : 0" in captured.out
    assert not (tmp_path / "t_windows_unresolved.tsv").exists()


@pytest.mark.logic
def test_cli_reports_unresolved_rows_and_exits_one(tmp_path, capsys,
                                                   test_proteome_path):
    """Exit 1, an unresolved file with reasons, and the counts printed."""
    table = _table(tmp_path, "acc\tpos\nP04637\tS15\nXXNOTREAL9\tS33\n")
    code = sequences.main([str(table), "--protein-col", "acc", "--pos-col", "pos",
                           "--fasta", test_proteome_path])
    captured = capsys.readouterr()
    assert code == 1
    unresolved_path = tmp_path / "t_windows_unresolved.tsv"
    assert unresolved_path.is_file()
    unresolved = pd.read_csv(unresolved_path, sep="\t")
    assert unresolved["reason"].tolist() == [sequences.REASON_NOT_IN_FASTA]
    assert sequences.REASON_NOT_IN_FASTA in captured.out
    # every input row still present in the main output, joinable
    assert len(pd.read_csv(tmp_path / "t_windows.tsv", sep="\t")) == 2


@pytest.mark.logic
def test_cli_allow_missing_exits_zero_with_unresolved_rows(tmp_path, capsys,
                                                           test_proteome_path):
    """The override says 'I know, report them and carry on'."""
    table = _table(tmp_path, "acc\tpos\nP04637\tS15\nXXNOTREAL9\tS33\n")
    code = sequences.main([str(table), "--protein-col", "acc", "--pos-col", "pos",
                           "--fasta", test_proteome_path, "--allow-missing"])
    assert code == 0
    assert (tmp_path / "t_windows_unresolved.tsv").is_file()


@pytest.mark.logic
def test_cli_runs_as_a_python_module(tmp_path, test_proteome_path):
    """One real subprocess run of `python -m kinase_library.modules.sequences`.

    Slow - it pays the package's ~11 s import - so exactly one test does it.
    Every other command line test calls `main` in process.
    """
    import subprocess
    import sys

    table = _table(tmp_path, "acc\tpos\nP04637\tS15\n")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    finished = subprocess.run(
        [sys.executable, "-m", "kinase_library.modules.sequences", str(table),
         "--protein-col", "acc", "--pos-col", "pos",
         "--fasta", test_proteome_path],
        capture_output=True, text=True, env=environment)
    assert finished.returncode == 0, finished.stderr
    assert (tmp_path / "t_windows.tsv").is_file()


@pytest.mark.edge
def test_cli_reads_an_xlsx_table(tmp_path, test_proteome_path):
    """The delimiter is inferred from the extension; Excel input is read."""
    openpyxl = pytest.importorskip("openpyxl")
    table = tmp_path / "sites.xlsx"
    pd.DataFrame({"acc": ["P04637"], "pos": ["S15"]}).to_excel(table, index=False)
    code = sequences.main([str(table), "--protein-col", "acc", "--pos-col", "pos",
                           "--fasta", test_proteome_path])
    assert code == 0
    written = pd.read_csv(tmp_path / "sites_windows.tsv", sep="\t")
    assert written[sequences.WINDOW_COL].iloc[0] == "PSVEPPLSQETFSDL"


@pytest.mark.expected_failure
def test_cli_strict_exits_one_with_the_reason_on_stderr(tmp_path, capsys,
                                                        test_proteome_path):
    """Stop at the first miss, say which row, and write nothing."""
    table = _table(tmp_path, "acc\tpos\nXXNOTREAL9\tS33\n")
    code = sequences.main([str(table), "--protein-col", "acc", "--pos-col", "pos",
                           "--fasta", test_proteome_path, "--strict"])
    captured = capsys.readouterr()
    assert code == 1
    assert "XXNOTREAL9" in captured.err
    assert sequences.REASON_NOT_IN_FASTA in captured.err
    assert not (tmp_path / "t_windows.tsv").exists()


@pytest.mark.expected_failure
def test_absent_column_exits_2_before_the_fasta_is_parsed(tmp_path, capsys,
                                                          test_proteome_path):
    """A typo must not cost a FASTA parse, and must list what is present."""
    table = _table(tmp_path)
    code = sequences.main([str(table), "--protein-col", "nope", "--pos-col", "pos",
                           "--fasta", test_proteome_path])
    captured = capsys.readouterr()
    assert code == 2
    assert "nope" in captured.err and "acc" in captured.err
    assert "parsing reference proteome" not in captured.out


@pytest.mark.expected_failure
def test_absent_table_exits_2_naming_the_path(tmp_path, capsys, test_proteome_path):
    """The table itself is the first thing that can be wrong."""
    code = sequences.main([str(tmp_path / "nope.tsv"), "--protein-col", "acc",
                           "--pos-col", "pos", "--fasta", test_proteome_path])
    captured = capsys.readouterr()
    assert code == 2
    assert "nope.tsv" in captured.err


@pytest.mark.expected_failure
def test_absent_fasta_exits_2_naming_the_path(tmp_path, capsys):
    """The message names the missing path and the override that fixes it."""
    table = _table(tmp_path)
    code = sequences.main([str(table), "--protein-col", "acc", "--pos-col", "pos",
                           "--fasta", "/nonexistent.fasta"])
    captured = capsys.readouterr()
    assert code == 2
    assert "/nonexistent.fasta" in captured.err and "--fasta" in captured.err


@pytest.mark.expected_failure
def test_bad_id_format_exits_2_listing_the_three_valid_values(tmp_path, capsys):
    """Never a silent fall-through - that is how the legacy copy failed."""
    table = _table(tmp_path)
    with pytest.raises(SystemExit) as exc:
        sequences.main([str(table), "--protein-col", "acc", "--pos-col", "pos",
                        "--id-format", "gene"])
    assert exc.value.code == 2
    message = capsys.readouterr().err
    for valid in kl._global_vars.valid_id_format:
        assert valid in message


#%% download_proteome
#
# The transport is monkeypatched in every one of these. The suite makes no
# network call: a test that reached UniProt would fail in CI, would be slow,
# and would be testing UniProt rather than this package.


class _FakeResponse:
    """Enough of an http response for `download_proteome` to stream from."""

    def __init__(self, payload, headers=None, fail_after=None):
        import io

        self._buffer = io.BytesIO(payload)
        self.headers = headers or {}
        self._fail_after = fail_after
        self._served = 0

    def read(self, size=-1):
        if self._fail_after is not None and self._served >= self._fail_after:
            raise OSError("connection reset by peer")
        chunk = self._buffer.read(size)
        self._served += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _gzipped(path):
    import gzip

    return gzip.compress(open(path, "rb").read())


@pytest.fixture
def fake_uniprot(monkeypatch, test_proteome_path):
    """Serve the committed subset proteome in place of UniProt."""
    payload = _gzipped(test_proteome_path)
    calls = {}

    def fake_urlopen(request, timeout=None):
        calls["url"] = request.full_url if hasattr(request, "full_url") else request
        calls["timeout"] = timeout
        return _FakeResponse(payload, headers={
            "X-UniProt-Release": "2026_03",
            "X-UniProt-Release-Date": "02-September-2026"})

    monkeypatch.setattr(sequences, "urlopen", fake_urlopen)
    return calls


@pytest.mark.logic
def test_download_proteome_writes_the_fasta_and_a_release_sidecar(fake_uniprot,
                                                                  tmp_path):
    """The file lands under the dateless name, with its provenance beside it."""
    path = kl.download_proteome("human", dest_dir=str(tmp_path))
    assert os.path.basename(path) == "human_uniprot_proteome_with_isoforms.fasta"
    assert os.path.isfile(path)
    assert not os.path.exists(path + ".part")

    sidecar = tmp_path / "human_uniprot_proteome_with_isoforms.release.txt"
    assert sidecar.is_file()
    text = sidecar.read_text()
    assert "2026_03" in text
    assert "UP000005640" in text or "proteome" in text
    assert "208" in text, "the record count belongs in the provenance file"
    assert "includeIsoform" in fake_uniprot["url"]


@pytest.mark.logic
def test_download_proteome_output_is_found_by_the_species_resolver(
        fake_uniprot, tmp_path, restore_proteome_dir):
    """What it writes is exactly what `species=` then finds."""
    kl.set_current_proteome_dir(str(tmp_path))
    kl.download_proteome("human")
    window, reason = _one("P04637", 15)
    assert reason is None and window == "PSVEPPLSQETFSDL"


@pytest.mark.edge
def test_download_proteome_creates_a_missing_destination_directory(fake_uniprot,
                                                                   tmp_path):
    """A fresh install has no proteome directory yet."""
    target = tmp_path / "not" / "there" / "yet"
    path = kl.download_proteome("human", dest_dir=str(target))
    assert os.path.isfile(path)


@pytest.mark.expected_failure
def test_download_proteome_refuses_to_clobber_without_overwrite(fake_uniprot,
                                                                tmp_path):
    """An existing proteome is not silently replaced."""
    existing = tmp_path / "human_uniprot_proteome_with_isoforms.fasta"
    existing.write_text("do not overwrite me\n")
    with pytest.raises(FileExistsError) as exc:
        kl.download_proteome("human", dest_dir=str(tmp_path))
    assert "overwrite" in str(exc.value)
    assert existing.read_text() == "do not overwrite me\n"

    kl.download_proteome("human", dest_dir=str(tmp_path), overwrite=True)
    assert existing.read_text() != "do not overwrite me\n"


@pytest.mark.expected_failure
def test_download_proteome_leaves_no_part_file_when_the_transport_fails(
        monkeypatch, tmp_path, test_proteome_path):
    """A half-written proteome must never be left under either name."""
    payload = _gzipped(test_proteome_path)

    def failing_urlopen(request, timeout=None):
        return _FakeResponse(payload, fail_after=64)

    monkeypatch.setattr(sequences, "urlopen", failing_urlopen)
    with pytest.raises(OSError):
        kl.download_proteome("human", dest_dir=str(tmp_path))

    assert list(tmp_path.iterdir()) == [], "a failed download left files behind"


@pytest.mark.expected_failure
def test_download_proteome_rejects_species_all_and_unknown_species(tmp_path):
    """'all' has no single proteome id; 'cat' is not a species we know."""
    with pytest.raises(ValueError) as exc:
        kl.download_proteome("all", dest_dir=str(tmp_path))
    assert "ref_prot_file" in str(exc.value)

    with pytest.raises(ValueError, match="species"):
        kl.download_proteome("cat", dest_dir=str(tmp_path))


#%% The legacy function, and the red-first proof


def _load_legacy_retrieve_sequences():
    """`retrieve_sequences` from the legacy source, WITHOUT importing it.

    Importing `py_kinase_library.utils.utils` would pull in logomaker, sklearn
    and matplotlib and write bytecode into a git repository. The function body
    is read by line range and exec'd in a namespace holding only what it needs.
    """
    import re as _re

    import numpy as _np
    import tqdm as _tqdm
    from Bio import SeqIO as _SeqIO

    lines = open(LEGACY_UTILS).read().split("\n")
    start = next(i for i, line in enumerate(lines)
                 if line.startswith("def retrieve_sequences"))
    end = next(i for i in range(start + 1, len(lines))
               if lines[i].startswith("def "))
    source = "\n".join(lines[start:end])

    namespace = {"os": os, "re": _re, "np": _np, "pd": pd,
                 "SeqIO": _SeqIO, "tqdm": _tqdm.tqdm,
                 "__file__": LEGACY_UTILS}
    exec(compile(source, LEGACY_UTILS, "exec"), namespace)
    return namespace["retrieve_sequences"]


def _prefix_gene_symbol_index(record):
    """The bundled legacy copy's `id_format` handling, reconstructed.

    It branched on `'gene_name'` and `'protein_name'`; the documented
    `'gene_symbol'` hit neither and fell through to the accession branch. So
    asking for gene symbols indexed the proteome by accession, and every gene
    lookup missed.
    """
    id_format = "gene_symbol"
    if id_format == "protein_name":
        return record.id.split("|")[2].split("_")[0]
    if id_format == "gene_name":                      # never true for gene_symbol
        import re as _re
        match = _re.search(r"GN=([^\s]+)", record.description)
        return match.group(1) if match else None
    return record.id.split("|")[1]                    # <- the silent fall-through


@pytest.mark.logic
def test_red_first_proof_gene_symbol_fix(test_proteome_path):
    """The gene-symbol test fails against the pre-fix behaviour and passes now.

    `scripts/red_proof.sh` needs a pre-fix commit and the fixed code arrived in
    this repository as a whole new module, so the pre-fix branch is
    reconstructed exactly and both are run side by side.
    """
    from Bio import SeqIO

    wanted = {"TP53", "MAPK1", "EGFR"}
    prefix_keys, postfix_keys = set(), set()
    with open(test_proteome_path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            prefix_keys.add(_prefix_gene_symbol_index(record))
            key = sequences._record_id(record, "gene_symbol")
            if key is not None:
                postfix_keys.add(key)

    pre_hits = wanted & prefix_keys
    post_hits = wanted & postfix_keys

    assert not pre_hits, (
        f"expected the pre-fix branch to match NO gene symbol, got {pre_hits}")
    assert post_hits == wanted, (
        f"the fixed branch must match every gene symbol, got {post_hits}")
