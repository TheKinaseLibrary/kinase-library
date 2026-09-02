# TEST PLAN - modules/sequences.py (retrieve_sequences)

**Target:** `src/kinase_library/modules/sequences.py`
**Runner:** pytest, invoked as
`env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONNOUSERSITE=1 python3 -m pytest -p no:cacheprovider -o addopts="" src/tests -ra -q`
**Catalog profiles selected:** `catalog-universal` (always) + `profile-data` (it
slices sequences by 1-based position, pads at boundaries, and de-duplicates an
index of identifiers where many records share a gene symbol) + `profile-cli-io`
(it reads a table and a FASTA, writes two files, and returns meaningful exit
codes) + `profile-library` (it is public API: `kl.retrieve_sequences` is imported
by analysis scripts, 133 call sites use argument spellings that must keep
working, and the name cannot be renamed once shipped).
`profile-external` is selected only for `download_proteome`, which fetches a
reference proteome over HTTP; it is not selected for the rest of the module,
which makes no network call and needs no credentials.

**What the code must guarantee.** Given a protein identifier and a position it
returns the 15-mer centred on that residue, identical to what the legacy
`retrieve_sequences` returned for the analyses already published — and where it
cannot, it says so per row with a reason rather than dropping the row or
guessing. The failure it exists to prevent is a window that is off by one, or
built from the wrong isoform, being scored silently: no downstream step can
detect that, because a wrong 15-mer is still a valid 15-mer. As package API it
must additionally keep the legacy call sites working, and must never lose rows
without saying so.

**Fixture provenance.** Every expected window is computed in
`src/tests/tools/make_windows_fixture.py` by slicing the raw sequence with
explicit bounds and explicit padding — never by calling the module under test.
The fixture is an oracle, not an echo. The same generator writes
`src/tests/data/inputs/uniprot_test_proteome.fasta`, a deterministic subset of
the July-2021 human UniProt proteome holding only the records the fixture needs,
so the suite runs offline in CI where the 52 MB source file does not exist. No
collaborator data is used.

## Logic

- `p53_golden_case` - `P04637`/15 returns `PSVEPPLSQETFSDL`, the ATM/ATR site;
  the single most-checkable value in the whole module
- `every_fixture_row_matches_its_independent_oracle` - parametrized over all
  fixture rows: the module's window equals the `expected_15mer` the fixture
  computed by a different route
- `legacy_equivalence_on_every_fixture_row` - the legacy `retrieve_sequences`,
  exec'd from source **without importing the legacy package**, produces the
  same `SITE_+/-7_AA` as the module for every fixture row; the analyses already
  published rest on the legacy values, so any divergence is a defect here.
  Skipped with a reason naming the path where the legacy source is absent
- `n_terminus_pads_on_the_left` - a position at or below 7 yields leading
  underscores and the acceptor still at index 7
- `c_terminus_pads_on_the_right` - a position within 7 of the end yields
  trailing underscores, same centring
- `isoform_suffix_resolves_to_the_isoform` - `<ACC>-2` returns the isoform's
  window; the bare accession returns the canonical one; they differ
- `gene_symbol_matches_the_accession_route` - the same site reached by gene
  symbol and by accession gives the same window
- `gene_symbol_prefers_the_canonical_record` - for a gene with several isoform
  records, the chosen `NAME` carries no isoform suffix (**regression: the
  bundled legacy copy branched on `'gene_name'`, so the documented
  `'gene_symbol'` silently matched nothing**)
- `red_first_proof_gene_symbol_fix` - the pre-fix branch reconstructed in the
  test matches no gene symbol at all while the shipped branch matches every one,
  which is the proof the regression test above would have failed before the fix
- `gene_symbol_prefers_swissprot_over_trembl` - on a synthetic FASTA whose
  records are deliberately ordered tr-canonical, sp-isoform, sp-canonical,
  tr-isoform under one `GN=`, the sp canonical record wins, not the first row
- `protein_name_id_format_resolves` - `id_format='protein_name'` matches the
  entry-name prefix, so `P53`/15 returns the p53 golden window
- `parse_position_across_every_real_form` - parametrized: bare int, `'33'`,
  `S33`, `s33`, `Ser33`, `SER33`, `Y1068`, `pS33`, `pY1068`, `S-33`, `T_33`,
  padded whitespace, and `33.0`
- `input_frame_is_never_mutated` - the caller's frame is unchanged and gains no
  window column
- `output_preserves_input_row_order_and_row_count` - including unresolved rows,
  so the result stays joinable against what the collaborator sent
- `window_is_always_fifteen_with_the_acceptor_at_index_seven` - invariant over
  every resolved row of every fixture
- `proteome_dir_accessors_round_trip` - `set_current_proteome_dir` then
  `get_current_proteome_dir` returns what was set, and `reset` restores the
  default
- `proteome_dir_reads_the_environment_variable` - `KL_PROTEOME_DIR` set then
  reset makes that directory current; unset then reset restores the legacy
  default
- `species_resolves_dateless_before_dated` - with both candidate filenames
  present in the proteome directory the dateless one wins, and with only the
  `_07_2021` name present that one is used
- `drop_unresolved_returns_the_legacy_single_frame` - one frame of only the
  resolved rows, original index preserved, same windows as the tuple path
- `partial_miss_warns_with_counts_per_reason` - dropping rows is never silent
- `deprecated_argument_aliases_match_the_modern_spelling` - parametrized over
  `id_column`, `protein_position_column`, `phospho_data` and the two real corpus
  call shapes: each warns `DeprecationWarning` and returns what the modern
  spelling returns
- `ignore_missing_true_restores_the_legacy_single_frame` - the 114 corpus calls
  that assign one variable keep working
- `id_format_gene_name_is_accepted_as_a_deprecated_alias` - six live corpus
  calls pass it; it means `gene_symbol` and warns
- `cli_resolves_everything_and_exits_zero` - the windowed table is written with
  the window column present
- `cli_reports_unresolved_rows_and_exits_one` - the `_unresolved.tsv` file is
  written with a `reason` column and the per-reason counts are printed
- `cli_allow_missing_exits_zero_with_unresolved_rows` - the override works
- `cli_runs_as_a_python_module` - one real subprocess `python -m
  kinase_library.modules.sequences` invocation returns 0
- `download_proteome_writes_the_fasta_and_a_release_sidecar` - with the
  transport monkeypatched, the file lands under the dateless name and the
  sidecar records the URL and release headers
- `download_proteome_output_is_found_by_the_species_resolver` - the dateless
  name it writes is exactly what `retrieve_sequences(species=...)` then finds

## Edge cases

- `position_one_is_all_left_padding` - the first residue resolves, seven
  underscores before it
- `position_equals_protein_length_resolves` - the last residue resolves, seven
  underscores after it
- `float_position_from_a_nan_column` - `33.0` is position 33; pandas turns an
  int column to float the moment one cell is NaN, so this is the common case,
  not an exotic one
- `whitespace_around_the_site_string` - `'  S33 '` resolves
- `duplicate_rows_resolve_identically` - the same site twice gives the same
  window twice, and both rows survive
- `empty_table_gives_empty_output_and_no_misses` - a header-only table is not an
  error: both frames come back empty, the window column still present, and no
  warning is raised
- `lowercase_accession_is_reported_not_silently_matched` - accessions are
  case-sensitive in UniProt; a lowercase one is a miss with a reason rather
  than a silent wrong match
- `records_without_a_gene_symbol_are_skipped_not_indexed_as_nan` - 845 of the
  100,100 human headers carry no `GN=`; they must drop out of a gene-symbol
  index rather than becoming a NaN key
- `cli_reads_an_xlsx_table` - the delimiter is inferred from the extension and
  Excel input is read through openpyxl
- `download_proteome_creates_a_missing_destination_directory` - a fresh install
  has no proteome directory yet

## Expected failures

Each verified against the four-part contract: the right exception or exit code,
a message naming the offending thing, no partial side effect, and failure
before expensive work where that is possible.

- `accession_absent_from_fasta_is_reported` - reason `accession_not_in_fasta`,
  row kept with an empty window
- `position_beyond_protein_length_is_reported` - reason
  `position_out_of_range`
- `unparseable_site_string_is_reported` - reason `position_unparseable`
- `non_phosphoacceptor_centre_is_reported` - reason `central_residue_not_STY`
- `stated_residue_disagreeing_with_the_sequence_is_reported` - reason
  `residue_mismatch`; `T33` on a protein whose position 33 is `S` passes the
  central-residue test and must still fail
- `strict_raises_on_the_first_miss` - `SiteResolutionError`, naming the
  identifier and the reason
- `retrieve_sequences_raises_KeyError_naming_an_absent_column` - the
  library-level counterpart of the CLI check
- `retrieve_sequences_raises_TypeError_on_a_non_dataframe` - a list of dicts is
  a plausible mistake and must be named, not indexed into
- `invalid_species_raises_ValueError_listing_the_valid_set` - the legacy code
  silently fell through to the all-species proteome instead
- `invalid_id_format_raises_ValueError_listing_the_valid_set` - the exact
  fall-through that made the legacy `gene_symbol` match nothing
- `missing_proteome_file_raises_FileNotFoundError_naming_the_recovery` - the
  message names the directory, both candidate filenames, `KL_PROTEOME_DIR`,
  `set_current_proteome_dir` and `ref_prot_file`
- `drop_unresolved_with_nothing_resolved_raises` - `SiteResolutionError` with
  the counts, rather than handing back an empty frame that looks like a result
- `deprecated_alias_with_its_replacement_raises_TypeError` - passing both
  `id_column` and `protein_column` is ambiguous and must be refused by name
- `peptide_column_raises_TypeError_naming_the_unsupported_mode` - ten corpus
  calls pass it, no surviving implementation ever accepted it, and silently
  ignoring it would score the wrong sites
- `absent_column_exits_2_before_the_fasta_is_parsed` - the message lists the
  columns present, and no "parsing reference proteome" line is emitted
- `absent_fasta_exits_2_naming_the_path` - and naming the `--fasta` override
- `bad_id_format_exits_2_listing_the_three_valid_values` - an unrecognised
  `--id-format` names all three accepted values rather than falling through to
  accession matching, which is exactly how the bundled legacy copy failed
- `cli_strict_exits_one_with_the_reason_on_stderr` - and writes no output file
- `download_proteome_refuses_to_clobber_without_overwrite` - `FileExistsError`,
  the existing file untouched
- `download_proteome_leaves_no_part_file_when_the_transport_fails` - a
  mid-stream error propagates and the partial file is removed
- `download_proteome_rejects_species_all_and_unknown_species` - `ValueError`
  pointing at `ref_prot_file` for the all-species case
