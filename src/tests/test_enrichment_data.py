import kinase_library as kl
import pandas as pd


def test_enrichment_ser_thr(
        enrichment_foreground_ser_thr_input_path, enrichment_background_ser_thr_input_path, enrichment_ser_thr_output_path):
    input_foreground_df = pd.read_csv(enrichment_foreground_ser_thr_input_path)
    input_background_df = pd.read_csv(enrichment_background_ser_thr_input_path)
    enrichment_object = kl.phosphoproteomics.EnrichmentData(
        input_foreground_df, input_background_df, kin_type="ser_thr")
    output_ser_thr = enrichment_object.kinase_enrichment(non_canonical=False,
                                                         kl_method="percentile_rank", kl_thresh=15, kinases=['AKT1', 'AKT2'])

    expected_df = pd.read_csv(enrichment_ser_thr_output_path, index_col=0)
    pd.testing.assert_frame_equal(output_ser_thr.enrichment_results, expected_df)


def test_enrichment_tyrosine(
        enrichment_foreground_tyrosine_input_path, enrichment_background_tyrosine_input_path, enrichment_tyrosine_output_path):
    input_foreground_df = pd.read_csv(enrichment_foreground_tyrosine_input_path)
    input_background_df = pd.read_csv(enrichment_background_tyrosine_input_path)
    enrichment_object = kl.phosphoproteomics.EnrichmentData(
        input_foreground_df, input_background_df, kin_type="tyrosine")
    output_tyrosine = enrichment_object.kinase_enrichment(non_canonical=False,
                                                          kl_method="percentile_rank", kl_thresh=15, kinases=['HER2', 'EGFR'])

    expected_df = pd.read_csv(enrichment_tyrosine_output_path, index_col=0)
    pd.testing.assert_frame_equal(output_tyrosine.enrichment_results, expected_df)
