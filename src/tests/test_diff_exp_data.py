import kinase_library as kl
import pandas as pd


def test_diff_exp_ser_thr(diff_exp_ser_thr_input_path, diff_exp_ser_thr_output_path):
    input_df = pd.read_csv(diff_exp_ser_thr_input_path)
    diff_exp_object = kl.DiffExpData(input_df, kin_type="ser_thr", lfc_col="logFC")
    output_ser_thr = diff_exp_object.kinase_enrichment(non_canonical=False,
                                                       kl_method="percentile_rank", kl_thresh=15, kinases=['AKT1', 'AKT2'])
    expected_df = pd.read_csv(diff_exp_ser_thr_output_path, index_col=0)
    pd.testing.assert_frame_equal(output_ser_thr.combined_enrichment_results, expected_df)


def test_diff_exp_tyrosine(diff_exp_tyrosine_input_path, diff_exp_tyrosine_output_path):
    input_df = pd.read_csv(diff_exp_tyrosine_input_path)
    diff_exp_object = kl.DiffExpData(input_df, kin_type="tyrosine", lfc_col="logFC")
    output_tyrosine = diff_exp_object.kinase_enrichment(non_canonical=False,
                                                        kl_method="percentile_rank", kl_thresh=15, kinases=['HER2', 'EGFR'])
    expected_df = pd.read_csv(diff_exp_tyrosine_output_path, index_col=0)
    pd.testing.assert_frame_equal(output_tyrosine.combined_enrichment_results, expected_df)
