import kinase_library as kl
import pandas as pd


def test_phosphoproteomics_ser_thr(phosphoproteomics_input_path, phosphoproteomics_ser_thr_output_path):
    input_df = pd.read_csv(phosphoproteomics_input_path)
    pp_object = kl.PhosphoProteomics(input_df)
    output_ser_thr = pp_object.predict(
        kin_type="ser_thr", values_only=True, st_fav=True, pp=True, kinases=['AKT1', 'AKT2'])
    expected_df = pd.read_csv(phosphoproteomics_ser_thr_output_path, index_col=0)
    pd.testing.assert_frame_equal(output_ser_thr, expected_df)


def test_phosphoproteomics_tyrosine(phosphoproteomics_input_path, phosphoproteomics_tyrosine_output_path):
    input_df = pd.read_csv(phosphoproteomics_input_path)
    pp_object = kl.PhosphoProteomics(input_df)
    output_tyrosine = pp_object.predict(
        kin_type="tyrosine", values_only=True, st_fav=True, pp=True, kinases=['HER2', 'EGFR'])
    expected_df = pd.read_csv(phosphoproteomics_tyrosine_output_path, index_col=0)
    pd.testing.assert_frame_equal(output_tyrosine, expected_df)
