import pandas as pd

import kinase_library as kl


KINASES = ["K1", "K2"]


def _mea_result(enrichment_results):
    return kl.MeaEnrichmentResults(
        enrichment_results=enrichment_results,
        pps_data=None,
        kin_sub_sets={},
        gseapy_obj=None,
        kin_type="custom",
        kl_method="custom",
        kl_thresh=None,
        tested_kins=KINASES,
        data_att="custom",
        kl_comp_direction=None,
    )


def test_combine_mea_objects_uses_native_raw_p_value_column():
    result_tables = {
        "condition_a": pd.DataFrame(
            {"NES": [1.5, -1.2], "p-value": [0.01, 0.02]}, index=KINASES
        ),
        "condition_b": pd.DataFrame(
            {"NES": [1.1, -1.4], "p-value": [0.03, 0.04]}, index=KINASES
        ),
    }
    results = {
        condition: _mea_result(table) for condition, table in result_tables.items()
    }
    expected_nes_data = pd.DataFrame(
        {"condition_a": [1.5, -1.2], "condition_b": [1.1, -1.4]},
        index=KINASES,
    )
    expected_pval_data = pd.DataFrame(
        {"condition_a": [0.01, 0.02], "condition_b": [0.03, 0.04]},
        index=KINASES,
    )

    nes_data, pval_data = kl.combine_mea_enrichment_results(
        results, adj_pval=False
    )

    pd.testing.assert_frame_equal(nes_data, expected_nes_data)
    pd.testing.assert_frame_equal(pval_data, expected_pval_data)


def test_combine_mea_dataframes_supports_native_and_legacy_raw_p_value_columns():
    results = {
        "native": pd.DataFrame(
            {"NES": [1.5, -1.2], "p-value": [0.01, 0.02]}, index=KINASES
        ),
        "legacy": pd.DataFrame(
            {"NES": [1.1, -1.4], "pvalue": [0.03, 0.04]}, index=KINASES
        ),
    }
    expected_nes_data = pd.DataFrame(
        {"native": [1.5, -1.2], "legacy": [1.1, -1.4]}, index=KINASES
    )
    expected_pval_data = pd.DataFrame(
        {"native": [0.01, 0.02], "legacy": [0.03, 0.04]}, index=KINASES
    )

    nes_data, pval_data = kl.combine_mea_enrichment_results(
        results, data_type="data_frame", adj_pval=False
    )

    pd.testing.assert_frame_equal(nes_data, expected_nes_data)
    pd.testing.assert_frame_equal(pval_data, expected_pval_data)


def test_combine_mea_dataframes_prefers_native_raw_p_value_column():
    results = {
        "condition": pd.DataFrame(
            {
                "NES": [1.5, -1.2],
                "p-value": [0.01, 0.02],
                "pvalue": [0.91, 0.92],
            },
            index=KINASES,
        )
    }

    _, pval_data = kl.combine_mea_enrichment_results(
        results, data_type="data_frame", adj_pval=False
    )

    expected = pd.DataFrame({"condition": [0.01, 0.02]}, index=KINASES)
    pd.testing.assert_frame_equal(pval_data, expected)


def test_combine_mea_dataframes_preserves_adjusted_p_value_default():
    results = {
        "condition": pd.DataFrame(
            {"NES": [1.5, -1.2], "FDR": [0.05, 0.06]}, index=KINASES
        )
    }

    _, pval_data = kl.combine_mea_enrichment_results(
        results, data_type="data_frame"
    )

    expected = pd.DataFrame({"condition": [0.05, 0.06]}, index=KINASES)
    pd.testing.assert_frame_equal(pval_data, expected)


def test_combine_mea_dataframes_honors_explicit_p_value_column():
    results = {
        "condition": pd.DataFrame(
            {
                "NES": [1.5, -1.2],
                "FDR": [0.05, 0.06],
                "p-value": [0.01, 0.02],
                "pvalue": [0.91, 0.92],
                "custom_p": [0.21, 0.22],
            },
            index=KINASES,
        )
    }

    _, pval_data = kl.combine_mea_enrichment_results(
        results,
        data_type="data_frame",
        pval_col_name="custom_p",
        adj_pval=False,
    )

    expected = pd.DataFrame({"condition": [0.21, 0.22]}, index=KINASES)
    pd.testing.assert_frame_equal(pval_data, expected)
