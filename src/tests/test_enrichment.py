import matplotlib.pyplot as plt
import pandas as pd

import kinase_library as kl


def _bubblemap_axis_titles(lff_data, pval_data, **kwargs):
    try:
        kl.plot_bubblemap(
            lff_data=lff_data,
            pval_data=pval_data,
            sort_kins_by=False,
            color_kins_by=None,
            family_legned=False,
            pval_legend=False,
            num_panels=1,
            max_window=False,
            **kwargs,
        )
        return [axis.get_title() for axis in plt.gcf().axes if axis.get_title()]
    finally:
        plt.close("all")


def test_mea_bubblemap_uses_custom_colorbar_title():
    kinases = ["K1", "K2"]
    mea_results = {
        "condition_a": pd.DataFrame(
            {"NES": [1.5, -1.2], "FDR": [0.01, 0.02]}, index=kinases
        ),
        "condition_b": pd.DataFrame(
            {"NES": [1.1, -1.4], "FDR": [0.03, 0.01]}, index=kinases
        ),
    }
    expected_nes_data = pd.DataFrame(
        {"condition_a": [1.5, -1.2], "condition_b": [1.1, -1.4]}, index=kinases
    )
    expected_pval_data = pd.DataFrame(
        {"condition_a": [0.01, 0.02], "condition_b": [0.03, 0.01]}, index=kinases
    )

    nes_data, pval_data = kl.combine_mea_enrichment_results(
        mea_results, data_type="data_frame"
    )

    pd.testing.assert_frame_equal(nes_data, expected_nes_data)
    pd.testing.assert_frame_equal(pval_data, expected_pval_data)
    assert _bubblemap_axis_titles(
        nes_data, pval_data, lff_cbar_title="NES"
    ) == ["NES"]


def test_bubblemap_preserves_default_colorbar_title():
    lff_data = pd.DataFrame({"condition": [1.5, -1.2]}, index=["K1", "K2"])
    pval_data = pd.DataFrame({"condition": [0.01, 0.02]}, index=["K1", "K2"])

    assert _bubblemap_axis_titles(lff_data, pval_data) == ["log2(FF)"]
