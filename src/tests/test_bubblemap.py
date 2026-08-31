import matplotlib.pyplot as plt
import pandas as pd

import kinase_library as kl


def _bubblemap_pval_legend_title(pval_legend_title=None):
    lff_data = pd.DataFrame({"condition": [1.5, -1.2]}, index=["K1", "K2"])
    pval_data = pd.DataFrame({"condition": [0.01, 0.02]}, index=["K1", "K2"])
    title_kwargs = {}
    if pval_legend_title is not None:
        title_kwargs["pval_legend_title"] = pval_legend_title

    try:
        kl.plot_bubblemap(
            lff_data=lff_data,
            pval_data=pval_data,
            sort_kins_by=False,
            color_kins_by=None,
            family_legned=False,
            lff_cbar=False,
            num_panels=1,
            max_window=False,
            **title_kwargs,
        )
        legends = [
            axis.get_legend()
            for axis in plt.gcf().axes
            if axis.get_legend() is not None
        ]
        assert len(legends) == 1
        return legends[0].get_title().get_text()
    finally:
        plt.close("all")


def test_bubblemap_uses_custom_p_value_legend_title():
    assert _bubblemap_pval_legend_title("p-value") == "p-value"


def test_bubblemap_preserves_default_p_value_legend_title():
    assert _bubblemap_pval_legend_title() == "Adj. p-value"
