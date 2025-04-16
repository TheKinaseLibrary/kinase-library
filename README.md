<div align="center">
    <picture>
        <img src="https://i.imgur.com/Y6PmRsQ.jpeg" alt="The Kinase Library" width="50%">
    </picture>

<hr/>

# [Click here for The Kinase Library Web Tool](https://kinase-library.phosphosite.org)

<picture>
    <img src="https://i.imgur.com/sWUA4Rk.png" alt="The Kinase Library QR Code" width="20%">
</picture>

[![Twitter Follow](https://img.shields.io/twitter/follow/KinaseLibrary?style=social)](https://twitter.com/KinaseLibrary) &ensp;
[![License: CC BY-NC-SA 3.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%203.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/3.0/) &ensp;
[![PyPI Latest Release](https://img.shields.io/pypi/v/kinase-library.svg)](https://pypi.org/project/kinase-library/)

<hr/>

</div>

**The Kinase Library** is a comprehensive Python package for analyzing phosphoproteomics data, focusing on kinase-substrate relationships. It provides tools for kinase prediction, enrichment analysis, and visualization, enabling researchers to gain insights into kinase activities and signaling pathways from phosphoproteomics datasets.

## Features

* **Kinase Prediction**: Predict potential kinases responsible for phosphorylation sites using a built-in kinase-substrate prediction algorithm.
* **Enrichment Analysis**: Perform kinase enrichment analysis using binary enrichment or differential phosphorylation analysis.
* **Motif Enrichment Analysis (MEA)**: Identify kinases potentially regulated in your dataset using MEA with the GSEA algorithm.
* **Visualization**: Generate volcano plots, bubble maps, and other visualizations to interpret enrichment results.
* **Downstream Substrate Identification**: Explore putative downstream substrates of enriched kinases.

## Installation

You can install the package via pip:

```
pip install kinase-library
```

## Getting Started

The Kinase Library package offers several tools for analyzing kinase phosphorylation sites. Below are some basic examples to help you get started. Please refer to [`Notebooks`](https://github.com/TheKinaseLibrary/kinase-library/tree/master/src/notebooks/) for more comprehensive usage.

## Data Updates

<div>

| Release | Date | New | Updated | Removed | Total Ser/Thr | Total Tyrosine | Total Non-Canonicals (Tyrosine) | Notes |
|:-------:|:----:|:---:|:-------:|:-------:|:-------------:|:--------------:|:-------------------------------:|:-----:|
| **v1.2.0** | Apr 15, 2025 | CDK15 | HUNK | _None_ | 311 | 78 | 15 | |
| **v1.1.0** | Feb 2, 2025 | CDKL2 | CK1D, GRK7, SRPK2 | _None_ | 310 | 78 | 15 | Fixed processing error for PDHK1 and PDHK4 |
| **v1.0.0** | Dec 5, 2024 | ALK1, ALK7, TSSK3, TSSK4, ULK3, WNK2 | CAMKK2, CDK3, CDK5, CDK13, CHAK1, CLK3, GRK1, GRK4, GRK5, ICK, IKKA, LATS1, MEKK6, MLK3, MNK2, MST1, NIM1, PASK, PBK, PKN3, SKMLCK, SMG1, VRK2, WNK3 | _None_ | 309 | 78 | 15 | |
| **v0.1.0** | Oct 30, 2024 | _None_ | _None_ | _None_ | 303 | 78 | 15 | Legacy version - data as described in papers |

</div>

## Citations

Please cite the following papers when using this package:

**For the serine/threonine kinome:**
> Johnson, J. L., Yaron, T. M., Huntsman, E. M., Kerelsky, A., Song, J., Regev, A., ... & Cantley, L. C. (2023). **An atlas of substrate specificities for the human serine/threonine kinome**. _Nature_, 613(7945), 759-766. [https://doi.org/10.1074/mcp.TIR118.000943](https://doi.org/10.1038/s41586-022-05575-3)

**For the tyrosine kinome:**
> Yaron-Barir, T. M., Joughin, B. A., Huntsman, E. M., Kerelsky, A., Cizin, D. M., Cohen, B. M., ... & Johnson, J. L. (2024). **The intrinsic substrate specificity of the human tyrosine kinome**. _Nature_, 1-8. [https://doi.org/10.1038/s41586-024-07407-y](https://doi.org/10.1038/s41586-024-07407-y)

## License

This package is distributed under the Creative Commons License. See `LICENSE` for more information.