<div align="center">
    <picture>
        <img src="https://i.imgur.com/TvQ8LmC.png" alt="The Kinase Library" width="50%">
    </picture>

<hr/>

# [Click here for The Kinase Library Web Tool](https://kinase-library.phosphosite.org)

[![Twitter Follow](https://img.shields.io/twitter/follow/KinaseLibrary?style=social)](https://twitter.com/KinaseLibrary) &ensp;
[![License: CC BY-NC-SA 3.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%203.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/3.0/) &ensp;
[![PyPI Latest Release](https://img.shields.io/pypi/v/kinase-library.svg)](https://pypi.org/project/kinase-library/)

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

### Example: Identify kinases capable of phosphorylating a site using `Substrate`

```
import kinase_library as kl

# Create a Substrate object with a target sequence (example: p53 S33)
s = kl.Substrate('PSVEPPLsQETFSDL')  # Lowercase 's' indicates a phosphoserine

# Predict potential kinase interactions for the substrate
s.predict()
```

Here’s an example of the output you can expect from using the Substrate.predict() function.

<div style="overflow-x:auto;">

| Kinase  | Score   | Score Rank | Percentile | Percentile Rank |
| ------- | ------- | ---------- | ---------- | --------------- |
| ATM     | 5.0385  | 1          | 99.83      | 1               |
| SMG1    | 4.2377  | 2          | 99.77      | 2               |
| ATR     | 3.5045  | 4          | 99.69      | 3               |
| DNAPK   | 3.8172  | 3          | 99.21      | 4               |
| FAM20C  | 3.1716  | 5          | 95.23      | 5               |
| ...     | ...     | ...        | ...        | ...             |
| BRAF    | -4.4003 | 241        | 7.86       | 305             |
| AKT2    | -5.6530 | 283        | 6.79       | 306             |
| P70S6KB | -3.9915 | 221        | 6.64       | 307             |
| NEK3    | -8.2455 | 309        | 4.85       | 308             |
| P70S6K  | -7.2917 | 305        | 4.19       | 309             |

</div>

### Example: Identify kinases capable of phosphorylating a site for multiple sites using `PhosphoProteomics`

Assuming you have a CSV file called "pps_data.csv" containing the following list of phosphosites:

```
uniprot,protein,gene,description,position,residue,best_localization_prob,sequence window
Q15149,PLEC,PLEC,Plectin,113,T,1.000000,MVMPARRtPHVQAVQ
O43865,SAHH2,AHCYL1,S-adenosylhomocysteine hydrolase-like protein 1,29,S,0.911752,EDAEKysFMATVT
Q8WX93,PALLD,PALLD,Palladin,35,S,0.999997,PGLsAFLSQEEINKS
Q96NY7,CLIC6,CLIC6,Chloride intracellular channel protein 6,322,S,1.000000,AGESAGRsPG_____
Q02790,FKBP4,FKBP4,Peptidyl-prolyl cis-trans isomerase FKBP4,336,S,0.999938,PDRRLGKLKLQAFsAXXESCHCGGPSA
```

```
import kinase_library as kl
import pandas as pd

phosphosites_data = pd.read_csv('pps_data.csv')
pps = kl.PhosphoProteomics(phosphosites_data, seq_col='sequence window')
pps.predict(kin_type='ser_thr')
```

This is the expected output from using the PhosphoProteomics.predict() function.

<div style="overflow-x:auto;">

| uniprot | protein | gene   | description                                     | position | residue | best_localization_prob | sequence window             | phos_res | Sequence             | ... | YSK1_percentile | YSK1_percentile_rank | YSK4_score | YSK4_score_rank | YSK4_percentile | YSK4_percentile_rank | ZAK_score | ZAK_score_rank | ZAK_percentile | ZAK_percentile_rank |
| ------- | ------- | ------ | ----------------------------------------------- | -------- | ------- | ---------------------- | --------------------------- | -------- | -------------------- | --- | --------------- | -------------------- | ---------- | --------------- | --------------- | -------------------- | --------- | -------------- | -------------- | ------------------- |
| Q15149  | PLEC    | PLEC   | Plectin                                         | 113      | T       | 1.000000               | MVMPARRtPHVQAVQ             | t        | MVMPARRtPHVQAVQ      | ... | 80.44           | 130                  | -3.004     | 249             | 32.17           | 244                  | -1.210    | 159            | 80.90          | 128                 |
| O43865  | SAHH2   | AHCYL1 | S-adenosylhomocysteine hydrolase-like protein 1 | 29       | S       | 0.911752               | EDAEKysFMATVT               | s        | \_EDAEKYsFMATVT\_    | ... | 63.85           | 150                  | -1.431     | 125             | 71.22           | 108                  | -1.481    | 129            | 76.87          | 82                  |
| Q8WX93  | PALLD   | PALLD  | Palladin                                        | 35       | S       | 0.999997               | PGLsAFLSQEEINKS             | s        | PGLSAFLsQEEINKS      | ... | 11.73           | 250                  | -2.567     | 128             | 44.07           | 119                  | -4.899    | 228            | 6.80           | 291                 |
| Q96NY7  | CLIC6   | CLIC6  | Chloride intracellular channel protein 6        | 322      | S       | 1.000000               | AGESAGRsPG\_\_\_\_\_        | s        | AGESAGRsPG\_\_\_\_\_ | ... | 52.69           | 134                  | -3.300     | 213             | 24.37           | 284                  | -2.839    | 182            | 47.81          | 163                 |
| Q02790  | FKBP4   | FKBP4  | Peptidyl-prolyl cis-trans isomerase FKBP4       | 336      | S       | 0.999938               | PDRRLGKLKLQAFsAXXESCHCGGPSA | s        | KLKLQAFsAXXESCH      | ... | 46.82           | 216                  | -2.265     | 186             | 52.25           | 178                  | -3.020    | 240            | 43.29          | 233                 |

</div>

## Citations

Please cite the following papers when using this package:

**For the serine/threonine kinome:**
> Johnson, J. L., Yaron, T. M., Huntsman, E. M., Kerelsky, A., Song, J., Regev, A., ... & Cantley, L. C. (2023). **An atlas of substrate specificities for the human serine/threonine kinome**. _Nature_, 613(7945), 759-766. [http://doi.org/10.1074/mcp.TIR118.000943](https://doi.org/10.1038/s41586-022-05575-3)

**For the tyrosine kinome:**
> Yaron-Barir, T. M., Joughin, B. A., Huntsman, E. M., Kerelsky, A., Cizin, D. M., Cohen, B. M., ... & Johnson, J. L. (2024). **The intrinsic substrate specificity of the human tyrosine kinome**. _Nature_, 1-8. [https://doi.org/10.1038/s41586-024-07407-y](https://doi.org/10.1038/s41586-024-07407-y)

**If you are using the MEA tool, please also cite:**
> Fang, Z., Liu, X., & Peltz, G. (2023). **GSEApy: a comprehensive package for performing gene set enrichment analysis in Python**. _Bioinformatics_, 39(1), btac757.

## License

This package is distributed under the Creative Commons License. See `LICENSE` for more information.
