<div align="center">
    <picture>
        <img src="./src/notebooks/images/logo_qr_combined.png" alt="The Kinase Library" width="50%">
    </picture>

<hr/>

[![Twitter Follow](https://img.shields.io/twitter/follow/KinaseLibrary?style=social)](https://twitter.com/KinaseLibrary) &ensp;
[![License: CC BY-NC-SA 3.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%203.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/3.0/) &ensp;
[![PyPI Latest Release](https://img.shields.io/pypi/v/kinase-library.svg)](https://pypi.org/project/kinase-library/)

</div>

The Kinase Library Python package provides tools for analyzing kinase phosphorylation preferences and predicting substrate specificity. This package is based on recent research that leverages bioinformatics and synthetic peptide libraries to identify potential phosphorylation sites for both Serine/Threonine and Tyrosine kinases. It includes useful functions and analyses, such as:

- **Score Site and Multiple Sites**: Calculate scores for phosphorylation at specific sites or across multiple potential sites.
- **Binary Enrichment Analysis**: Kinase enrichment analysis based on Fisher's exact test for foreground and background substrate lists
- **Differential Phosphorylation Analysis**: Kinase enrichment analysis based on Fisher's exact test for differentially phosphorylated phosphosites

## Installation

You can install the package via pip:

```
pip install kinase-library
```

## Getting Started

The Kinase Library package offers several tools for analyzing kinase phosphorylation sites. Below are some examples to help you get started. See [`src/notebooks`](https://github.com/TheKinaseLibrary/kinase-library/tree/master/src/notebooks/) for more comprehensive usage.

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

Let's say you have a CSV file called "pps_data.csv" containing the following list of phosphosites:

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

phosphosites_data = pd.read_csv("pps_data.csv")
pps = kl.PhosphoProteomics(phosphosites_data, seq_col='sequence window')
pps.predict(kin_type='ser_thr')
```

Here’s an example of the output you can expect from using the PhosphoProteomics.predict() function.

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

- "An atlas of substrate specificities for the human serine/threonine kinome"
  Johnson, J.L., et al., Nature, 2023 Jan 11; 613, 759–766; DOI: [10.1038/s41586-022-05575-3](https://doi.org/10.1038/s41586-022-05575-3)

- "The intrinsic substrate specificity of the human tyrosine kinome"
  Yaron-Barir, T.M., et al., Nature, 2024 May 8; 629, 1174-1181; DOI: [10.1038/s41586-024-07407-y](https://doi.org/10.1038/s41586-024-07407-y)

## License

This package is distributed under the Creative Commons License. See `LICENSE` for more information.

## Contributors

<a href="https://github.com/TheKinaseLibrary/kinase-library/graphs/contributors" alt="View Contributors">
  <img src="https://contrib.rocks/image?repo=TheKinaseLibrary/kinase-library&max=1000&columns=10" alt="Contributors" />
</a>
