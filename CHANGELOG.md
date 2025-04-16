# Changelog

## [1.2.0] - 2025-04-15

Added

- Matrices (serine/threonine): CDK15

Changed

- Matrices (serine/threonine): HUNK
- Replaced certain kinase "protein" names with more familiar and recognizable names for researchers:
  - p90RSK → RSK1
  - p70S6K1 → S6K1
  - p70S6K2 → S6K2
  - PRKD1 → PKD1
  - PRKD2 → PKD2
  - PRKD3 → PKD3
  - CRIK → CITRON
  - MEKK6 → ASK2
  - MAP3K15 → ASK3
- Upgraded dependencies

## [1.1.1] - 2025-03-16

Fixed

- Bug in plotting kinase data for tyrosine kinases
- Bug in retrieving activated and inhibited kinases in MEA

## [1.1.0] - 2025-02-02

Added

- Matrices (serine/threonine): CDKL2

Changed

- Matrices (serine/threonine): CK1D, GRK7, SRPK2

Fixed

- Processing error for PDHK1 and PDHK4 (Y column should be replaced with F - dual specificity)

## [1.0.5] - 2024-02-02

Fixed

- Stats (upreg/downreg/unreg numbers) in title in volcano plot to be specific to the kinase type

## [1.0.4] - 2025-01-28

Fixed

- Syntax

## [1.0.3] - 2024-12-13

Changed

- Compatible python versions
- Docs

## [1.0.2] - 2024-12-09

Added

- MEA checkpoint
- tqdm logger

Changed

- README

## [1.0.1] - 2024-12-05

Changed

- Bumped version for release; no code changes

## [1.0.0] - 2024-12-05

Added

- Motif Enrichment Analysis (MEA) function
- Tutorial notebooks
- Matrices (serine/threonine): ALK1, ALK7, TSSK3, TSSK4, ULK3, WNK2

Changed

- Matrices (serine/threonine): CAMKK2, CDK3, CDK5, CDK13, CHAK1, CLK3, GRK1, GRK4, GRK5, ICK, IKKA, LATS1, MEKK6, MLK3, MNK2, MST1, NIM1, PASK, PBK, PKN3, SKMLCK, SMG1, VRK2, WNK3
- Restructured modules
- Versions of dependencies
- README

Fixed

- Bugs

## [0.1.0] - 2024-10-30

Added

- The Kinase Library Legacy version used in [Johnson et al. (2023)](https://doi.org/10.1038/s41586-022-05575-3) and [Yaron-Barir et al. (2024)](https://doi.org/10.1038/s41586-024-07407-y)