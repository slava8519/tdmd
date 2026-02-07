# EAM/Alloy Potentials (Open Sources)

This directory stores open `eam/alloy` (DYNAMO setfl) files for TDMD tasks.

Use `SHA256SUMS` to verify local file integrity after updates.
Machine-readable catalog:
- `library.json`

## Source catalogs

1. LAMMPS bundled potentials
- Tree: `https://github.com/lammps/lammps/tree/stable/potentials`
- Branch head at import time:
  - `stable` -> `9f06a79b033159c606ca880ee63f0ec82b37dc3a`

2. NIST Interatomic Potentials Repository (IPR)
- Browser:
  - `https://www.ctcms.nist.gov/potentials/`
- System pages used to locate files:
  - `https://www.ctcms.nist.gov/potentials/testing/system/Fe/`
  - `https://www.ctcms.nist.gov/potentials/testing/system/Ni/`
  - `https://www.ctcms.nist.gov/potentials/testing/system/Ti/`

## Included setfl files

- `AlCu.eam.alloy`
  - Elements: `Al Cu`
  - Citation (header): `Cai and Ye, Phys Rev B, 54, 8398-8410 (1996)`
  - Source: LAMMPS
- `Al_zhou.eam.alloy`
  - Elements: `Al`
  - Citation (header): `Zhou et al, Acta Mater, 49, 4005 (2001)`
  - Source: LAMMPS
- `Cu_mishin1.eam.alloy`
  - Elements: `Cu`
  - Citation (header): `Mishin, Phys Rev B, 63, 224106 (2001)`
  - Source: LAMMPS
- `Cu_zhou.eam.alloy`
  - Elements: `Cu`
  - Citation (header): `Zhou et al, Acta Mater, 49, 4005 (2001)`
  - Source: LAMMPS
- `CuNi.eam.alloy`
  - Elements: `Ni Cu`
  - Citation (header): `Onat and Durukanoglu, J Phys Cond Matt, 26, 035404 (2014)`
  - Source: LAMMPS
- `Fe_Mishin2006.eam.alloy`
  - Elements: `Fe`
  - Source: NIST IPR download
  - Original model family: Chamati/Papanicolaou/Mishin/Papaconstantopoulos (2006)
- `Ni99.eam.alloy`
  - Elements: `Ni`
  - Source: NIST IPR download
  - Citation text in file header: `Phys. Rev. B 59, 3393 (1999)`
- `Ti_Zhou04.eam.alloy`
  - Elements: `Ti`
  - Citation (header): `X. W. Zhou, R. A. Johnson, H. N. G. Wadley, Phys. Rev. B, 69, 144113 (2004)`
  - Source: NIST IPR download

## Notes

- These are external scientific models; choose by validated regime (phase, temperature, defects, loading path).
- TDMD uses them as read-only runtime inputs via:
  - `potential.kind: eam/alloy`
  - `potential.params.file`
  - `potential.params.elements`
