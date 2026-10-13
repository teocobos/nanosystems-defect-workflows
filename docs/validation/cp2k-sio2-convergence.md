# CP2K numerical convergence validation: alpha-quartz SiO2

**Document date:** 2026-10-13
**NSDW code revision:** `6416ddc`
**Execution platform:** ARCHER2
**CP2K version:** 2025.2

## Scope

Nine-atom alpha-quartz SiO2 benchmark using CP2K Quickstep, PBE exchange-correlation, GTH-PBE pseudopotentials and TZV2P-MOLOPT-GTH basis functions.

This study tests total-energy convergence with respect to plane-wave cutoff, relative cutoff and k-point sampling. The acceptance criterion is 1 meV/atom.

## Selected numerical settings

| Parameter | Setting |
|---|---|
| Plane-wave cutoff | 400 Ry |
| Relative cutoff | 30 Ry |
| K-point mesh | Monkhorst-Pack 2x2x2 |
| Basis | TZV2P-MOLOPT-GTH (provisional) |
| Functional | PBE |
| Pseudopotentials | GTH-PBE-q4 (Si), GTH-PBE-q6 (O) |
| SCF threshold | EPS_SCF 1e-6 |

## Individual convergence campaigns

### TZV2P cutoff

- Campaign: `outputs/archer2-validation/sio2-pbe-tzv2p-cutoff`
- Selected candidate: `400-Ry`
- Reference candidate: `600-Ry`
- Tolerance satisfied: `True`

| Candidate | Energy (eV/atom) | Difference (meV/atom) |
|---|---:|---:|
| 400-Ry | -328.097601069817 | 0.821810 |
| 500-Ry | -328.097490946759 | 0.931934 |
| 600-Ry | -328.098422880271 | 0.000000 |

### TZV2P relative cutoff

- Campaign: `outputs/archer2-validation/sio2-pbe-tzv2p-relative-cutoff`
- Selected candidate: `30-Ry`
- Reference candidate: `100-Ry`
- Tolerance satisfied: `True`

| Candidate | Energy (eV/atom) | Difference (meV/atom) |
|---|---:|---:|
| 20-Ry | -328.111760109476 | 14.667637 |
| 30-Ry | -328.097601069817 | 0.508598 |
| 40-Ry | -328.097099641296 | 0.007169 |
| 60-Ry | -328.097092623111 | 0.000151 |
| 80-Ry | -328.097092472128 | 0.000000 |
| 100-Ry | -328.097092472054 | 0.000000 |

### TZV2P k-points

- Campaign: `outputs/archer2-validation/sio2-pbe-tzv2p-kpoints`
- Selected candidate: `2x2x2`
- Reference candidate: `3x3x3`
- Tolerance satisfied: `True`

| Candidate | Energy (eV/atom) | Difference (meV/atom) |
|---|---:|---:|
| 1x1x1 | -328.037362360305 | 60.056370 |
| 2x2x2 | -328.097601069817 | 0.182339 |
| 3x3x2 | -328.097439503096 | 0.020772 |
| 3x3x3 | -328.097418730742 | 0.000000 |

### Extended basis study

- Campaign: `outputs/archer2-validation/sio2-pbe-basis-extended`
- Selected candidate: `None`
- Reference candidate: `TZV2PX`
- Tolerance satisfied: `False`

| Candidate | Energy (eV/atom) | Difference (meV/atom) |
|---|---:|---:|
| DZVP | -328.021259885318 | 175.042186 |
| TZVP | -328.058962802699 | 137.339269 |
| TZV2P | -328.097601069817 | 98.701002 |
| TZV2PX | -328.196302071648 | 0.000000 |

## Final joint numerical verification

| Quantity | Value |
|---|---:|
| Baseline (400/30 Ry, 2x2x2) | -328.097601069817 eV/atom |
| Joint reference (600/100 Ry, 3x3x3) | -328.097502661377 eV/atom |
| Absolute difference | 0.098408 meV/atom |
| Acceptance threshold | 1.000000 meV/atom |
| Result | PASS |

Joint-reference output: `outputs/archer2-validation/sio2-pbe-tzv2p-joint-verification/600-Ry/alpha_quartz_cod1526860-600-Ry.out`

## Basis-set convergence limitation

The DZVP, TZVP, TZV2P and TZV2PX basis study did not demonstrate total-energy convergence within 1 meV/atom. TZV2PX is an extended basis variant, not a systematic quadruple-zeta reference. TZV2P is therefore a provisional working basis, not a proven converged basis.

## Interpretation and limitations

The individual campaigns and joint-reference comparison support the selected numerical settings for this specific nine-atom SiO2 PBE benchmark at fixed TZV2P basis.

The joint comparison is a finite-reference check, not proof of convergence against the infinite-cutoff, complete-k-point or complete-basis limits. Error cancellation is possible.

These results do not establish convergence of defect formation energies, charge transition levels, migration barriers, supercell-size effects or charged-defect corrections. Those require separate validation in future workflows.

## Reproducibility

The campaign directories contain generated CP2K inputs, coordinates, calculation outputs and NSDW analysis reports. Preserve the original files as scientific evidence. This document is a summary, not a replacement for them.

