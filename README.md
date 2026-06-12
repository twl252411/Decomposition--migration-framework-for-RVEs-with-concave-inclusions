# rve_concave_inc_decom_mig

Generation of dense periodic 2D representative volume elements (RVEs) with concave inclusions using convex decomposition and migration-based overlap removal.

## Overview

This repository provides an end-to-end workflow for building periodic 2D RVEs with non-convex inclusion shapes. The main pipeline:

1. Generates a concave inclusion boundary from an analytic curve or fixed vertices.
2. Decomposes the concave polygon into convex parts.
3. Places multiple inclusions inside a square RVE.
4. Optimizes inclusion orientations when a target second-order orientation tensor is prescribed.
5. Removes overlaps through iterative migration under periodic boundary conditions.
6. Exports geometry templates, position files, plots, and run metrics.

The main entry point is [main.py](main.py).

## Features

- Supports three built-in inclusion types:
  - `lobular2`
  - `pea`
  - `concave_poly`
- Convex decomposition templates stored in Excel workbooks
- Periodic broad-phase candidate search with cell lists
- Narrow-phase collision handling using:
  - GJK/EPA when `numba` is available
  - SAT-based fallback otherwise
- Orientation optimization against a target second-order tensor
- Adaptive migration step size
- Export of both non-periodic and periodic position files
- Abaqus/CAE post-processing scripts for 2D RVE model construction

## Repository Structure

```text
.
|-- main.py
|-- README.md
|-- shape_decomposition/
|   |-- circum_lobular2.py
|   |-- circum_pea.py
|   |-- circum_concave_poly.py
|   |-- circum_concave_general.py
|   |-- concave_decomp_l2.py
|   |-- concave_decomp_pea.py
|   `-- concave_decomp_concave_poly.py
|-- intersection_seperation/
|   |-- rve_runner.py
|   |-- solver.py
|   |-- candidates.py
|   |-- collision_kernel.py
|   |-- periodic.py
|   |-- io_excel.py
|   |-- viz.py
|   `-- accelerators_numba.py
|-- abaqus_rve/
|   |-- geometry_shape_2d.py
|   `-- rve_generation.py
`-- intermediate_final_files/
```

Note: the package directory is named `intersection_seperation` in the current codebase for backward compatibility with the existing scripts.

## Requirements

- Python 3.10+
- `numpy`
- `scipy`
- `matplotlib`
- `openpyxl`
- `numba` (optional, for acceleration only)

The code has been checked in a Conda environment named `statistics` with Python 3.12. It also runs without `numba`; in that case the solver falls back to the pure Python / NumPy path.

## Installation

### Option 1: Use an existing Conda environment

```bash
conda activate statistics
pip install numpy scipy matplotlib openpyxl
# optional
# pip install numba
```

### Option 2: Create a new environment

```bash
conda create -n rve python=3.12 numpy scipy matplotlib openpyxl
conda activate rve
# optional
# pip install numba
```

### Option 3: Use the provided environment file

```bash
conda env create -f environment.yml
conda activate rve-concave
```

### Option 4: Install from requirements

```bash
pip install -r requirements.txt
```

## Quick Start

Run the full workflow:

```bash
python main.py
```

This will:

- generate or refresh the outer polygon and convex partition templates
- build the initial RVE configuration
- optimize orientations if requested
- perform overlap-removal migration
- export output files into `intermediate_final_files/`

## Workflow Summary

### 1. Shape generation

Shape definitions are handled in `shape_decomposition/circum_*.py`.

### 2. Concave-to-convex decomposition

The decomposition scripts generate `convex_partition_*.xlsx`, which contain:

- the original outer polygon
- convex sub-polygons
- curve samples
- area information

### 3. RVE generation and migration

The RVE solver lives in `intersection_seperation/rve_runner.py` and `intersection_seperation/solver.py`.

The current sequence is:

1. initial placement
2. orientation optimization
3. position optimization by migration

The exported `*_polygon_positions_initial.txt` and optional `initial_rve_*.png` correspond to the actual state after orientation optimization and before position migration.

### 4. Export

Geometry, position files, and run statistics are written into `intermediate_final_files/`.

## Configuration

The default batch settings are defined in `build_default_config()` inside [main.py](main.py).

Key parameters:

- `demo_shape`: inclusion type, one of `"lobular2"`, `"pea"`, or `"concave_poly"`
- `rve_size`: side length of the square RVE
- `vol_frac_inc`: target inclusion area fraction
- `ori_ten2`: target second-order orientation tensor
- `rand_ori`: if `True`, use random orientations instead of tensor-driven optimization
- `scale_ratio_collision`: collision-envelope scaling factor
- `step_scale`: initial migration step-size scale
- `max_iter`: maximum number of migration iterations
- `seed`: random seed; use `None` for non-reproducible runs
- `circumscribe_mode`: circumscribing strategy for supported shapes
- `lobular2_num_vertices`, `pea_num_vertices`: polygon discretization controls

The batch driver is `run_case(config, n_runs=20, max_attempts=None)`.

- `n_runs`: requested number of converged samples
- `max_attempts`: hard cap on total attempts to avoid infinite retry loops

If `max_attempts` is not provided, it defaults to `max(10 * n_runs, n_runs)`.

## Outputs

Typical outputs written to `intermediate_final_files/` include:

- `outer_polygon_vertices_*.xlsx`
- `convex_partition_*.xlsx`
- `*_polygon_positions_initial.txt`
- `*_polygon_positions_after.txt`
- `*_polygon_positions_after_periodic.txt`
- `initial_rve_*.png`
- `final_rve_*.png`
- `run_metrics_<shape>_<mode>_<run_id>.txt`
- `run_metrics_<shape>_<mode>_avg<n>_<step_scale>.txt`

### Position files

- `*_polygon_positions_initial.txt`
  - orientation-optimized initial state before migration
- `*_polygon_positions_after.txt`
  - final converged non-periodic positions
- `*_polygon_positions_after_periodic.txt`
  - final positions with periodic image copies appended

### Run metrics

For each successful run, one metrics file is written:

```text
run_metrics_<shape>_<mode>_<run_id>.txt
```

The averaged summary file is named using the actual number of successful runs:

```text
run_metrics_<shape>_<mode>_avg<n>_<step_scale>.txt
```

Example:

```text
run_metrics_concave_poly_external_avg10_0.25.txt
```

## Abaqus/CAE Workflow

The scripts in `abaqus_rve/` build 2D Abaqus/CAE models from exported position files.

Typical usage:

1. generate RVE data in this repository
2. open Abaqus/CAE
3. run `abaqus_rve/rve_generation.py` inside the Abaqus Python environment
4. set the desired inclusion type in the Abaqus script

The Abaqus workflow expects the corresponding `*_polygon_positions_after_periodic.txt` file to be present in `intermediate_final_files/`.

## Troubleshooting

### Missing dependencies

- `ImportError: scipy`
  - install `scipy`, which is required for Halton sampling
- Excel files not written
  - install `openpyxl`
- slow execution
  - install `numba` to enable the accelerated collision kernels

### Solver behavior

- no converged runs collected
  - reduce `vol_frac_inc`
  - increase `max_iter`
  - reduce the initial `step_scale`
  - inspect the convex partition quality
- unexpected residual overlaps
  - verify that the convex partition fully covers the intended outer polygon
  - check whether the selected discretization is too coarse

### Environment notes

- if `numba` is unavailable, the code still runs through the fallback path
- on Windows, running inside the intended Conda environment is recommended

## Citation

If this repository is useful in your work, please cite:

Ying Ye, Lehua Qi, Wenlong Tian, *A geometry-consistent decomposition--migration framework for constructing dense periodic RVEs with concave inclusions*, submitted to *Computers and strutures*, 2026.

## License

This repository is distributed under the license provided in [LICENSE](LICENSE).
