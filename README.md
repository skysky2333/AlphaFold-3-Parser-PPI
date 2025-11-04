# AlphaFold-3-Parser-PPI

## Functionality
- A simple yet effective parser for AF3 outputs to screen for PPIs
- Emits two outputs:
  - One CSV per run (best model only): rows are cross‑chain residue–residue interactions that pass either
    - distance + confidence: minimal heavy‑atom distance ≤ `--distance_cutoff` AND symmetrized PAE ≤ `--pae_cutoff`, or
    - contact_probs: token‑level contact probability ≥ `--contact_prob_threshold`.
  - One aggregate CSV across all runs: one row per run summarizing counts and extremes.

## To Run:
- `python SCRIPTS_PPIXLS/af3/parse_af3_runs.py <input_dir> [--distance_cutoff 6.0] [--pae_cutoff 8.0] [--contact_prob_threshold 0.5] [--output_dir <dir>]`

## Inputs
- `input_dir`: directory containing multiple AF3 run subfolders. It can be ran directly on the AF3 server downloaded outputs.
- A run is detected if it contains files like:
  - `*_summary_confidences_*.json`
  - `*_full_data_*.json`
  - `*_model_*.cif`
  - `*_job_request.json`
- The script expects AF3 full JSON keys: `token_chain_ids`, `token_res_ids`, `pae`, `contact_probs`.
- Chain pairs are treated as unordered (e.g., A–B only; B–A not duplicated).

## Outputs
- Aggregate (all runs): `<output_dir>/runs_summary.csv`
  - Columns:
    - `run_name`, `model_index`, `ranking_score`, `ptm`, `iptm`
    - `num_pairs_distance_conf` — count of residue pairs passing distance + PAE
    - `num_pairs_contact_prob` — count of residue pairs passing contact prob threshold
    - `min_distance_conf` — smallest minimal heavy‑atom distance among pairs passing distance + PAE (Å)
    - `max_contact_prob` — maximum cross‑chain contact probability (any pair)
    - `max_contact_prob_passed` — maximum contact probability among pairs passing the contact prob threshold
    - `num_unique_residues_any` — unique residues involved in any passing interaction (union across chains)
    - `num_unique_residues_dist_conf` — unique residues involved in distance + PAE pairs
    - `num_unique_residues_contact_prob` — unique residues involved in contact‑prob pairs

- Per‑run interactions: `<output_dir>/per_run/<run_name>.csv`
  - Each row is a cross‑chain residue pair meeting either criterion above.
  - Columns:
    - `run_name`, `model_index`, `chain_i`, `resid_i`, `chain_j`, `resid_j`
    - `min_distance` — minimal heavy‑atom residue–residue distance (Å)
    - `pae_avg` — average of `PAE[i,j]` and `PAE[j,i]` for the token pair
    - `contact_prob` — contact probability for the token pair
    - `passed_distance_conf` — 1 if passes distance + PAE, else 0
    - `passed_contact_prob` — 1 if passes contact prob threshold, else 0

## Design choises
- Minimal distances are computed over all heavy‑atom pairs between residues
- PAE gating uses the average of `PAE[i,j]` and `PAE[j,i]` from the AF3 full JSON.
- Residue‑level distances are computed only for standard amino acids parsed from the CIF.
