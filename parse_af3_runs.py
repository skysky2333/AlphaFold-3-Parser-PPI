#!/usr/bin/env python3
"""
AF3 run parser

Parses an AlphaFold 3 server output directory where each subdirectory is a run
containing multiple models (0..N). For each model, computes interaction metrics
based on geometric contacts (CIF), PAE (JSON), and contact_probs (JSON).

Outputs:
 - One CSV per run: rows are cross-chain residue-residue interactions that pass
   either (distance <= cutoff AND PAE <= cutoff) OR (contact_prob >= threshold)
   using only the best model by ranking_score for that run.
 - One aggregate CSV across all runs: one row per run summarizing counts and
   extremes (min passing distance, max contact prob, etc.).

Defaults:
 - distance_cutoff: 6.0 Å (heavy-atom contact)
 - pae_cutoff: 8.0 (average of PAE[i,j] and PAE[j,i])
 - contact_prob_threshold: 0.5 (used for counts; summary also reports max)

The script characterizes every run (best model only). If no interactions meet
thresholds, counts will be zero and distances may be NaN.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Set

import numpy as np
import pandas as pd

try:
    from Bio.PDB import MMCIFParser
    from Bio.PDB.Polypeptide import is_aa
except Exception as e:  # pragma: no cover - allow import on machines without BioPython during static reading
    MMCIFParser = None  # type: ignore
    def is_aa(residue, standard: bool = False) -> bool:  # type: ignore
        return False


# ------------- Utilities -------------

def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return json.load(f)


def _is_run_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        entries = os.listdir(path)
    except Exception:
        return False
    # Recognize AF3 run by presence of expected file name patterns
    has_summary = any(("_summary_confidences_" in fn and fn.endswith(".json")) for fn in entries)
    has_full = any(("_full_data_" in fn and fn.endswith(".json")) for fn in entries)
    has_cif = any(("_model_" in fn and fn.endswith(".cif")) for fn in entries)
    has_job = any(fn.endswith("_job_request.json") for fn in entries)
    return has_summary or has_full or has_cif or has_job


def _list_run_dirs(input_dir: str) -> List[str]:
    run_dirs = []
    for name in sorted(os.listdir(input_dir)):
        if name.startswith('.'):
            continue
        path = os.path.join(input_dir, name)
        if os.path.isdir(path) and _is_run_dir(path):
            run_dirs.append(path)
    return run_dirs


def _detect_models(run_dir: str) -> List[int]:
    # Detect model indices from full_data JSON files
    idxs = []
    for fn in os.listdir(run_dir):
        if fn.endswith(".json") and fn.endswith(".json") and "_full_data_" in fn:
            try:
                idx = int(fn.rsplit("_full_data_", 1)[1].split(".")[0])
                idxs.append(idx)
            except Exception:
                continue
    return sorted(set(idxs))


def _base_prefix(run_dir: str) -> str:
    # Determine the common base prefix like fold_<run_name>
    # e.g., files look like: fold_pos02_cdk8_znf143_full_data_0.json
    for fn in os.listdir(run_dir):
        if fn.endswith("_job_request.json"):
            return fn.rsplit("_job_request.json", 1)[0]
        if "_summary_confidences_" in fn and fn.endswith(".json"):
            return fn.rsplit("_summary_confidences_", 1)[0]
        if "_full_data_" in fn and fn.endswith(".json"):
            return fn.rsplit("_full_data_", 1)[0]
    # fallback: derive from CIF
    for fn in os.listdir(run_dir):
        if fn.endswith(".cif") and "_model_" in fn:
            return fn.rsplit("_model_", 1)[0]
    raise FileNotFoundError(f"Cannot determine base prefix in {run_dir}")


def _sym_pae(pae: np.ndarray, i: int, j: int) -> float:
    # Average to symmetrize
    return float((pae[i, j] + pae[j, i]) * 0.5)


def _get_tqdm(total: Optional[int] = None, desc: Optional[str] = None):
    """Return a tqdm progress bar if available, else a no-op stub."""
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(total=total, desc=desc)
    except Exception:
        class _Dummy:
            def update(self, n: int = 1):
                return None
            def close(self):
                return None
        return _Dummy()


# ------------- CIF parsing -------------

@dataclass
class AtomRec:
    chain_id: str
    res_id: int
    atom_name: str
    coord: np.ndarray  # shape (3,)


def load_cif_atoms(cif_path: str) -> Tuple[List[AtomRec], Dict[Tuple[str, int], List[int]]]:
    """
    Loads atoms from the CIF using Bio.PDB and returns:
    - flat list of AtomRec
    - mapping from (chain, resid) to indices in the atom list
    """
    if MMCIFParser is None:
        raise RuntimeError("Bio.PDB is required to parse CIF files. Please install biopython.")

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", cif_path)
    atoms: List[AtomRec] = []
    res_to_atom_idxs: Dict[Tuple[str, int], List[int]] = {}

    # Assume model id 0
    model = list(structure)[0]
    for chain in model:
        cid = chain.id
        for residue in chain:
            # Keep standard amino acids only for residue-level contact metrics
            if not is_aa(residue, standard=False):
                continue
            resid = residue.id[1]
            for atom in residue:
                coord = np.asarray(atom.coord, dtype=float)
                idx = len(atoms)
                atoms.append(AtomRec(chain_id=cid, res_id=resid, atom_name=atom.name, coord=coord))
                res_to_atom_idxs.setdefault((cid, resid), []).append(idx)

    return atoms, res_to_atom_idxs


def residue_min_distance(
    atoms: List[AtomRec], res_to_atom_idxs: Dict[Tuple[str, int], List[int]],
    chain_i: str, chain_j: str, resid_i: int, resid_j: int, *, cutoff: Optional[float] = None
) -> Optional[float]:
    idxs_i = res_to_atom_idxs.get((chain_i, resid_i))
    idxs_j = res_to_atom_idxs.get((chain_j, resid_j))
    if not idxs_i or not idxs_j:
        return None
    min_d = math.inf
    for ii in idxs_i:
        ci = atoms[ii].coord
        for jj in idxs_j:
            cj = atoms[jj].coord
            d = float(np.linalg.norm(ci - cj))
            if d < min_d:
                min_d = d
                # Early exit only on exact match
                if min_d == 0.0:
                    break
        if min_d == 0.0:
            break
    return min_d if min_d != math.inf else None


def contact_residue_pairs(
    atoms: List[AtomRec], res_to_atom_idxs: Dict[Tuple[str, int], List[int]],
    chain_a: str, chain_b: str, dist_cutoff: float
) -> Dict[Tuple[int, int], float]:
    """Return mapping {(resA, resB): min_distance} for pairs within cutoff.

    Implementation uses a straightforward per-residue min-distance check. This is
    reliable across environments and sufficiently fast for typical AF3 outputs.
    """
    result: Dict[Tuple[int, int], float] = {}
    resids_a = sorted({r for (c, r) in res_to_atom_idxs.keys() if c == chain_a})
    resids_b = sorted({r for (c, r) in res_to_atom_idxs.keys() if c == chain_b})
    for ra in resids_a:
        for rb in resids_b:
            d = residue_min_distance(atoms, res_to_atom_idxs, chain_a, chain_b, ra, rb, cutoff=dist_cutoff)
            if d is not None and d <= dist_cutoff:
                result[(ra, rb)] = d
    return result


# ------------- Token mapping -------------

@dataclass
class TokenMap:
    # token index -> (chain, resid)
    token_chain: List[str]
    token_resid: List[int]
    # (chain,resid) -> token index (first occurrence)
    resid_to_token: Dict[Tuple[str, int], int]
    chain_to_token_idxs: Dict[str, np.ndarray]


def build_token_map(full_json: dict) -> TokenMap:
    chains = full_json["token_chain_ids"]
    resids = full_json["token_res_ids"]
    token_chain: List[str] = list(chains)
    token_resid: List[int] = list(resids)
    resid_to_token: Dict[Tuple[str, int], int] = {}
    for i, (c, r) in enumerate(zip(token_chain, token_resid)):
        resid_to_token.setdefault((c, int(r)), i)
    chain_to_token_idxs: Dict[str, np.ndarray] = {}
    for c in sorted(set(token_chain)):
        chain_to_token_idxs[c] = np.array([i for i, cc in enumerate(token_chain) if cc == c], dtype=int)
    return TokenMap(token_chain=token_chain, token_resid=list(map(int, token_resid)), resid_to_token=resid_to_token, chain_to_token_idxs=chain_to_token_idxs)


# (intentionally left blank — previous unused metrics/model code removed)


def _choose_best_model_index(run_dir: str, base: str) -> Optional[int]:
    idxs = _detect_models(run_dir)
    best_idx = None
    best_rank = -float("inf")
    for idx in idxs:
        summary_path = os.path.join(run_dir, f"{base}_summary_confidences_{idx}.json")
        if not os.path.exists(summary_path):
            continue
        try:
            summary = _read_json(summary_path)
            rank = float(summary.get("ranking_score", -1e9))
            if (rank > best_rank) or (best_idx is None):
                best_rank = rank
                best_idx = idx
        except Exception:
            continue
    return best_idx


def _pairs_from_contact_probs(
    full: dict,
    chain_pairs: List[Tuple[str, str]],
    threshold: float,
) -> Dict[Tuple[str, int, str, int], float]:
    """Return mapping {(ci,ri,cj,rj): max_contact_prob} for pairs meeting threshold."""
    tmap = build_token_map(full)
    cprob = np.array(full.get("contact_probs"), dtype=float)
    pairs: Dict[Tuple[str, int, str, int], float] = {}
    for ci, cj in chain_pairs:
        ti = tmap.chain_to_token_idxs.get(ci)
        tj = tmap.chain_to_token_idxs.get(cj)
        if ti is None or tj is None or ti.size == 0 or tj.size == 0:
            continue
        sub = cprob[np.ix_(ti, tj)]
        hits = np.where(sub >= threshold)
        for ii, jj in zip(hits[0], hits[1]):
            ta = int(ti[ii])
            tb = int(tj[jj])
            ra = int(tmap.token_resid[ta])
            rb = int(tmap.token_resid[tb])
            key = (ci, ra, cj, rb)
            val = float(cprob[ta, tb])
            if key not in pairs or val > pairs[key]:
                pairs[key] = val
    return pairs


def _pairs_from_distance_and_pae(
    full: dict,
    cif_path: str,
    chain_pairs: List[Tuple[str, str]],
    dist_cutoff: float,
    pae_cutoff: float,
) -> Dict[Tuple[str, int, str, int], Tuple[float, float]]:
    """Return mapping {(ci,ri,cj,rj): (min_distance, pae_avg)} for pairs meeting both distance and PAE cutoffs."""
    tmap = build_token_map(full)
    pae = np.array(full.get("pae"), dtype=float)
    atoms, res_to_atom_idxs = load_cif_atoms(cif_path)

    result: Dict[Tuple[str, int, str, int], Tuple[float, float]] = {}
    for ci, cj in chain_pairs:
        # Find geometry contacts first
        geom = contact_residue_pairs(atoms, res_to_atom_idxs, ci, cj, dist_cutoff)

        # Check confidence via token PAE
        for (ra, rb), d in geom.items():
            ta = tmap.resid_to_token.get((ci, int(ra)))
            tb = tmap.resid_to_token.get((cj, int(rb)))
            if ta is None or tb is None:
                continue
            pae_avg = _sym_pae(pae, ta, tb)
            if pae_avg <= pae_cutoff:
                result[(ci, int(ra), cj, int(rb))] = (float(d), float(pae_avg))
    return result


def parse_run_best_model(
    run_dir: str,
    dist_cutoff: float,
    pae_cutoff: float,
    contact_prob_threshold: float,
) -> Tuple[str, int, dict, List[dict]]:
    """
    Returns:
      run_name, best_model_index, aggregate_metrics (dict), interaction_rows (list of dict)
    """
    run_name = os.path.basename(run_dir)
    base = _base_prefix(run_dir)
    best_idx = _choose_best_model_index(run_dir, base)
    if best_idx is None:
        return run_name, -1, {}, []

    summary_path = os.path.join(run_dir, f"{base}_summary_confidences_{best_idx}.json")
    full_path = os.path.join(run_dir, f"{base}_full_data_{best_idx}.json")
    cif_path = os.path.join(run_dir, f"{base}_model_{best_idx}.cif")
    if not (os.path.exists(summary_path) and os.path.exists(full_path) and os.path.exists(cif_path)):
        return run_name, best_idx, {}, []

    summary = _read_json(summary_path)
    full = _read_json(full_path)
    tmap = build_token_map(full)
    pae = np.array(full.get("pae"), dtype=float)
    cprob = np.array(full.get("contact_probs"), dtype=float)

    chain_ids = sorted(set(tmap.token_chain))
    # Use unordered unique chain pairs (avoid duplicate A-B and B-A)
    chain_pairs = [(chain_ids[i], chain_ids[j]) for i in range(len(chain_ids)) for j in range(i+1, len(chain_ids))]

    # Collect pairs from both criteria
    pairs_dist_pae = _pairs_from_distance_and_pae(full, cif_path, chain_pairs, dist_cutoff, pae_cutoff)
    pairs_cprob = _pairs_from_contact_probs(full, chain_pairs, contact_prob_threshold)

    # Union pairs, annotate details
    atoms, res_to_atom_idxs = load_cif_atoms(cif_path)
    interactions: Dict[Tuple[str, int, str, int], dict] = {}
    for key, (d, pavg) in pairs_dist_pae.items():
        ci, ri, cj, rj = key
        ta = tmap.resid_to_token.get((ci, ri))
        tb = tmap.resid_to_token.get((cj, rj))
        cp = float(cprob[ta, tb]) if (ta is not None and tb is not None) else float("nan")
        interactions[key] = {
            "run_name": run_name,
            "model_index": best_idx,
            "chain_i": ci,
            "resid_i": int(ri),
            "chain_j": cj,
            "resid_j": int(rj),
            "min_distance": float(d),
            "pae_avg": float(pavg),
            "contact_prob": cp,
            "passed_distance_conf": 1,
            "passed_contact_prob": 1 if key in pairs_cprob else 0,
        }

    for key, cp in pairs_cprob.items():
        if key in interactions:
            # Fill in/overwrite contact_prob if higher
            interactions[key]["contact_prob"] = max(float(cp), float(interactions[key]["contact_prob"]))
            interactions[key]["passed_contact_prob"] = 1
            continue
        ci, ri, cj, rj = key
        # Compute min distance for info (without cutoff; may be > cutoff)
        d = residue_min_distance(atoms, res_to_atom_idxs, ci, cj, int(ri), int(rj))
        # Compute PAE avg
        ta = tmap.resid_to_token.get((ci, int(ri)))
        tb = tmap.resid_to_token.get((cj, int(rj)))
        pavg = _sym_pae(pae, ta, tb) if (ta is not None and tb is not None) else float("nan")
        interactions[key] = {
            "run_name": run_name,
            "model_index": best_idx,
            "chain_i": ci,
            "resid_i": int(ri),
            "chain_j": cj,
            "resid_j": int(rj),
            "min_distance": float(d) if d is not None else float("nan"),
            "pae_avg": float(pavg),
            "contact_prob": float(cp),
            "passed_distance_conf": 0,
            "passed_contact_prob": 1,
        }

    # Aggregate per-run metrics
    num_pairs_dist_conf = sum(1 for v in interactions.values() if v["passed_distance_conf"] == 1)
    num_pairs_cprob = sum(1 for v in interactions.values() if v["passed_contact_prob"] == 1)
    min_distance_conf_vals = [v["min_distance"] for v in interactions.values() if v["passed_distance_conf"] == 1 and not math.isnan(v["min_distance"])]
    min_distance_conf = min(min_distance_conf_vals) if min_distance_conf_vals else float("nan")
    # Max contact prob across cross-chain tokens only
    max_contact_prob_vals: List[float] = []
    for ci, cj in chain_pairs:
        ti = tmap.chain_to_token_idxs.get(ci)
        tj = tmap.chain_to_token_idxs.get(cj)
        if ti is None or tj is None or ti.size == 0 or tj.size == 0:
            continue
        sub = cprob[np.ix_(ti, tj)]
        if sub.size:
            try:
                max_contact_prob_vals.append(float(np.max(sub)))
            except ValueError:
                pass
    max_contact_prob = max(max_contact_prob_vals) if max_contact_prob_vals else float("nan")

    # Max contact prob among pairs that passed the contact_prob threshold
    max_contact_prob_passed = max(pairs_cprob.values()) if pairs_cprob else float("nan")

    # Unique residues involved (union across chains)
    def _unique_residues(filter_key: Optional[str] = None) -> int:
        seen: Set[Tuple[str, int]] = set()
        for v in interactions.values():
            if filter_key is None or v[filter_key] == 1:
                seen.add((v["chain_i"], int(v["resid_i"])))
                seen.add((v["chain_j"], int(v["resid_j"])))
        return len(seen)

    num_unique_residues_any = _unique_residues(None)
    num_unique_residues_dist_conf = _unique_residues("passed_distance_conf")
    num_unique_residues_contact_prob = _unique_residues("passed_contact_prob")

    agg = {
        "run_name": run_name,
        "model_index": best_idx,
        "ranking_score": float(summary.get("ranking_score", float("nan"))),
        "ptm": float(summary.get("ptm", float("nan"))),
        "iptm": float(summary.get("iptm", float("nan"))),
        "num_pairs_distance_conf": int(num_pairs_dist_conf),
        "num_pairs_contact_prob": int(num_pairs_cprob),
        "min_distance_conf": float(min_distance_conf),
        "max_contact_prob": float(max_contact_prob),
        "max_contact_prob_passed": float(max_contact_prob_passed),
        "num_unique_residues_any": int(num_unique_residues_any),
        "num_unique_residues_dist_conf": int(num_unique_residues_dist_conf),
        "num_unique_residues_contact_prob": int(num_unique_residues_contact_prob),
    }

    interaction_rows = list(interactions.values())
    return run_name, best_idx, agg, interaction_rows


# (intentionally left blank — removed unused aggregate helper)


def main():
    ap = argparse.ArgumentParser(description="Parse AF3 runs, pick best model per run, output per-run interactions and an aggregate table.")
    ap.add_argument("input_dir", help="Path to a directory containing AF3 run subdirectories")
    ap.add_argument("--distance_cutoff", type=float, default=6.0, help="Contact distance cutoff in Å (default 6.0)")
    ap.add_argument("--pae_cutoff", type=float, default=8.0, help="PAE cutoff for confidence gating (default 8.0)")
    ap.add_argument("--contact_prob_threshold", type=float, default=0.5, help="Threshold for counting high contact probability (used in future extensions)")
    ap.add_argument("--output_dir", default=None, help="Directory to write outputs (default: <script_dir>/outputs/<input_basename>)")
    args = ap.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "outputs", os.path.basename(input_dir.rstrip(os.sep)))
    os.makedirs(out_dir, exist_ok=True)

    runs = _list_run_dirs(input_dir)
    run_summary_rows: List[dict] = []

    pbar = _get_tqdm(total=len(runs), desc="Runs")
    for run_dir in runs:
        run_name, best_idx, agg, interaction_rows = parse_run_best_model(
            run_dir,
            dist_cutoff=args.distance_cutoff,
            pae_cutoff=args.pae_cutoff,
            contact_prob_threshold=args.contact_prob_threshold,
        )
        # Update progress bar per run
        try:
            pbar.update(1)
        except Exception:
            pass

        # Write per-run CSV (interactions)
        per_run_dir = os.path.join(out_dir, "per_run")
        os.makedirs(per_run_dir, exist_ok=True)
        per_run_path = os.path.join(per_run_dir, f"{run_name}.csv")
        pd.DataFrame(interaction_rows).to_csv(per_run_path, index=False)

        if agg:
            run_summary_rows.append(agg)
        else:
            run_summary_rows.append({
                "run_name": run_name,
                "model_index": best_idx,
                "ranking_score": float("nan"),
                "ptm": float("nan"),
                "iptm": float("nan"),
                "num_pairs_distance_conf": 0,
                "num_pairs_contact_prob": 0,
                "min_distance_conf": float("nan"),
                "max_contact_prob": float("nan"),
                "max_contact_prob_passed": float("nan"),
                "num_unique_residues_any": 0,
                "num_unique_residues_dist_conf": 0,
                "num_unique_residues_contact_prob": 0,
            })

    # Write CSVs
    def to_csv(rows: List[dict], path: str):
        if rows:
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame()
        df.to_csv(path, index=False)

    to_csv(run_summary_rows, os.path.join(out_dir, "runs_summary.csv"))

    try:
        pbar.close()
    except Exception:
        pass

    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
