# ML Screening of Metal–Organic Frameworks for Hydrogen Storage

Machine-learning models that predict cryogenic hydrogen uptake in metal–organic frameworks (MOFs), so that expensive grand-canonical Monte Carlo (GCMC) simulations are spent only on the most promising materials.

**Data:** 137,953 hypothetical MOFs (hMOF) with simulated H₂ isotherms at 77 K (MOFDB release of Bobbitt & Snurr, *J. Phys. Chem. C* 2016, 120, 27328).
**Targets:** uptake at 77 K / 2 bar, 77 K / 100 bar, and deliverable capacity (DC = 100 bar − 2 bar), all in g/L.

## Key results

Deliverable capacity, end-to-end pipeline (zero/non-zero classifier + regression), cleaned held-out test set:

| Split | Model | R² | MAE (g/L) | Top-5% recall |
|---|---|---|---|---|
| Random | XGBoost (geometry) | 0.972 | 1.84 | 0.67 |
| Random | XGBoost (geometry + composition) | **0.994** | **0.79** | **0.90** |
| Random | GNN (graph only) | 0.935 | 2.48 | 0.43 |
| Random | GNN (graph + global descriptors) | 0.994 | 0.85 | 0.87 |
| Random | GNN ensemble (4 seeds) | 0.994 | 0.83 | 0.88 |
| Building-block grouped | XGBoost (geometry + composition) | **0.994** | **0.79** | **0.89** |
| Building-block grouped | GNN (graph + global descriptors) | 0.994 | 0.86 | 0.87 |
| Leave-topology-out | XGBoost (geometry + composition) | 0.984 | 1.24 | 0.88 |
| Leave-topology-out | GNN (graph + global descriptors) | **0.987** | **1.12** | 0.88 |

**Findings**
- **Pore geometry plus composition is enough on familiar chemistry.** A gradient-boosted model on six pore descriptors and composition statistics matches the GNN on random and building-block splits.
- **Composition matters most at low pressure.** Adding composition raises R² at 2 bar from 0.90 to 0.98 (binding-site chemistry), while 100-bar uptake is dominated by pore volume.
- **Graph-only models miss pore-scale information.** Mean-pooled local atomic environments cannot distinguish frameworks with similar chemistry but different pore sizes (R² 0.935 vs 0.994 with global descriptors).
- **The GNN generalises somewhat better to unseen topologies** (DC MAE 1.12 vs 1.24 g/L). This is a single-seed result and still being tested over multiple seeds.
- **Ensemble uncertainty tracks error.** Mean absolute error rises from 0.64 to 1.05 g/L across uncertainty quintiles, although the raw ensemble spread underestimates the error and needs calibration.

## Data quality: physically inconsistent labels

For total adsorption in accessible pores, uptake cannot fall below the bulk-gas contribution (void fraction × bulk H₂ density, 31.3 g/L at 77 K and 100 bar). Accessible MOFs (pore-limiting diameter ≥ 3.2 Å) normally sit at 1.5–3× this floor; a separate cluster sits at 0.1–0.5×.

- **159 labels (0.12%) flagged**: 122 below 50% of the physical floor and 81 with deliverable capacity below −1 g/L (some overlap).
- **91 of the 122 floor violations come from a single ID block (hMOF-3001xxx)**, which points to a batch-level simulation or processing issue.
- Removing them lowers deliverable-capacity MAE from 0.814 to 0.789 g/L on the same test set.
- These labels are flagged as **suspected** errors; GCMC re-simulation with the original settings (UFF + Darkrim–Levesque H₂) is in progress.

Zero uptake (3,114 MOFs) is explained by pore size: almost all have pore-limiting diameters below the size of an H₂ molecule. A geometry-only classifier separates them with 99.7% accuracy on the random split.

## Pipeline

1. **Dataset deep dive** (`hmof_dataset_deep_dive.ipynb`): schema, available conditions, label quality, descriptor correlations, MOFkey topology parsing, near-duplicate check.
2. **Modelling** (`hmof_h2_modelling.ipynb`):
   - label cleaning with a physics-based lower bound
   - three splits: random, grouped by building blocks (MOFkey), and leave-topology-out
   - stage 1: zero / non-zero classifier on pore geometry
   - stage 2: XGBoost baselines and a multi-task graph attention network with a global-descriptor branch
   - ablation (graph only vs graph + descriptors), ensemble uncertainty, error by topology

## How to run

The notebooks are written for Kaggle (GPU T4 × 2, Internet on).

1. Run `hmof_dataset_deep_dive.ipynb` to download the data and reproduce the exploratory analysis.
2. Run `hmof_h2_modelling.ipynb`. Graph featurization is cached in `/kaggle/working/model/graph_chunks`; attach the saved output as an input in later sessions to skip it. A full run (4 GNN runs + 3 ensemble members) takes about 7 hours.

Main dependencies: `pymatgen`, `torch_geometric`, `xgboost`, `CoolProp`, `scikit-learn`, `pandas`.

## Limitations

- hMOF structures are hypothetical and cover a limited set of metals and linkers; predictions for real MOFs are only reliable inside this chemical domain.
- The pcu topology dominates the dataset, so the leave-topology-out test holds out smaller topologies only. MOFs with unidentified topology (`ERROR`, `UNKNOWN`) should stay in the training set; this fix is in progress.
- Deliverable capacity here is an isothermal pressure swing at 77 K (100 → 2 bar), because the dataset contains no 160 K data; the field's common standard is 77 K/100 bar → 160 K/5 bar.

## Next steps

- Screen real MOFs from CoRE MOF 2024 with Zeo++ descriptors calibrated against the hMOF values and an applicability-domain check.
- Verify suspected label errors and top-ranked candidates with RASPA GCMC, after reproducing known hMOF labels as a control.
- Repeat the leave-topology-out comparison over several seeds.

## Author

Mohd Hasnain, Dual Degree (B.Tech + M.Tech) in Materials Science and Engineering, NIT Patna.
