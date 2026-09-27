# Water-Quality-Monitor

# Phase 2 — Water Quality Estimation: Documentation

## Objective
Predict water quality indicators (turbidity, chlorophyll-a, CDOM) from Sentinel-2 satellite reflectance data, using real ground-truth measurements to train and validate the models.

## Data Source
**USGS Aquatic Reflectance Dataset** — Sentinel-2 satellite reflectance matched to continuous water quality sensor measurements across 5 US river basins (Delaware, Illinois, Trinity, Upper Colorado, Willamette), July 2015–September 2024.
- 2 CSV files, 1,748,467 combined rows, 85 columns
- Each row: one ground sensor reading + reflectance values across 11 Sentinel-2 spectral bands (raw pixel value + 250m buffer mean/std/median per band)

## Data Cleaning

**Validity filtering.** Only ~32% of rows (566,638 / 1,748,467) are usable — the rest are cloud-obscured or otherwise invalid. Validity is marked by the `l2flag_center` column (1 = valid, 0 = invalid), **not** by null values in the band columns — an early check using `.isna()` on a band column incorrectly suggested 100% of data was usable, since invalid rows still contain populated (but unreliable) numbers rather than nulls. This was caught and corrected before any modeling.

**Unit mismatches.** Both the chlorophyll and CDOM parameters in this dataset mix two incompatible units: RFU (arbitrary relative fluorescence, sensor-specific) and µg/L (real physical concentration). All modeling used µg/L-only subsets.

**Station coverage check.** A candidate 4th target — suspended sediment (a TSS proxy) — was excluded after discovering its 114,902 rows came from a single monitoring station, making it unusable for a generalizable model (no station diversity to validate against).

## Methodology

**Train/test splitting:** All models were evaluated using station-based splitting (GroupKFold, grouped by `station_nm`), never random row-based splitting. This ensures the model is tested on rivers/stations it never saw during training — a stricter, more honest test of generalization than a random split, which could otherwise let the model "leak" information about a station across both sets.

**Evaluation approach:** Per-fold R² scores were found to be statistically unstable when fold sizes vary (e.g., a fold with only 1 test station gives a noisy, unreliable R²). The corrected approach pools every out-of-fold prediction across all folds into one combined set, then computes R²/MAE once — giving each data point equal weight rather than each fold equal weight.

**Features used:** Raw band reflectance (11 Sentinel-2 bands), 250m buffer-averaged reflectance per band, and engineered spectral indices — NDTI (turbidity-oriented), green/red ratio, NIR/red ratio, and NDCI (chlorophyll-oriented, using the red-edge band).

**Target transformation:** All three target variables (turbidity, chlorophyll, CDOM) are heavily right-skewed (a small number of extreme pollution-event readings alongside mostly low/clear values). Targets were log-transformed (`log1p`) before training and predictions were converted back to original units (`expm1`) for evaluation.

## Results

### Turbidity — Successful
- **Pooled R² = 0.42, MAE ≈ 10.4 FNU**, evaluated across 44 stations
- Feature importance analysis showed **Band 5 (red-edge, ~705nm)** dominated the model's predictions (54% of total importance), consistent with published remote-sensing literature on how suspended particles scatter light in that spectral range
- Engineered NIR/red ratio contributed meaningfully (15% importance); the literature-standard NDTI index contributed less than expected (2.6%) — a genuine, dataset-specific finding rather than an assumption
- Compared Random Forest (R²=0.42) vs. Gradient Boosting (R²=0.40, but lower MAE) — Random Forest selected as primary model for better overall variance explained
- **Conclusion: turbidity is reliably predictable from Sentinel-2 reflectance in this multi-basin dataset.** Model and feature list saved (`turbidity_model.pkl`, `turbidity_features.pkl`)

### Chlorophyll — Did Not Generalize
Tested across four independent approaches, escalating in rigor:

1. **Baseline single station-split:** R² = -0.81 — worse than predicting the mean
2. **Diagnosis:** the specific 80/20 split had put disproportionately high-chlorophyll stations (Illinois rivers) into the test set versus low-chlorophyll stations (Delaware rivers) into training — a distribution mismatch, not necessarily a broken model
3. **5-fold grouped cross-validation (pooled):** R² = -0.25 — still negative under a statistically fair evaluation, ruling out "unlucky split" as the sole explanation
4. **Per-basin models** (hypothesis: regional water chemistry differences were the cause): Basin 1709 R² ≈ 0.00, Basin 712 R² = -2.00, pooled smaller basins R² = -1.43 — per-basin modeling did not fix generalization, and performed worse in most cases due to reduced training data per basin
5. **Reframed as 3-class classification** (Low/Medium/High, percentile-based bins, hypothesis: coarser targets might be more learnable): **31% accuracy** — at chance level for 3 balanced classes (33%). Confusion matrix showed the model could weakly separate "Low" from everything else, but almost never correctly identified "High" (68% misclassified as "Medium")

**Conclusion:** chlorophyll fluorescence measurements did not generalize from Sentinel-2 reflectance in this dataset, regardless of modeling approach. Likely contributing factors: fewer stations than turbidity (25 vs. 44), inherently noisier fluorescence-based ground truth, and algae composition/type varying by region and season in ways not captured by 11 spectral bands alone.

### CDOM — Did Not Generalize
- Isolated 35,144 µg/L (QSE) rows across 7 stations (all 7 stations retained after unit filtering)
- Pooled 5-fold grouped cross-validation: **R² = -0.08, MAE ≈ 13.7 µg/L** (against an interquartile range of ~9–31 µg/L)
- Better than chlorophyll's result but still not a working model
- Not pursued further given the very limited station count (7) offers little room for additional diagnostic approaches (e.g., per-basin modeling is not viable with this few stations)

## Overall Phase 2 Conclusion

A consistent pattern emerged across all three targets: **the parameter with the most stations and most direct optical signal (turbidity, light scattering off particles) generalized well; the two relying on fluorescence-based sensors with fewer stations (chlorophyll, CDOM) did not**, despite feature engineering, proper cross-validation, regional modeling, and reframing as classification.

**Final scope:** the app's live prediction pipeline uses the turbidity model only. Chlorophyll and CDOM are documented as rigorously tested negative results rather than shipped as unreliable predictions — a deliberate scope decision favoring one trustworthy indicator over three unreliable ones.

## Artifacts Produced
- `turbidity_model.pkl`, `turbidity_features.pkl` — trained model ready for use in the app
- `data_cleaning.ipynb` — full pipeline notebook (load → clean → feature engineer → train → evaluate → save)