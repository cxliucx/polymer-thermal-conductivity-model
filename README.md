# Polymer thermal conductivity descriptor model

This repository contains the processed dataset and Python scripts used to generate descriptors and perform exhaustive four-descriptor model selection for chain-direction thermal conductivity of crystalline polymers.

## Files

- `base_5.xlsx`: Metadata and baseline descriptor inputs used by `build_descriptors.py`.
- `build_descriptors.py`: Descriptor-generation script. It reads `base_5.xlsx` and structural CIF files, then writes `157_all_des_kmd.xlsx`.
- `157_all_des_kmd.xlsx`: Processed 157-polymer dataset containing thermal conductivity values and descriptors used by the fitting script.
- `exhaustive_search.py`: Exhaustive four-descriptor search script. It reads `157_all_des_kmd.xlsx`, enumerates all `C(16,4)=1820` descriptor combinations, ranks them by 5-fold cross-validation loss within the training set, refits the selected model on the full training set, and evaluates it on the independent test set.

## Requirements

Install the required Python packages:

```bash
pip install numpy pandas scipy scikit-learn matplotlib openpyxl pymatgen
```

## Descriptor generation

Place `base_5.xlsx`, `build_descriptors.py`, and the CIF structure folders in the same working directory. The script searches for CIF files in folders named `cifx`, `cify`, `cifz`, `cifs_x`, `cifs_y`, or `cifs_z`.

Run:

```bash
python build_descriptors.py
```

The default output file is:

```text
157_all_des_kmd.xlsx
```

Note: `base_5.xlsx` alone is not sufficient to recompute all structure-derived descriptors. The corresponding CIF structure files must also be obtained from the dataset reported in A Polymer Dataset for Accelerated Property Prediction and Design. Therefore, the processed dataset 157_all_des_kmd.xlsx is provided directly to support reproducibility of the model fitting.

## Exhaustive model search

After `157_all_des_kmd.xlsx` is available, run:

```bash
python exhaustive_search.py
```

The script performs the following workflow:

1. Read `157_all_des_kmd.xlsx`.
2. Split the 157 polymers into an 80% training set and a 20% independent test set using stratified sampling based on thermal-conductivity terciles.
3. Enumerate all `C(16,4)=1820` four-descriptor combinations from the 16 candidate correction descriptors.
4. Rank each combination by 5-fold cross-validation loss within the training set.
5. Select the best descriptor combination for each route.
6. Refit the selected model using the full training set.
7. Evaluate the final model once on the independent test set.

The default output folder is:

```text
exhaustive_best80_20_complete_outputs
```

The default compressed output archive is:

```text
exhaustive_best80_20_complete_outputs.zip
```

## Error metrics

The script reports MALE and RMSLE based on the signed log-ratio error:

```text
e_i = log(kappa_MD_i / kappa_TM_i)
MALE = mean(abs(e_i))
RMSLE = sqrt(mean(e_i^2))
```
