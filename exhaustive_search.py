from __future__ import annotations
import itertools
import json
import math
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

INPUT_XLSX = Path('157_all_des_kmd.xlsx')
OUTDIR = Path('exhaustive_best80_20_complete_outputs')
ZIPFILE = Path('exhaustive_best80_20_complete_outputs.zip')
ROUTES = ['route1_fixed_f1', 'route2_refit_f1_f2']
CANDIDATE_F2_DESCRIPTORS = ['P', 'δ', 'Ms', 'C', 'L', 'S', 'Nρ', 'Mρ', 'ρN', 'Φ', 'ρin', 'Lb', 'θb', 'Ψ', 'χ', 'Lstb']
TEST_SIZE = 0.2
KFOLD = 5
SEED = 20260316
CV_RANDOM_STATE = SEED + 50000
MAX_PREFACTOR_A = 100000.0
LOGA_BOUNDS = (-50.0, math.log(MAX_PREFACTOR_A))
EXPONENT_BOUNDS = (-8.0, 8.0)
N_MULTISTARTS = 4
MAXITER = 800
FTOL = 1e-12
GTOL = 1e-8
MAXLS = 30
SCATTER_DPI = 300
SCATTER_FIGSIZE = (5.2, 5.0)

@dataclass(frozen=True)
class LossParams:
    omega_mae: float
    omega_bal: float
    omega_mean: float
    omega_l2: float
    sigmoid_alpha: float

ROUTE_LOSS_PARAMS = {
    'route1_fixed_f1': LossParams(0.2, 0.05, 0.05, 1e-7, 6.0),
    'route2_refit_f1_f2': LossParams(0.2, 0.05, 0.05, 0.001, 6.0),
}

COLUMN_ALIASES = {
    'kmd': ['kmd', 'KMD', 'κMD', 'k_MD', 'kappa_MD', 'kappa', 'k', 'thermal_conductivity'],
    'T': ['T', 'Temp', 'Temperature', 'temperature'],
    'Ejn': ['Ejn', 'EJN', 'E_jn', 'Eavg', 'E'],
    'V': ['V', 'Volume', 'volume'],
    'Mb': ['Mb', 'M_b', 'MB', 'backbone_mass', 'Mbackbone'],
    'Ms': ['Ms', 'M_s', 'MS', 'Mside', 'side_mass', 'average_side_chain_mass', 'average side-chain mass'],
    'δ': ['δ', 'delta', 'Delta', 'd'],
    'Nρ': ['Nρ', 'Nrho', 'N_rho', 'N rho'],
    'Mρ': ['Mρ', 'Mrho', 'M_rho', 'M rho'],
    'ρN': ['ρN', 'rhoN', 'rho_N', 'rho n', 'atomic_number_density'],
    'Φ': ['Φ', 'Phi', 'phi', 'packing_fraction', 'packing fraction'],
    'ρin': ['ρin', 'rhoin', 'rho_in', 'rho in'],
    'θb': ['θb', 'thetab', 'theta_b', 'theta b', 'θ_b'],
    'Ψ': ['Ψ', 'Psi', 'psi'],
    'Lstb': ['Lstb', 'L_stb', 'L st b', 'Lstb_mean'],
}

def normalize_colname(x: str) -> str:
    return re.sub(r'\s+', '', str(x).strip()).lower()

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    norm_to_original = {normalize_colname(c): c for c in df.columns}
    renames = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            key = normalize_colname(alias)
            if key in norm_to_original:
                renames[norm_to_original[key]] = canonical
                break
    if renames:
        df = df.rename(columns=renames)
    return df

def resolve_input_path(filename: str | Path) -> Path:
    paths = [Path(filename), Path(__file__).resolve().parent / filename, Path.cwd() / filename]
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f'Cannot find input file: {filename}')

def positive_finite_mask(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for c in cols:
        v = pd.to_numeric(df[c], errors='coerce').to_numpy(float)
        mask &= np.isfinite(v) & (v > 0)
    return mask

def load_clean_dataframe(input_xlsx: str | Path) -> pd.DataFrame:
    df = standardize_columns(pd.read_excel(resolve_input_path(input_xlsx)))
    required = ['kmd', 'T', 'Ejn', 'V', 'Mb'] + CANDIDATE_F2_DESCRIPTORS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    mask = positive_finite_mask(df, required)
    df = df.loc[mask].reset_index(drop=True).copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
    if 'id' not in df.columns:
        df.insert(0, 'id', np.arange(1, len(df) + 1))
    return df

def make_tercile_labels(y: np.ndarray) -> np.ndarray:
    q1, q2 = np.quantile(np.asarray(y, dtype=float), [1 / 3, 2 / 3])
    labels = np.zeros(len(y), dtype=int)
    labels[y > q1] = 1
    labels[y > q2] = 2
    return labels

def train_test_split_stratified(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strata = make_tercile_labels(df['kmd'].to_numpy(float))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df, strata))
    train_df = df.iloc[train_idx].reset_index(drop=True).copy()
    test_df = df.iloc[test_idx].reset_index(drop=True).copy()
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    return train_df, test_df

def prepare_arrays(df: pd.DataFrame) -> dict:
    cols = ['kmd', 'T', 'Ejn', 'V', 'Mb'] + CANDIDATE_F2_DESCRIPTORS
    log_cols = {c: np.log(df[c].to_numpy(float)) for c in cols}
    y = df['kmd'].to_numpy(float)
    seg_idx = [idx.astype(int) for idx in np.array_split(np.argsort(y), 3)]
    return {'n': len(df), 'y': y, 'logy': log_cols['kmd'], 'log_cols': log_cols, 'seg_idx': seg_idx}

def model_design(data: dict, route: str, combo: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    lc = data['log_cols']
    if route == 'route1_fixed_f1':
        base = 1.5 * lc['Ejn'] - 2.0 / 3.0 * lc['V'] - 0.5 * lc['Mb'] - lc['T']
        cols = list(combo)
    elif route == 'route2_refit_f1_f2':
        base = -lc['T']
        cols = ['Ejn', 'V', 'Mb'] + list(combo)
    else:
        raise ValueError(f'Unknown route: {route}')
    X = np.column_stack([np.ones(data['n'])] + [lc[c] for c in cols])
    return base, X, data['y'], cols

def theta_bounds(n: int) -> list[tuple[float, float]]:
    return [LOGA_BOUNDS] + [EXPONENT_BOUNDS] * (n - 1)

def clip_theta(theta: np.ndarray) -> np.ndarray:
    bounds = theta_bounds(len(theta))
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return np.clip(np.asarray(theta, dtype=float), lo, hi)

def initial_theta_ols(base: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    target = np.log(y) - base
    theta, *_ = np.linalg.lstsq(X, target, rcond=None)
    return clip_theta(theta)

def segment_loss_grad(theta: np.ndarray, base: np.ndarray, X: np.ndarray, y: np.ndarray, logy: np.ndarray, seg_idx: list[np.ndarray], lp: LossParams, eps: float = 1e-10) -> tuple[float, np.ndarray]:
    e = base + X @ theta - logy
    total = 0.0
    grad = np.zeros_like(theta, dtype=float)
    ns = float(len(seg_idx))
    for idx in seg_idx:
        Xm = X[idx]
        em = e[idx]
        n = float(len(em))
        rms = math.sqrt(float(np.mean(em * em)) + eps)
        total += rms / ns
        grad += Xm.T @ em / (ns * n * rms)
        denom = np.sqrt(em * em + eps)
        total += lp.omega_mae * float(np.mean(denom)) / ns
        grad += lp.omega_mae * (Xm.T @ (em / denom)) / (ns * n)
        sig = 1.0 / (1.0 + np.exp(-np.clip(lp.sigmoid_alpha * em, -50, 50)))
        p_over = float(np.mean(sig))
        sig_der = lp.sigmoid_alpha * sig * (1.0 - sig)
        dp = Xm.T @ sig_der / n
        total += lp.omega_bal * (p_over - 0.5) ** 2 / ns
        grad += lp.omega_bal * 2.0 * (p_over - 0.5) * dp / ns
        bias = float(np.mean(em))
        total += lp.omega_mean * bias ** 2 / ns
        grad += lp.omega_mean * 2.0 * bias * np.mean(Xm, axis=0) / ns
    total += lp.omega_l2 * float(np.sum(theta[1:] ** 2))
    grad[1:] += 2.0 * lp.omega_l2 * theta[1:]
    return float(total), grad

def fit_lbfgsb_multistart(data: dict, route: str, combo: tuple[str, ...], lp: LossParams, seed: int) -> tuple[np.ndarray, float, str, list[str], int]:
    base, X, y, cols = model_design(data, route, combo)
    theta_ols = initial_theta_ols(base, X, y)
    starts = [theta_ols]
    if route == 'route2_refit_f1_f2':
        base1, X1, y1, _ = model_design(data, 'route1_fixed_f1', combo)
        theta1_ols = initial_theta_ols(base1, X1, y1)
        starts.append(clip_theta(np.array([theta1_ols[0], 1.5, -2.0 / 3.0, -0.5] + list(theta1_ols[1:]), dtype=float)))
    rng = np.random.default_rng(seed)
    while len(starts) < N_MULTISTARTS:
        base_start = starts[0]
        jitter = rng.normal(scale=0.2, size=len(theta_ols))
        jitter[0] *= 0.25
        starts.append(clip_theta(base_start + jitter))
    bounds = theta_bounds(len(theta_ols))
    def fun(th: np.ndarray) -> float:
        return segment_loss_grad(th, base, X, y, data['logy'], data['seg_idx'], lp)[0]
    def jac(th: np.ndarray) -> np.ndarray:
        return segment_loss_grad(th, base, X, y, data['logy'], data['seg_idx'], lp)[1]
    best_obj = float('inf')
    best_theta = theta_ols
    best_success = False
    best_message = 'ols'
    n_success = 0
    for st in starts:
        res = minimize(fun, st, jac=jac, method='L-BFGS-B', bounds=bounds, options={'maxiter': MAXITER, 'ftol': FTOL, 'gtol': GTOL, 'maxls': MAXLS})
        if np.isfinite(res.fun):
            if res.success:
                n_success += 1
            if float(res.fun) < best_obj:
                best_obj = float(res.fun)
                best_theta = np.asarray(res.x, dtype=float)
                best_success = bool(res.success)
                best_message = str(res.message)
    best_theta = clip_theta(best_theta)
    status = 'success' if best_success else f'not_converged_best:{best_message}'
    return best_theta, float(fun(best_theta)), status, cols, n_success

def predict_on_data(data: dict, route: str, combo: tuple[str, ...], theta: np.ndarray) -> np.ndarray:
    base, X, _, _ = model_design(data, route, combo)
    return np.exp(np.clip(base + X @ clip_theta(theta), -50, 50))

def metric_dict(data: dict, route: str, combo: tuple[str, ...], theta: np.ndarray, lp: LossParams) -> dict:
    y = data['y']
    yp = predict_on_data(data, route, combo, theta)
    signed = np.log(y / yp)
    ratio = yp / y
    base, X, _, _ = model_design(data, route, combo)
    L = segment_loss_grad(clip_theta(theta), base, X, y, data['logy'], data['seg_idx'], lp)[0]
    return {
        'L': float(L),
        'MALE': float(np.mean(np.abs(signed))),
        'RMSLE': float(np.sqrt(np.mean(signed ** 2))),
        'mean_log_bias_MD_over_pred': float(np.mean(signed)),
        'median_abs_log_ratio': float(np.median(np.abs(signed))),
        'MAE_linear': float(np.mean(np.abs(yp - y))),
        'RMSE_linear': float(np.sqrt(np.mean((yp - y) ** 2))),
        'frac_within_50pct': float(np.mean(np.abs(ratio - 1.0) <= 0.5)),
        'frac_within_100pct': float(np.mean(np.abs(ratio - 1.0) <= 1.0)),
        'frac_within_200pct': float(np.mean(np.abs(ratio - 1.0) <= 2.0)),
        'frac_within_factor2': float(np.mean((ratio >= 0.5) & (ratio <= 2.0))),
    }

def prefactor_A(theta: np.ndarray) -> float:
    return float(math.exp(float(clip_theta(theta)[0])))

def max_abs_exponent(theta: np.ndarray) -> float:
    return float(np.max(np.abs(clip_theta(theta)[1:])))

def theta_to_formula(route: str, combo: tuple[str, ...], theta: np.ndarray) -> str:
    theta = clip_theta(theta)
    A = prefactor_A(theta)
    if route == 'route1_fixed_f1':
        parts = [f'k = {A:.10g} * Ejn^1.5 * V^-0.6666667 * Mb^-0.5 * T^-1']
        cols = list(combo)
    else:
        parts = [f'k = {A:.10g} * T^-1']
        cols = ['Ejn', 'V', 'Mb'] + list(combo)
    for c, b in zip(cols, theta[1:]):
        parts.append(f' * {c}^{float(b):+.6f}')
    return ''.join(parts)

def prediction_dataframe(df: pd.DataFrame, route: str, combo: tuple[str, ...], theta: np.ndarray, split: str) -> pd.DataFrame:
    data = prepare_arrays(df)
    out = df.copy()
    yp = predict_on_data(data, route, combo, theta)
    y = out['kmd'].to_numpy(float)
    ratio = yp / y
    out.insert(0, 'eval_split', split)
    out.insert(1, 'route', route)
    out.insert(2, 'combo', ' + '.join(combo))
    out['kTM_pred'] = yp
    out['kMD_true'] = y
    out['log_ratio_kMD_over_kTM'] = np.log(y / yp)
    out['abs_log_ratio'] = np.abs(out['log_ratio_kMD_over_kTM'])
    out['relative_error_kTM_minus_kMD_over_kMD'] = (yp - y) / y
    out['within_50pct'] = np.abs(ratio - 1.0) <= 0.5
    out['within_100pct'] = np.abs(ratio - 1.0) <= 1.0
    out['within_200pct'] = np.abs(ratio - 1.0) <= 2.0
    out['within_factor2'] = (ratio >= 0.5) & (ratio <= 2.0)
    return out

def safe_limits(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    vmin, vmax = float(values.min()), float(values.max())
    log_min, log_max = math.log10(vmin), math.log10(vmax)
    span = max(log_max - log_min, 0.5)
    pad = 0.08 * span
    return 10 ** (log_min - pad), 10 ** (log_max + pad)

def save_scatter(train_pred: pd.DataFrame, test_pred: pd.DataFrame, outpath: Path, title: str, train_m: dict, test_m: dict) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tx, ty = train_pred['kTM_pred'].to_numpy(float), train_pred['kMD_true'].to_numpy(float)
    vx, vy = test_pred['kTM_pred'].to_numpy(float), test_pred['kMD_true'].to_numpy(float)
    lower, upper = safe_limits(np.concatenate([tx, ty, vx, vy]))
    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)
    ax.scatter(tx, ty, c='black', s=32, alpha=0.78, label=f'Training set (n={len(train_pred)})')
    ax.scatter(vx, vy, c='red', s=42, alpha=0.88, label=f'Test set (n={len(test_pred)})')
    ax.plot([lower, upper], [lower, upper], '--', color='gray', linewidth=1.2, label='y = x')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Predicted $\\kappa_{TM}$ (W m$^{-1}$ K$^{-1}$)')
    ax.set_ylabel('$\\kappa_{MD}$ (W m$^{-1}$ K$^{-1}$)')
    ax.set_title(title, fontsize=9)
    ax.text(0.04, 0.96, 'Train: MALE={:.3g}, RMSLE={:.3g}\nTest:  MALE={:.3g}, RMSLE={:.3g}'.format(train_m['MALE'], train_m['RMSLE'], test_m['MALE'], test_m['RMSLE']), transform=ax.transAxes, va='top', ha='left', fontsize=8, bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='none'))
    ax.legend(loc='lower right', fontsize=8, frameon=True)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.45)
    fig.tight_layout()
    fig.savefig(outpath, dpi=SCATTER_DPI, bbox_inches='tight')
    plt.close(fig)

def exhaustive_cv_select(train_df: pd.DataFrame, route: str, lp: LossParams, keep_all: bool = True) -> pd.DataFrame:
    combos = list(itertools.combinations(CANDIDATE_F2_DESCRIPTORS, 4))
    strata = make_tercile_labels(train_df['kmd'].to_numpy(float))
    cv_state = CV_RANDOM_STATE + (0 if route == 'route1_fixed_f1' else 777)
    kf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=cv_state)
    fold_data = []
    for fold, (idx_tr, idx_val) in enumerate(kf.split(train_df, strata), start=1):
        dtr = train_df.iloc[idx_tr].reset_index(drop=True)
        dval = train_df.iloc[idx_val].reset_index(drop=True)
        fold_data.append((fold, prepare_arrays(dtr), prepare_arrays(dval)))
    rows = []
    for i, combo in enumerate(combos, start=1):
        vals_L, vals_MALE, vals_RMSLE = [], [], []
        vals_MAE, vals_RMSE = [], []
        vals_50, vals_100, vals_f2 = [], [], []
        trains_L, trains_MALE, trains_RMSLE = [], [], []
        statuses = []
        for fold, tr_data, val_data in fold_data:
            seed = SEED + 100000 + 1000 * fold + i + (0 if route == 'route1_fixed_f1' else 500000)
            theta, _, status, _, _ = fit_lbfgsb_multistart(tr_data, route, combo, lp, seed=seed)
            trm = metric_dict(tr_data, route, combo, theta, lp)
            valm = metric_dict(val_data, route, combo, theta, lp)
            vals_L.append(valm['L'])
            vals_MALE.append(valm['MALE'])
            vals_RMSLE.append(valm['RMSLE'])
            vals_MAE.append(valm['MAE_linear'])
            vals_RMSE.append(valm['RMSE_linear'])
            vals_50.append(valm['frac_within_50pct'])
            vals_100.append(valm['frac_within_100pct'])
            vals_f2.append(valm['frac_within_factor2'])
            trains_L.append(trm['L'])
            trains_MALE.append(trm['MALE'])
            trains_RMSLE.append(trm['RMSLE'])
            statuses.append(status)
        row = {
            'route': route,
            'combo': ' + '.join(combo),
            'combo_json': json.dumps(list(combo), ensure_ascii=False),
            'cv_val_L_mean': float(np.mean(vals_L)),
            'cv_val_L_std': float(np.std(vals_L, ddof=1)),
            'cv_val_MALE_mean': float(np.mean(vals_MALE)),
            'cv_val_MALE_std': float(np.std(vals_MALE, ddof=1)),
            'cv_val_RMSLE_mean': float(np.mean(vals_RMSLE)),
            'cv_val_RMSLE_std': float(np.std(vals_RMSLE, ddof=1)),
            'cv_val_MAE_linear_mean': float(np.mean(vals_MAE)),
            'cv_val_RMSE_linear_mean': float(np.mean(vals_RMSE)),
            'cv_val_within_50pct_mean': float(np.mean(vals_50)),
            'cv_val_within_100pct_mean': float(np.mean(vals_100)),
            'cv_val_within_factor2_mean': float(np.mean(vals_f2)),
            'cv_train_L_mean': float(np.mean(trains_L)),
            'cv_train_MALE_mean': float(np.mean(trains_MALE)),
            'cv_train_RMSLE_mean': float(np.mean(trains_RMSLE)),
            'n_folds': KFOLD,
            'fit_status_summary': ';'.join(sorted(set(statuses))),
            **asdict(lp),
        }
        for j in range(KFOLD):
            row[f'val_L_fold{j + 1}'] = vals_L[j]
            row[f'val_MALE_fold{j + 1}'] = vals_MALE[j]
            row[f'val_RMSLE_fold{j + 1}'] = vals_RMSLE[j]
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(['cv_val_L_mean', 'cv_val_RMSLE_mean', 'cv_val_MALE_mean']).reset_index(drop=True)
    out.insert(0, 'cv_rank_by_L', np.arange(1, len(out) + 1))
    return out if keep_all else out.iloc[[0]].copy()

def save_metric_plot(final_df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    x = np.arange(len(final_df))
    labels = final_df['route'].map({'route1_fixed_f1': 'Route I', 'route2_refit_f1_f2': 'Route II'}).tolist()
    w = 0.2
    ax.bar(x - 1.5 * w, final_df['cv_val_MALE_mean'], w, label='CV MALE')
    ax.bar(x - 0.5 * w, final_df['test_MALE'], w, label='Test MALE')
    ax.bar(x + 0.5 * w, final_df['cv_val_RMSLE_mean'], w, label='CV RMSLE')
    ax.bar(x + 1.5 * w, final_df['test_RMSLE'], w, label='Test RMSLE')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Error metric')
    ax.set_title('CV-selected exhaustive models: CV vs test metrics')
    ax.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.55)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / 'cv_vs_test_MALE_RMSLE_by_route.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def run() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = load_clean_dataframe(INPUT_XLSX)
    train_df, test_df = train_test_split_stratified(df)
    split_assignment = pd.concat([train_df, test_df], ignore_index=True)
    all_combo_summaries = []
    best_combo_rows = []
    final_rows = []
    pred_sheets = {}
    formula_lines = []
    for route in ROUTES:
        lp = ROUTE_LOSS_PARAMS[route]
        print(f'Running exhaustive C(16,4) search for {route}', flush=True)
        summary_df = exhaustive_cv_select(train_df, route, lp, keep_all=True)
        all_combo_summaries.append(summary_df)
        best_row = summary_df.iloc[0]
        combo = tuple(json.loads(best_row['combo_json']))
        best_combo_rows.append(best_row.to_dict())
        train_data = prepare_arrays(train_df)
        test_data = prepare_arrays(test_df)
        seed = SEED + 90000 + (0 if route == 'route1_fixed_f1' else 500)
        theta, obj, status, _, n_success = fit_lbfgsb_multistart(train_data, route, combo, lp, seed=seed)
        train_m = metric_dict(train_data, route, combo, theta, lp)
        test_m = metric_dict(test_data, route, combo, theta, lp)
        formula = theta_to_formula(route, combo, theta)
        pred_train = prediction_dataframe(train_df, route, combo, theta, 'train')
        pred_test = prediction_dataframe(test_df, route, combo, theta, 'test')
        pred_all = pd.concat([pred_train, pred_test], ignore_index=True)
        pred_sheets[route] = pred_all
        plot_path = OUTDIR / 'scatter_plots' / f'{route}_cv_selected_exhaustive_scatter.png'
        save_scatter(pred_train, pred_test, plot_path, f"{route}: {' + '.join(combo)}", train_m, test_m)
        row = {
            'route': route,
            'combo': ' + '.join(combo),
            'combo_json': json.dumps(list(combo), ensure_ascii=False),
            'selection_metric': 'minimum mean 5-fold CV validation L within the 80% training set',
            'cv_rank_by_L': 1,
            'cv_val_L_mean': float(best_row['cv_val_L_mean']),
            'cv_val_L_std': float(best_row['cv_val_L_std']),
            'cv_val_MALE_mean': float(best_row['cv_val_MALE_mean']),
            'cv_val_MALE_std': float(best_row['cv_val_MALE_std']),
            'cv_val_RMSLE_mean': float(best_row['cv_val_RMSLE_mean']),
            'cv_val_RMSLE_std': float(best_row['cv_val_RMSLE_std']),
            'formula': formula,
            'theta_json': json.dumps([float(x) for x in clip_theta(theta)], ensure_ascii=False),
            'prefactor_A': prefactor_A(theta),
            'max_abs_exponent': max_abs_exponent(theta),
            'objective_train_L': obj,
            'fit_status': status,
            'n_success_starts': n_success,
            'train_L': train_m['L'],
            'train_MALE': train_m['MALE'],
            'train_RMSLE': train_m['RMSLE'],
            'train_frac_within_50pct': train_m['frac_within_50pct'],
            'train_frac_within_100pct': train_m['frac_within_100pct'],
            'train_frac_within_factor2': train_m['frac_within_factor2'],
            'test_L': test_m['L'],
            'test_MALE': test_m['MALE'],
            'test_RMSLE': test_m['RMSLE'],
            'test_mean_log_bias_MD_over_pred': test_m['mean_log_bias_MD_over_pred'],
            'test_MAE_linear': test_m['MAE_linear'],
            'test_RMSE_linear': test_m['RMSE_linear'],
            'test_frac_within_50pct': test_m['frac_within_50pct'],
            'test_frac_within_100pct': test_m['frac_within_100pct'],
            'test_frac_within_factor2': test_m['frac_within_factor2'],
            'gap_test_minus_cv_MALE': test_m['MALE'] - float(best_row['cv_val_MALE_mean']),
            'gap_test_minus_cv_RMSLE': test_m['RMSLE'] - float(best_row['cv_val_RMSLE_mean']),
            'n_train_rows': len(train_df),
            'n_test_rows': len(test_df),
            'scatter_plot_png': str(plot_path),
            **asdict(lp),
        }
        final_rows.append(row)
        formula_lines.append(f"Route: {route}\nCV-selected best exhaustive combo: {' + '.join(combo)}\nSelection metric: minimum mean 5-fold CV validation L within the 80% training set\nCV validation: L={float(best_row['cv_val_L_mean']):.8g} ± {float(best_row['cv_val_L_std']):.8g}; MALE={float(best_row['cv_val_MALE_mean']):.8g}; RMSLE={float(best_row['cv_val_RMSLE_mean']):.8g}\nFormula: {formula}\nTrain: L={train_m['L']:.8g}, MALE={train_m['MALE']:.8g}, RMSLE={train_m['RMSLE']:.8g}\nIndependent test: L={test_m['L']:.8g}, MALE={test_m['MALE']:.8g}, RMSLE={test_m['RMSLE']:.8g}\n\n")
        print(f"{route}: {' + '.join(combo)}; test MALE={test_m['MALE']:.6f}; test RMSLE={test_m['RMSLE']:.6f}", flush=True)
    all_combo_cv_summary = pd.concat(all_combo_summaries, ignore_index=True)
    best_combo_df = pd.DataFrame(best_combo_rows)
    final_df = pd.DataFrame(final_rows)
    hp_df = pd.DataFrame([{'route': r, **asdict(lp), 'selection_data': '80% training set, 5-fold CV', 'final_evaluation_data': '20% independent test set'} for r, lp in ROUTE_LOSS_PARAMS.items()])
    split_assignment.to_excel(OUTDIR / 'data_split_80_20.xlsx', index=False)
    all_combo_cv_summary.to_excel(OUTDIR / 'all_4descriptor_combos_cv_summary_fixedHP.xlsx', index=False)
    best_combo_df.to_excel(OUTDIR / 'best_combos_selected_by_cvL_fixedHP.xlsx', index=False)
    final_df.to_excel(OUTDIR / 'best_exhaustive_final_train_test_metrics_fixedHP.xlsx', index=False)
    hp_df.to_excel(OUTDIR / 'selected_loss_hyperparams_by_route_fixedHP.xlsx', index=False)
    final_df[['route', 'combo', 'cv_val_MALE_mean', 'test_MALE', 'gap_test_minus_cv_MALE', 'cv_val_RMSLE_mean', 'test_RMSLE', 'gap_test_minus_cv_RMSLE', 'test_MAE_linear', 'test_RMSE_linear', 'test_frac_within_50pct', 'test_frac_within_100pct', 'test_frac_within_factor2']].to_excel(OUTDIR / 'best_exhaustive_cv_test_metrics_by_route.xlsx', index=False)
    with pd.ExcelWriter(OUTDIR / 'best_exhaustive_predictions_fixedHP.xlsx', engine='openpyxl') as writer:
        for sheet, data in pred_sheets.items():
            data.to_excel(writer, sheet_name=sheet[:31], index=False)
    with pd.ExcelWriter(OUTDIR / 'summary_best_exhaustive_route1_route2.xlsx', engine='openpyxl') as writer:
        pd.DataFrame([{'metric_definition': 'signed_log_error = log(kMD/kTM_pred); MALE = mean(abs(error)); RMSLE = sqrt(mean(error^2))', 'test_size': TEST_SIZE, 'kfold': KFOLD, 'search_space_per_route': math.comb(len(CANDIDATE_F2_DESCRIPTORS), 4), 'selection_rule': 'minimum mean 5-fold CV validation L within the 80% training set'}]).to_excel(writer, sheet_name='README', index=False)
        final_df.to_excel(writer, sheet_name='final_train_test', index=False)
        best_combo_df.to_excel(writer, sheet_name='best_combos_by_cvL', index=False)
        all_combo_cv_summary.to_excel(writer, sheet_name='all_combo_cv_summary', index=False)
        hp_df.to_excel(writer, sheet_name='selected_hyperparams', index=False)
        split_assignment.to_excel(writer, sheet_name='data_split', index=False)
    (OUTDIR / 'best_exhaustive_formulas_fixedHP.txt').write_text(''.join(formula_lines), encoding='utf-8')
    save_metric_plot(final_df, OUTDIR)
    summary = {
        'input': str(INPUT_XLSX),
        'outdir': str(OUTDIR),
        'n_valid_rows': int(len(df)),
        'n_train_rows': int(len(train_df)),
        'n_test_rows': int(len(test_df)),
        'test_size': TEST_SIZE,
        'kfold': KFOLD,
        'candidate_f2_descriptors': CANDIDATE_F2_DESCRIPTORS,
        'search_space_per_route': math.comb(len(CANDIDATE_F2_DESCRIPTORS), 4),
        'selected_combos': dict(zip(final_df['route'], final_df['combo'])),
        'fixed_loss_hyperparams_by_route': {r: asdict(lp) for r, lp in ROUTE_LOSS_PARAMS.items()},
        'metric_definition': 'signed_log_error = log(kMD/kTM_pred)',
        'final_results': final_df.to_dict(orient='records'),
    }
    (OUTDIR / 'summary_best_exhaustive_route1_route2.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copy2(Path(__file__).resolve(), OUTDIR / Path(__file__).name)
    if ZIPFILE.exists():
        ZIPFILE.unlink()
    with zipfile.ZipFile(ZIPFILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in OUTDIR.rglob('*'):
            if f.is_file():
                zf.write(f, f.relative_to(OUTDIR.parent))
    print(final_df[['route', 'combo', 'cv_val_MALE_mean', 'test_MALE', 'cv_val_RMSLE_mean', 'test_RMSLE']].to_string(index=False), flush=True)
    print(str(ZIPFILE), flush=True)

if __name__ == '__main__':
    run()
