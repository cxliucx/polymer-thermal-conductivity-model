from __future__ import annotations
import argparse
import heapq
import math
import os
import re
from collections import deque
from functools import reduce
from math import gcd
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd
from pymatgen.core import Element, Structure
FOLDER_AXIS_CANDIDATES: Sequence[Tuple[str, int, str]] = (('cifx', 0, 'x'), ('cify', 1, 'y'), ('cifz', 2, 'z'), ('cifs_x', 0, 'x'), ('cifs_y', 1, 'y'), ('cifs_z', 2, 'z'))
OUT_XLSX = 'cifxyz_all_descriptors.xlsx'
BASE_XLSX = 'base_5.xlsx'
T_DEFAULT = 300.0
PACKING_RADIUS_MODE = 'covalent'
BOND_RADIUS_MODE = 'covalent'
COVALENT_RADII_A: Dict[str, float] = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'S': 1.05}
ATOMIC_RADII_A: Dict[str, float] = {'H': 0.53, 'C': 0.67, 'N': 0.56, 'O': 0.48, 'F': 0.42, 'S': 0.88}
VDW_RADII_A: Dict[str, float] = {'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'S': 1.8}
RADIUS_TABLES: Dict[str, Dict[str, float]] = {'covalent': COVALENT_RADII_A, 'atomic': ATOMIC_RADII_A, 'vdw': VDW_RADII_A}
ALLOWED_ELEMENTS = {'C', 'H', 'O', 'N', 'S', 'F'}
BOND_TOL = 0.45
BOND_TOL_GRID = [0.45, 0.55, 0.65, 0.75]
CONTACT_CUTOFF = 3.5
PROJ_FRAC = 0.75
PROJ_FRAC_GRID = [0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
OFFSET_FACTOR_GRID = [3, 4, 5, 6]
MAX_PERIOD_MULT = 8
RHOIN_MODE = 'all'
RHOIN_MIN_CHAIN_PROJ_FRAC = 0.5

def natural_key(s: str) -> List[Any]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split('(\\d+)', s)]

def parse_id_from_filename(fname: str) -> Optional[int]:
    m = re.search('(\\d+)', os.path.basename(fname))
    return int(m.group(1)) if m else None

def collect_cifs(root_dir: str) -> List[Tuple[Optional[int], str, int, str, str]]:
    rows: List[Tuple[Optional[int], str, int, str, str]] = []
    seen_folder_paths: Set[str] = set()
    for folder_name, axis, axis_label in FOLDER_AXIS_CANDIDATES:
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        real_path = os.path.realpath(folder_path)
        if real_path in seen_folder_paths:
            continue
        seen_folder_paths.add(real_path)
        files = [fn for fn in os.listdir(folder_path) if fn.lower().endswith('.cif')]
        files = sorted(files, key=lambda x: natural_key(os.path.splitext(x)[0]))
        for fn in files:
            rows.append((parse_id_from_filename(fn), os.path.join(folder_path, fn), axis, axis_label, folder_name))
    return rows

def clean_element_symbol(raw: Any) -> str:
    s = str(raw).strip()
    if not s:
        return 'C'
    if s in ALLOWED_ELEMENTS:
        return s
    for el in sorted(ALLOWED_ELEMENTS, key=len, reverse=True):
        if re.match(f'^{el}(?:[0-9_+\\-\\.].*)?$', s):
            return el
    try:
        return Element(s).symbol
    except Exception:
        pass
    m = re.match('^([A-Z][a-z]?)', s)
    if m:
        sym = m.group(1)
        if sym in ALLOWED_ELEMENTS:
            return sym
    return s

def site_element_symbol(site) -> str:
    try:
        return clean_element_symbol(site.specie.symbol)
    except Exception:
        pass
    try:
        d = site.species.get_el_amt_dict()
        if d:
            raw = max(d.items(), key=lambda kv: kv[1])[0]
            return clean_element_symbol(raw)
    except Exception:
        pass
    try:
        return clean_element_symbol(site.species_string)
    except Exception:
        return 'C'

def radius_for_element(sym: str, mode: str='covalent') -> float:
    sym = clean_element_symbol(sym)
    mode = mode.lower()
    table = RADIUS_TABLES.get(mode)
    if table is None:
        raise ValueError(f'Unknown radius mode: {mode}. Choose from {list(RADIUS_TABLES)}')
    if sym in table:
        return float(table[sym])
    try:
        el = Element(sym)
        if mode == 'covalent':
            r = el.covalent_radius
        elif mode == 'atomic':
            r = el.atomic_radius or el.atomic_radius_calculated
        elif mode == 'vdw':
            r = el.van_der_waals_radius
        else:
            r = None
        if r is not None:
            return float(r)
    except Exception:
        pass
    return COVALENT_RADII_A['C']

def composition_amounts(struct: Structure) -> Dict[str, float]:
    out = {el: 0.0 for el in ALLOWED_ELEMENTS}
    try:
        d = struct.composition.get_el_amt_dict()
        for raw, amt in d.items():
            sym = clean_element_symbol(raw)
            out[sym] = out.get(sym, 0.0) + float(amt)
        return out
    except Exception:
        pass
    for i in range(len(struct)):
        sym = site_element_symbol(struct[i])
        out[sym] = out.get(sym, 0.0) + 1.0
    return out

def total_atoms_from_composition(struct: Structure) -> float:
    try:
        return float(struct.composition.num_atoms)
    except Exception:
        return float(len(struct))

def structure_mass_amu(struct: Structure) -> float:
    try:
        return float(struct.composition.weight)
    except Exception:
        mass = 0.0
        for i in range(len(struct)):
            sym = site_element_symbol(struct[i])
            try:
                mass += float(Element(sym).atomic_mass)
            except Exception:
                pass
        return mass

def get_neighbor_list_robust(struct: Structure, r: float):
    try:
        return struct.get_neighbor_list(r)
    except Exception:
        centers, neighs, offsets, dists = ([], [], [], [])
        all_neighbors = struct.get_all_neighbors(r, include_index=True)
        for i, nnlist in enumerate(all_neighbors):
            for nb in nnlist:
                centers.append(i)
                neighs.append(nb.index)
                offsets.append(nb.image)
                dists.append(nb.nn_distance)
        return (np.array(centers, dtype=int), np.array(neighs, dtype=int), np.array(offsets, dtype=int), np.array(dists, dtype=float))

def tuple_offset(off: Any) -> Tuple[int, int, int]:
    a = np.array(off, dtype=float).round().astype(int).tolist()
    return (int(a[0]), int(a[1]), int(a[2]))

def canonical_edge_key(i: int, j: int, off: Tuple[int, int, int]) -> Tuple[int, int, int, int, int]:
    a = (int(i), int(j), int(off[0]), int(off[1]), int(off[2]))
    b = (int(j), int(i), int(-off[0]), int(-off[1]), int(-off[2]))
    return a if a <= b else b

def build_bond_graph(struct: Structure, bond_tol: float=BOND_TOL, radius_mode: str=BOND_RADIUS_MODE) -> Tuple[List[List[Tuple[int, Tuple[int, int, int], float, np.ndarray]]], Set[Tuple[int, int, int, int, int]], List[str]]:
    n = len(struct)
    elem_sym = [site_element_symbol(struct[i]) for i in range(n)]
    if n == 0:
        return ([], set(), elem_sym)
    radii = np.array([radius_for_element(s, radius_mode) for s in elem_sym], dtype=float)
    max_cut = float(2.0 * np.max(radii) + bond_tol + 0.25)
    centers, neighs, offsets, dists = get_neighbor_list_robust(struct, max_cut)
    lat_mat = np.array(struct.lattice.matrix, dtype=float)
    coords = np.array([struct[i].coords for i in range(n)], dtype=float)
    adj: List[List[Tuple[int, Tuple[int, int, int], float, np.ndarray]]] = [[] for _ in range(n)]
    bond_keys: Set[Tuple[int, int, int, int, int]] = set()
    for i0, j0, off0, dist0 in zip(centers, neighs, offsets, dists):
        i = int(i0)
        j = int(j0)
        off = tuple_offset(off0)
        dist = float(dist0)
        if i == j and off == (0, 0, 0):
            continue
        cutoff = float(radii[i] + radii[j] + bond_tol)
        if dist > cutoff:
            continue
        vec = coords[j] + np.dot(np.array(off, dtype=float), lat_mat) - coords[i]
        adj[i].append((j, off, dist, vec))
        bond_keys.add(canonical_edge_key(i, j, off))
    return (adj, bond_keys, elem_sym)

def count_cc_bonds(bond_keys: Set[Tuple[int, int, int, int, int]], elem_sym: Sequence[str]) -> int:
    n_cc = 0
    for key in bond_keys:
        i, j = (key[0], key[1])
        if i < len(elem_sym) and j < len(elem_sym) and (elem_sym[i] == 'C') and (elem_sym[j] == 'C'):
            n_cc += 1
    return int(n_cc)

def packing_fraction(struct: Structure, radius_mode: str=PACKING_RADIUS_MODE) -> float:
    V = float(struct.lattice.volume)
    if V <= 0:
        return float('nan')
    counts = composition_amounts(struct)
    atom_vol_sum = 0.0
    for sym, amount in counts.items():
        if amount <= 0:
            continue
        r = radius_for_element(sym, radius_mode)
        atom_vol_sum += float(amount) * (4.0 / 3.0) * math.pi * r ** 3
    return float(atom_vol_sum / V)

def compute_rhoin(struct: Structure, bond_keys: Set[Tuple[int, int, int, int, int]], axis: int, cutoff: float=CONTACT_CUTOFF, mode: str=RHOIN_MODE) -> Tuple[float, int]:
    _BOND_TOL = 0.55
    _CONTACT_CUTOFF = 3.5
    _PROJ_FRAC = 0.75
    _OFFSET_FACTOR = 3
    _BOND_TOL_GRID = [0.55, 0.65, 0.75, 0.85, 0.95]
    _PROJ_FRAC_GRID = [0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]
    _OFFSET_FACTOR_GRID = [3, 4, 5, 6]
    _CONTACT_CUTOFF_GRID = [3.5, 3.8, 4.0]
    _MAX_PERIOD_MULT = 8

    def _site_primary_element_symbol(site) -> str:
        try:
            return site.specie.symbol
        except Exception:
            d = site.species.get_el_amt_dict()
            if not d:
                return 'C'
            return max(d.items(), key=lambda kv: kv[1])[0]

    def _covalent_radius_symbol(sym: str) -> float:
        try:
            r = Element(sym).covalent_radius
            if r is None:
                return 0.77
            return float(r)
        except Exception:
            return 0.77

    def _build_bond_graph(structure: Structure, bond_tol: float):
        n = len(structure)
        if n == 0:
            return ([[]], set(), [])
        elem_sym = [_site_primary_element_symbol(structure[i]) for i in range(n)]
        rcov = np.array([_covalent_radius_symbol(s) for s in elem_sym], dtype=float)
        max_cut = float(2.0 * np.max(rcov) + bond_tol + 0.3)
        centers, neighs, offsets, dists = get_neighbor_list_robust(structure, max_cut)
        lat_mat = np.array(structure.lattice.matrix, dtype=float)
        coords = np.array([structure[i].coords for i in range(n)], dtype=float)
        adj = [[] for _ in range(n)]
        bonded_pairs = set()
        for i, j, off, dist in zip(centers, neighs, offsets, dists):
            i = int(i)
            j = int(j)
            off = tuple((int(x) for x in np.array(off).tolist()))
            if i == j and off == (0, 0, 0):
                continue
            cutoff_ij = float(rcov[i] + rcov[j] + bond_tol)
            if float(dist) > cutoff_ij:
                continue
            shift = np.dot(np.array(off, dtype=float), lat_mat)
            vec = coords[j] + shift - coords[i]
            adj[i].append((j, off, float(dist), vec))
            a, b = (i, j) if i < j else (j, i)
            bonded_pairs.add((a, b))
        return (adj, bonded_pairs, elem_sym)

    def _compute_interchain_contact_density(structure: Structure, bonded_pairs: set, contact_cutoff: float) -> Tuple[float, int]:
        n = len(structure)
        if n == 0:
            return (float('nan'), 0)
        centers, neighs, offsets, dists = get_neighbor_list_robust(structure, contact_cutoff)
        contacts = set()
        for i, j, off, dist in zip(centers, neighs, offsets, dists):
            i = int(i)
            j = int(j)
            off = tuple((int(x) for x in np.array(off).tolist()))
            if i == j and off == (0, 0, 0):
                continue
            if float(dist) >= contact_cutoff:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in bonded_pairs:
                continue
            if i < j:
                key = (i, j, off)
            else:
                inv = (-off[0], -off[1], -off[2])
                key = (j, i, inv)
            contacts.add(key)
        V = float(structure.lattice.volume)
        return (float(len(contacts) / max(V, 1e-12)), int(len(contacts)))

    def _dihedral(p0, p1, p2, p3) -> float:
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        b1 /= np.linalg.norm(b1) + 1e-12
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return math.atan2(y, x)

    def _dijkstra_periodic_backbone(structure: Structure, adj, axis_index: int, proj_frac: float, offset_factor: int, max_mult: int=_MAX_PERIOD_MULT) -> Optional[Tuple[List[int], List[np.ndarray]]]:
        n = len(structure)
        if n == 0:
            return None
        chain_vec = np.array(structure.lattice.matrix[axis_index], dtype=float)
        chain_unit = chain_vec / (np.linalg.norm(chain_vec) + 1e-12)
        cand_dt: List[int] = []
        seed_score = np.zeros(n, dtype=int)
        for i in range(n):
            for j, img, w, vec in adj[i]:
                dt = int(img[axis_index])
                if dt == 0:
                    continue
                v = np.array(vec, dtype=float)
                vnorm = np.linalg.norm(v) + 1e-12
                proj = float(np.dot(v, chain_unit))
                if abs(proj) / vnorm < proj_frac:
                    continue
                cand_dt.append(abs(dt))
                seed_score[i] += 1
                seed_score[j] += 1
        if not cand_dt:
            return None
        period = int(min(cand_dt))
        elems = [_site_primary_element_symbol(structure[k]) for k in range(n)]
        score2 = seed_score.copy()
        for k, e in enumerate(elems):
            if e == 'H':
                score2[k] = max(score2[k] - 1, 0)
        seed = int(np.argmax(score2)) if np.max(score2) > 0 else int(np.argmax(seed_score))

        def run_dijkstra(target_off: int):
            off_min = -offset_factor * target_off
            off_max = offset_factor * target_off
            n_off = off_max - off_min + 1

            def sid(atom: int, off: int) -> int:
                return (off - off_min) * n + atom
            start_id = sid(seed, 0)
            target_id = sid(seed, target_off)
            dist_arr = [math.inf] * (n * n_off)
            prev = [None] * (n * n_off)
            dist_arr[start_id] = 0.0
            pq = [(0.0, seed, 0)]
            while pq:
                dcur, u, off = heapq.heappop(pq)
                if dcur != dist_arr[sid(u, off)]:
                    continue
                if u == seed and off == target_off:
                    break
                for v, img, w, vec in adj[u]:
                    dt = int(img[axis_index])
                    vv = np.array(vec, dtype=float)
                    vnorm = np.linalg.norm(vv) + 1e-12
                    proj = float(np.dot(vv, chain_unit))
                    if abs(proj) / vnorm < proj_frac:
                        continue
                    if dt != 0 and abs(dt) % period != 0:
                        continue
                    noff = off + dt
                    if noff < off_min or noff > off_max:
                        continue
                    nid = sid(v, noff)
                    nd = dcur + float(w)
                    if nd < dist_arr[nid]:
                        dist_arr[nid] = nd
                        prev[nid] = (u, off, v, noff, vv)
                        heapq.heappush(pq, (nd, v, noff))
            if not math.isfinite(dist_arr[target_id]):
                return None
            atom_order: List[int] = []
            step_vecs: List[np.ndarray] = []
            cur = target_id
            while cur != start_id:
                p = prev[cur]
                if p is None:
                    break
                u, off_u, v, off_v, vv = p
                atom_order.append(v)
                step_vecs.append(vv)
                cur = sid(u, off_u)
            atom_order.append(seed)
            atom_order.reverse()
            step_vecs.reverse()
            if len(atom_order) < 2:
                return None
            return (atom_order, step_vecs)
        best = None
        for mult in range(1, max_mult + 1):
            res = run_dijkstra(mult * period)
            if res is None:
                continue
            best = res
            if len(res[0]) >= 4:
                return res
        return best

    def _longest_path_fallback(structure: Structure, adj) -> Optional[List[int]]:
        n = len(structure)
        if n == 0:
            return None
        elems = [_site_primary_element_symbol(structure[i]) for i in range(n)]
        nonH = [i for i, e in enumerate(elems) if e != 'H']
        prefer_nodes = set(nonH) if len(nonH) >= 2 else set(range(n))
        g = [[] for _ in range(n)]
        for u in range(n):
            if u not in prefer_nodes:
                continue
            for v, img, w, vec in adj[u]:
                if img != (0, 0, 0):
                    continue
                if v not in prefer_nodes:
                    continue
                g[u].append(v)
        deg = [len(g[i]) for i in range(n)]
        if max(deg) == 0:
            return None
        start = int(np.argmax(deg))

        def bfs(src: int):
            dist = [-1] * n
            prev = [-1] * n
            q = deque([src])
            dist[src] = 0
            while q:
                u = q.popleft()
                for v in g[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        q.append(v)
            far = int(np.argmax(dist))
            return (far, prev, dist)
        far1, _, _ = bfs(start)
        far2, prev, dist = bfs(far1)
        if dist[far2] <= 0:
            return None
        path = []
        cur = far2
        while cur != -1:
            path.append(cur)
            if cur == far1:
                break
            cur = prev[cur]
        path.reverse()
        return path

    def _compute_backbone_side_descriptors(structure: Structure, adj, axis_index: int, proj_frac: float, offset_factor: int) -> Tuple[Dict[str, float], str]:
        out = {'backbone_bondlen_mean': np.nan, 'backbone_bondlen_std': np.nan, 'backbone_bondangle_mean': np.nan, 'backbone_bondangle_std': np.nan, 'backbone_dihedral_order': np.nan, 'side_atoms_per_backbone': np.nan, 'sidechain_extent_mean': np.nan}
        n = len(structure)
        if n < 2:
            return (out, 'OK_NO_BACKBONE')
        periodic = _dijkstra_periodic_backbone(structure, adj, axis_index, proj_frac, offset_factor)
        if periodic is not None:
            atom_order, step_vecs = periodic
            status = 'OK'
        else:
            atom_order = _longest_path_fallback(structure, adj)
            if atom_order is None or len(atom_order) < 2:
                return (out, 'OK_NO_BACKBONE')
            coords = np.array([structure[i].coords for i in range(n)], dtype=float)
            step_vecs = [coords[atom_order[i + 1]] - coords[atom_order[i]] for i in range(len(atom_order) - 1)]
            status = 'OK_NO_BACKBONE'
        step_vecs = np.array(step_vecs, dtype=float)
        bl = np.linalg.norm(step_vecs, axis=1)
        if len(bl) > 0:
            out['backbone_bondlen_mean'] = float(np.mean(bl))
            out['backbone_bondlen_std'] = float(np.std(bl))
        angles = []
        for i in range(len(step_vecs) - 1):
            v1 = step_vecs[i]
            v2 = step_vecs[i + 1]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-08 or n2 < 1e-08:
                continue
            cosang = float(np.dot(v1, v2) / (n1 * n2))
            cosang = max(-1.0, min(1.0, cosang))
            angles.append(math.degrees(math.acos(cosang)))
        if angles:
            out['backbone_bondangle_mean'] = float(np.mean(angles))
            out['backbone_bondangle_std'] = float(np.std(angles))
        coords0 = np.array(structure[atom_order[0]].coords, dtype=float)
        unwrapped = [coords0.copy()]
        for v in step_vecs:
            unwrapped.append(unwrapped[-1] + v)
        unwrapped = np.array(unwrapped, dtype=float)
        dihs = []
        for i in range(len(unwrapped) - 3):
            phi = _dihedral(unwrapped[i], unwrapped[i + 1], unwrapped[i + 2], unwrapped[i + 3])
            dihs.append(abs(math.cos(phi)))
        if dihs:
            out['backbone_dihedral_order'] = float(np.mean(dihs))
        backbone_set = set(atom_order)
        Nb = max(len(backbone_set), 1)
        adj0 = [[] for _ in range(n)]
        for u in range(n):
            for v, img, w, vec in adj[u]:
                if img == (0, 0, 0):
                    adj0[u].append(v)
        side_set = set()
        stack = []
        for b in backbone_set:
            for nb in adj0[b]:
                if nb not in backbone_set:
                    stack.append(nb)
        while stack:
            u = stack.pop()
            if u in backbone_set or u in side_set:
                continue
            side_set.add(u)
            for v in adj0[u]:
                if v not in backbone_set and v not in side_set:
                    stack.append(v)
        out['side_atoms_per_backbone'] = float(len(side_set) / Nb)
        if side_set:
            dists = []
            bb_list = list(backbone_set)
            for s in side_set:
                best = None
                for b in bb_list:
                    try:
                        ds = float(structure.get_distance(s, b))
                    except Exception:
                        ds, _img = structure.get_distance_and_image(s, b)
                        ds = float(ds)
                    if best is None or ds < best:
                        best = ds
                if best is not None:
                    dists.append(best)
            if dists:
                out['sidechain_extent_mean'] = float(np.mean(dists))
        return (out, status)

    def _auto_tune_backbone_params(structure: Structure, axis_index: int):
        best = None
        best_tuple = None
        for bt in _BOND_TOL_GRID:
            adj, bonded_pairs, elem_sym = _build_bond_graph(structure, bt)
            if len(adj) == 0 or len(structure) == 0:
                continue
            for pf in _PROJ_FRAC_GRID:
                for of in _OFFSET_FACTOR_GRID:
                    periodic = _dijkstra_periodic_backbone(structure, adj, axis_index, pf, of)
                    if periodic is not None:
                        atom_order, _ = periodic
                        if len(atom_order) >= 4:
                            bb_desc, bb_status = _compute_backbone_side_descriptors(structure, adj, axis_index, pf, of)
                            return (adj, bonded_pairs, elem_sym, bb_desc, bb_status, (bt, pf, of))
                        else:
                            score = 1000 + len(atom_order)
                            if best is None or score > best:
                                bb_desc, bb_status = _compute_backbone_side_descriptors(structure, adj, axis_index, pf, of)
                                best = score
                                best_tuple = (adj, bonded_pairs, elem_sym, bb_desc, bb_status, (bt, pf, of))
                    else:
                        bb_desc, bb_status = _compute_backbone_side_descriptors(structure, adj, axis_index, pf, of)
                        filled = sum([0 if isinstance(v, float) and math.isnan(v) else 1 for v in bb_desc.values()])
                        score = filled
                        if best is None or score > best:
                            best = score
                            best_tuple = (adj, bonded_pairs, elem_sym, bb_desc, bb_status, (bt, pf, of))
        if best_tuple is None:
            adj, bonded_pairs, elem_sym = _build_bond_graph(structure, _BOND_TOL)
            bb_desc, bb_status = _compute_backbone_side_descriptors(structure, adj, axis_index, _PROJ_FRAC, _OFFSET_FACTOR)
            return (adj, bonded_pairs, elem_sym, bb_desc, bb_status, (_BOND_TOL, _PROJ_FRAC, _OFFSET_FACTOR))
        return best_tuple
    if len(struct) == 0:
        return (float('nan'), 0)
    _adj, bonded_pairs, _elem_sym, _bb_desc, _bb_status, _tuned = _auto_tune_backbone_params(struct, axis)
    contact_best = None
    contact_nin = 0
    contact_used = _CONTACT_CUTOFF
    for cc in _CONTACT_CUTOFF_GRID:
        try:
            val, nin = _compute_interchain_contact_density(struct, bonded_pairs, cc)
        except Exception:
            continue
        if contact_best is None:
            contact_best = val
            contact_nin = nin
            contact_used = cc
        elif contact_best == 0.0 and val > 0.0 or (abs(val - contact_best) < 1e-12 and cc < contact_used):
            contact_best = val
            contact_nin = nin
            contact_used = cc
    if contact_best is None:
        return (float('nan'), 0)
    return (float(contact_best), int(contact_nin))

def dihedral_angle_rad(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / (np.linalg.norm(b1) + 1e-12)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    return math.atan2(y, x)

def min_image_vector(struct: Structure, i: int, j: int) -> np.ndarray:
    try:
        _dist, img = struct.get_distance_and_image(i, j)
        off = tuple_offset(img)
    except Exception:
        off = (0, 0, 0)
    lat_mat = np.array(struct.lattice.matrix, dtype=float)
    return np.array(struct[j].coords, dtype=float) + np.dot(np.array(off, dtype=float), lat_mat) - np.array(struct[i].coords, dtype=float)

def periodic_backbone_path(struct: Structure, adj: List[List[Tuple[int, Tuple[int, int, int], float, np.ndarray]]], elem_sym: Sequence[str], axis: int, proj_frac: float, offset_factor: int, max_mult: int=MAX_PERIOD_MULT) -> Optional[Tuple[List[int], List[np.ndarray]]]:
    n = len(struct)
    if n == 0:
        return None
    heavy = [e != 'H' for e in elem_sym]
    if sum(heavy) < 2:
        return None
    chain_vec = np.array(struct.lattice.matrix[axis], dtype=float)
    chain_unit = chain_vec / (np.linalg.norm(chain_vec) + 1e-12)
    cand_dt: List[int] = []
    seed_score = np.zeros(n, dtype=int)
    for i in range(n):
        if not heavy[i]:
            continue
        for j, img, _w, vec in adj[i]:
            if not heavy[j]:
                continue
            dt = int(img[axis])
            if dt == 0:
                continue
            vnorm = float(np.linalg.norm(vec)) + 1e-12
            pf = abs(float(np.dot(vec, chain_unit))) / vnorm
            if pf < proj_frac:
                continue
            cand_dt.append(abs(dt))
            seed_score[i] += 1
            seed_score[j] += 1
    if not cand_dt:
        return None
    period = reduce(gcd, cand_dt)
    period = max(int(period), 1)
    seed_candidates = [i for i in range(n) if heavy[i]]
    seed = max(seed_candidates, key=lambda k: (seed_score[k], len(adj[k])))

    def sid(atom: int, off_axis: int, off_min: int, n_sites: int) -> int:
        return (off_axis - off_min) * n_sites + atom

    def run_dijkstra(target_axis_off: int) -> Optional[Tuple[List[int], List[np.ndarray]]]:
        off_min = -offset_factor * abs(target_axis_off)
        off_max = offset_factor * abs(target_axis_off)
        n_off = off_max - off_min + 1
        start_id = sid(seed, 0, off_min, n)
        target_id = sid(seed, target_axis_off, off_min, n)
        dist_arr = [math.inf] * (n * n_off)
        prev: List[Optional[Tuple[int, int, int, int, np.ndarray]]] = [None] * (n * n_off)
        dist_arr[start_id] = 0.0
        pq: List[Tuple[float, int, int]] = [(0.0, seed, 0)]
        while pq:
            dcur, u, off_axis = heapq.heappop(pq)
            cur_id = sid(u, off_axis, off_min, n)
            if dcur != dist_arr[cur_id]:
                continue
            if u == seed and off_axis == target_axis_off:
                break
            for v, img, w, vec in adj[u]:
                if not heavy[v]:
                    continue
                dt = int(img[axis])
                if dt != 0 and abs(dt) % period != 0:
                    continue
                vnorm = float(np.linalg.norm(vec)) + 1e-12
                pf = abs(float(np.dot(vec, chain_unit))) / vnorm
                if pf < proj_frac:
                    continue
                new_off_axis = off_axis + dt
                if new_off_axis < off_min or new_off_axis > off_max:
                    continue
                nid = sid(v, new_off_axis, off_min, n)
                nd = dcur + float(w)
                if nd < dist_arr[nid]:
                    dist_arr[nid] = nd
                    prev[nid] = (u, off_axis, v, new_off_axis, np.array(vec, dtype=float))
                    heapq.heappush(pq, (nd, v, new_off_axis))
        if not math.isfinite(dist_arr[target_id]):
            return None
        atom_order: List[int] = []
        step_vecs: List[np.ndarray] = []
        cur = target_id
        while cur != start_id:
            p = prev[cur]
            if p is None:
                break
            u, off_u, v, off_v, vec = p
            atom_order.append(v)
            step_vecs.append(vec)
            cur = sid(u, off_u, off_min, n)
        atom_order.append(seed)
        atom_order.reverse()
        step_vecs.reverse()
        if len(atom_order) < 2:
            return None
        return (atom_order, step_vecs)
    best: Optional[Tuple[List[int], List[np.ndarray]]] = None
    for mult in range(1, max_mult + 1):
        res = run_dijkstra(mult * period)
        if res is None:
            continue
        best = res
        if len(res[0]) >= 4:
            return res
    return best

def longest_heavy_path_fallback(struct: Structure, adj: List[List[Tuple[int, Tuple[int, int, int], float, np.ndarray]]], elem_sym: Sequence[str]) -> Optional[Tuple[List[int], List[np.ndarray]]]:
    n = len(struct)
    heavy_nodes = [i for i, e in enumerate(elem_sym) if e != 'H']
    if len(heavy_nodes) < 2:
        return None
    heavy_set = set(heavy_nodes)
    g: List[List[int]] = [[] for _ in range(n)]
    for u in heavy_nodes:
        for v, img, _w, _vec in adj[u]:
            if v in heavy_set and img == (0, 0, 0):
                g[u].append(v)
    if max((len(g[i]) for i in heavy_nodes), default=0) == 0:
        return None
    start = max(heavy_nodes, key=lambda k: len(g[k]))

    def bfs(src: int) -> Tuple[int, List[int], List[int]]:
        dist = [-1] * n
        prev = [-1] * n
        q: deque[int] = deque([src])
        dist[src] = 0
        while q:
            u = q.popleft()
            for v in g[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    q.append(v)
        reachable = [i for i in heavy_nodes if dist[i] >= 0]
        far = max(reachable, key=lambda k: dist[k])
        return (far, prev, dist)
    far1, _prev1, _dist1 = bfs(start)
    far2, prev2, dist2 = bfs(far1)
    if dist2[far2] <= 0:
        return None
    path = []
    cur = far2
    while cur != -1:
        path.append(cur)
        if cur == far1:
            break
        cur = prev2[cur]
    path.reverse()
    step_vecs = [min_image_vector(struct, path[k], path[k + 1]) for k in range(len(path) - 1)]
    return (path, step_vecs)

def compute_backbone_side_descriptors(struct: Structure, adj: List[List[Tuple[int, Tuple[int, int, int], float, np.ndarray]]], elem_sym: Sequence[str], axis: int, proj_frac: float, offset_factor: int) -> Tuple[Dict[str, float], str, int, int]:
    out = {'Lb': np.nan, 'Lbq': np.nan, 'thetab': np.nan, 'thetabq': np.nan, 'Psi': np.nan, 'chi': np.nan, 'Lstb': np.nan}
    if len(struct) < 2:
        return (out, 'NO_BACKBONE', 0, 0)
    periodic = periodic_backbone_path(struct, adj, elem_sym, axis, proj_frac, offset_factor)
    if periodic is not None:
        atom_order, step_vecs = periodic
        status = 'OK_PERIODIC_BACKBONE'
    else:
        fallback = longest_heavy_path_fallback(struct, adj, elem_sym)
        if fallback is None:
            return (out, 'NO_BACKBONE', 0, 0)
        atom_order, step_vecs = fallback
        status = 'FALLBACK_LONGEST_HEAVY_PATH'
    step_arr = np.array(step_vecs, dtype=float)
    if step_arr.ndim != 2 or len(step_arr) == 0:
        return (out, 'NO_BACKBONE', 0, 0)
    bond_lengths = np.linalg.norm(step_arr, axis=1)
    if len(bond_lengths) > 0:
        out['Lb'] = float(np.mean(bond_lengths))
        out['Lbq'] = float(np.std(bond_lengths))
    angles: List[float] = []
    for k in range(len(step_arr) - 1):
        v1 = step_arr[k]
        v2 = step_arr[k + 1]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        cosang = float(np.dot(v1, v2) / (n1 * n2))
        cosang = max(-1.0, min(1.0, cosang))
        angles.append(math.degrees(math.acos(cosang)))
    if angles:
        out['thetab'] = float(np.mean(angles))
        out['thetabq'] = float(np.std(angles))
    p0 = np.array(struct[atom_order[0]].coords, dtype=float)
    unwrapped = [p0]
    for vec in step_arr:
        unwrapped.append(unwrapped[-1] + np.array(vec, dtype=float))
    unwrapped_arr = np.array(unwrapped, dtype=float)
    dihedral_order: List[float] = []
    for k in range(len(unwrapped_arr) - 3):
        phi = dihedral_angle_rad(unwrapped_arr[k], unwrapped_arr[k + 1], unwrapped_arr[k + 2], unwrapped_arr[k + 3])
        dihedral_order.append(abs(math.cos(phi)))
    if dihedral_order:
        out['Psi'] = float(np.mean(dihedral_order))
    backbone_set: Set[int] = set(atom_order)
    nb_backbone = max(len(backbone_set), 1)
    adj0: List[List[int]] = [[] for _ in range(len(struct))]
    for u in range(len(struct)):
        for v, img, _w, _vec in adj[u]:
            if img == (0, 0, 0):
                adj0[u].append(v)
    side_set: Set[int] = set()
    stack: List[int] = []
    for b in backbone_set:
        for v in adj0[b]:
            if v not in backbone_set:
                stack.append(v)
    while stack:
        u = stack.pop()
        if u in backbone_set or u in side_set:
            continue
        side_set.add(u)
        for v in adj0[u]:
            if v not in backbone_set and v not in side_set:
                stack.append(v)
    out['chi'] = float(len(side_set) / nb_backbone)
    if side_set:
        dists: List[float] = []
        for s in side_set:
            nearest = min((float(struct.get_distance(s, b)) for b in backbone_set))
            dists.append(nearest)
        if dists:
            out['Lstb'] = float(np.mean(dists))
    return (out, status, int(len(backbone_set)), int(len(side_set)))

def auto_tune_backbone(struct: Structure, axis: int) -> Tuple[List[List[Tuple[int, Tuple[int, int, int], float, np.ndarray]]], Set[Tuple[int, int, int, int, int]], List[str], Dict[str, float], str, float, float, int, int, int]:
    best: Optional[Tuple[Any, ...]] = None
    best_score = -1
    for bt in BOND_TOL_GRID:
        adj, bond_keys, elem_sym = build_bond_graph(struct, bond_tol=bt)
        for pf in PROJ_FRAC_GRID:
            for of in OFFSET_FACTOR_GRID:
                desc, status, nb, ns = compute_backbone_side_descriptors(struct, adj, elem_sym, axis, pf, of)
                filled = sum((0 if isinstance(v, float) and math.isnan(v) else 1 for v in desc.values()))
                periodic_bonus = 100 if status == 'OK_PERIODIC_BACKBONE' else 0
                strict_bonus = int(100 * pf)
                tol_penalty = int(20 * (bt - BOND_TOL))
                score = periodic_bonus + filled + strict_bonus - tol_penalty
                candidate = (adj, bond_keys, elem_sym, desc, status, bt, pf, of, nb, ns)
                if score > best_score:
                    best_score = score
                    best = candidate
                if status == 'OK_PERIODIC_BACKBONE' and filled >= 7:
                    return candidate
    if best is not None:
        return best
    adj, bond_keys, elem_sym = build_bond_graph(struct, bond_tol=BOND_TOL)
    desc, status, nb, ns = compute_backbone_side_descriptors(struct, adj, elem_sym, axis, PROJ_FRAC, OFFSET_FACTOR_GRID[0])
    return (adj, bond_keys, elem_sym, desc, status, BOND_TOL, PROJ_FRAC, OFFSET_FACTOR_GRID[0], nb, ns)

def compute_descriptors_for_structure(struct: Structure, axis: int) -> Tuple[Dict[str, Any], str]:
    row: Dict[str, Any] = {}
    status_parts: List[str] = []
    elem_sym_all = [site_element_symbol(struct[i]) for i in range(len(struct))]
    unknown_elements = sorted(set(elem_sym_all) - ALLOWED_ELEMENTS)
    if unknown_elements:
        status_parts.append('UNKNOWN_ELEMENTS=' + ','.join(unknown_elements))
    V = float(struct.lattice.volume)
    M = structure_mass_amu(struct)
    N = total_atoms_from_composition(struct)
    counts = composition_amounts(struct)
    L = float(struct.lattice.abc[axis])
    if L <= 0:
        L = float(np.linalg.norm(np.array(struct.lattice.matrix[axis], dtype=float)))
    row['V'] = V
    row['M'] = M
    row['N'] = N
    row['C'] = float(counts.get('C', 0.0) / N) if N > 0 else np.nan
    row['H'] = float(counts.get('H', 0.0) / N) if N > 0 else np.nan
    row['L'] = L
    row['S'] = float(V / L) if L > 0 else np.nan
    row['Np'] = float(N / L) if L > 0 else np.nan
    row['Mp'] = float(M / L) if L > 0 else np.nan
    row['rhoN'] = float(N / V) if V > 0 else np.nan
    row['Phi'] = packing_fraction(struct, PACKING_RADIUS_MODE)
    adj, bond_keys, elem_sym, bb_desc, bb_status, bt_used, pf_used, of_used, nb, ns = auto_tune_backbone(struct, axis)
    row['Ncc'] = count_cc_bonds(bond_keys, elem_sym)
    rhoin, Nin = compute_rhoin(struct, bond_keys, axis, CONTACT_CUTOFF, RHOIN_MODE)
    row['rhoin'] = rhoin
    row.update(bb_desc)
    row['Nin'] = Nin
    row['Nb_backbone'] = nb
    row['Ns_side'] = ns
    row['bond_tol_used'] = bt_used
    row['proj_frac_used'] = pf_used
    row['offset_factor_used'] = of_used
    row['packing_radius_mode'] = PACKING_RADIUS_MODE
    row['rhoin_mode'] = RHOIN_MODE
    status_parts.append(bb_status)
    if abs(bt_used - BOND_TOL) > 1e-12 or abs(pf_used - PROJ_FRAC) > 1e-12 or of_used != OFFSET_FACTOR_GRID[0]:
        status_parts.append(f'TUNED(bt={bt_used:.2f},pf={pf_used:.2f},of={of_used})')
    return (row, '__'.join(status_parts))

def blank_descriptor_row(error: str) -> Dict[str, Any]:
    cols = ['V', 'M', 'N', 'C', 'H', 'L', 'S', 'Np', 'Mp', 'Ncc', 'rhoN', 'Phi', 'rhoin', 'Lb', 'Lbq', 'thetab', 'thetabq', 'Psi', 'chi', 'Lstb', 'Nin', 'Nb_backbone', 'Ns_side', 'bond_tol_used', 'proj_frac_used', 'offset_factor_used', 'packing_radius_mode', 'rhoin_mode']
    row = {c: np.nan for c in cols}
    row['packing_radius_mode'] = PACKING_RADIUS_MODE
    row['rhoin_mode'] = RHOIN_MODE
    row['error'] = error
    return row

def load_optional_base(base_xlsx: str) -> Optional[pd.DataFrame]:
    if not os.path.isfile(base_xlsx):
        return None
    try:
        base = pd.read_excel(base_xlsx)
    except Exception:
        return None
    if 'id' not in base.columns:
        return None
    base = base.copy()
    base['id'] = pd.to_numeric(base['id'], errors='coerce')
    base = base.dropna(subset=['id'])
    base['id'] = base['id'].astype(int)
    preferred = ['id', 'kmd', 'Ejn', 'rot_ratio', 'inplane_bond_ratio', 'm_main', 'm_side', 'T']
    keep = [c for c in preferred if c in base.columns]
    if 'id' not in keep:
        keep = ['id']
    return base[keep].drop_duplicates(subset=['id'], keep='first')

def descriptor_map_df() -> pd.DataFrame:
    data = [('V', 'V', 'unit-cell volume', 'Å^3'), ('M', 'M', 'unit-cell mass', 'amu'), ('N', 'N', 'number of atoms in the unit cell', '-'), ('C', 'frac_C', 'fraction of C atoms in the unit cell', '-'), ('H', 'frac_H', 'fraction of H atoms in the unit cell', '-'), ('L', 'lattice_chain', 'lattice constant along the chain direction', 'Å'), ('S', 'A_perp_A2', 'cross-sectional area along the chain direction; S=V/L', 'Å^2'), ('Np', 'Nρ / n_atom_per_A', 'atomic linear density along the chain direction; N/L', 'Å^-1'), ('Mp', 'Mρ / mass_per_A_amu', 'mass linear density along the chain direction; M/L', 'amu Å^-1'), ('Ncc', 'n_cc_bonds', 'number of C-C covalent bonds in the unit cell', '-'), ('rhoN', 'ρN / atom_number_density', 'atomic number density in the unit cell; N/V', 'Å^-3'), ('Phi', 'Φ / packing_fraction', 'unit-cell packing fraction from atom-specific radii', '-'), ('rhoin', 'ρin / interchain_contact_density', 'density of noncovalent atom pairs within 3.5 Å; Nin/V', 'Å^-3'), ('Lb', 'backbone_bondlen_mean', 'mean adjacent backbone bond length', 'Å'), ('Lbq', 'backbone_bondlen_std', 'std of adjacent backbone bond length', 'Å'), ('thetab', 'θb / backbone_bondangle_mean', 'mean angle between two consecutive backbone bonds', 'degree'), ('thetabq', 'θbq / backbone_bondangle_std', 'std of backbone bond angle', 'degree'), ('Psi', 'Ψ / backbone_dihedral_order', 'mean absolute cosine of backbone dihedral angle', '-'), ('chi', 'χ / side_atoms_per_backbone', 'side-chain atoms per backbone atom', '-'), ('Lstb', 'sidechain_extent_mean', 'mean distance from side-chain atoms to nearest backbone atom', 'Å')]
    return pd.DataFrame(data, columns=['short_column', 'long_name', 'meaning', 'unit'])

def radius_table_df(mode: str) -> pd.DataFrame:
    rows = []
    for sym in sorted(ALLOWED_ELEMENTS):
        rows.append({'element': sym, 'radius_mode_for_Phi': mode, 'radius_A_for_Phi': radius_for_element(sym, mode), 'covalent_radius_A_for_bonds': radius_for_element(sym, BOND_RADIUS_MODE)})
    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--out', default=OUT_XLSX)
    parser.add_argument('--base', default=BASE_XLSX)
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    cifs = collect_cifs(root)
    if not cifs:
        raise RuntimeError('No .cif files found under cifx/cify/cifz or cifs_x/cifs_y/cifs_z.')
    rows: List[Dict[str, Any]] = []
    for cid, cif_path, axis, axis_label, folder_name in cifs:
        base_row: Dict[str, Any] = {'id': cid if cid is not None else np.nan, 'cif_folder': folder_name, 'cif_file': os.path.basename(cif_path), 'chain_axis': axis_label}
        try:
            struct = Structure.from_file(cif_path)
            desc, status = compute_descriptors_for_structure(struct, axis)
            base_row.update(desc)
            base_row['cif_status'] = status
            base_row['error'] = ''
        except Exception as exc:
            base_row.update(blank_descriptor_row(f'{type(exc).__name__}: {exc}'))
            base_row['cif_status'] = 'ERROR'
        rows.append(base_row)
    df = pd.DataFrame(rows)
    main_cols = ['id', 'cif_folder', 'cif_file', 'chain_axis', 'V', 'M', 'N', 'C', 'H', 'L', 'S', 'Np', 'Mp', 'Ncc', 'rhoN', 'Phi', 'rhoin', 'Lb', 'Lbq', 'thetab', 'thetabq', 'Psi', 'chi', 'Lstb']
    debug_cols = ['Nin', 'Nb_backbone', 'Ns_side', 'bond_tol_used', 'proj_frac_used', 'offset_factor_used', 'packing_radius_mode', 'rhoin_mode', 'cif_status', 'error']
    for c in main_cols + debug_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[main_cols + debug_cols]
    base = load_optional_base(os.path.join(root, args.base) if not os.path.isabs(args.base) else args.base)
    if base is not None:
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
        df = df.merge(base, on='id', how='left')
        base_cols = [c for c in ['kmd', 'Ejn', 'rot_ratio', 'inplane_bond_ratio', 'm_main', 'm_side', 'T'] if c in df.columns]
        ordered = ['id'] + base_cols + [c for c in df.columns if c not in ['id'] + base_cols]
        df = df[ordered]
    else:
        df.insert(1, 'T', T_DEFAULT)
    out_path = os.path.abspath(args.out)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='descriptors')
        descriptor_map_df().to_excel(writer, index=False, sheet_name='descriptor_map')
        radius_table_df(PACKING_RADIUS_MODE).to_excel(writer, index=False, sheet_name='radius_used')
    print(f'[OK] Wrote: {out_path}')
    print(f'[OK] CIF rows: {len(df)}')
    print(f'[OK] Phi radius mode: {PACKING_RADIUS_MODE}; rhoin mode: {RHOIN_MODE}')
if __name__ == '__main__':
    main()
