#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 09:45:17 2024

@author: vision
"""

import numpy as np
import anndata as ad
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags, coo_matrix, isspmatrix
import pandas as pd
# import anndata as ad
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
import scanpy as sc
import matplotlib.colors as mcolors
import os
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from scipy.cluster.hierarchy import fcluster, linkage, leaves_list, dendrogram, cut_tree
from scipy.spatial.distance import squareform
from typing import Union, Tuple

from scipy.sparse import issparse
from sklearn.cluster import MiniBatchKMeans


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import random
from joblib import Parallel, delayed 
from tqdm import tqdm
import seaborn as sns
import networkx as nx

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import matplotlib.lines as mlines
from matplotlib.patches import Circle
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon


from pandas import DataFrame
from anndata import AnnData
from typing import Optional
from matplotlib.colors import ListedColormap
from matplotlib import cm, colors

from sklearn.preprocessing import LabelEncoder
from collections import Counter
import matplotlib.patheffects as pe



######################### enrichment_heatmap ##################################
# -*- coding: utf-8 -*-




def _assign_color(value, color: list):
    color_dict = dict()
    for i in range(len(value)):
        color_dict[value[i]] = color[i]
    return color_dict


def _convert_pval_to_asterisks(pval):
    if pval <= 0.0001:
        return "***"
    elif pval <= 0.001:
        return "**"
    elif pval <= 0.05:
        return "*"
    return ""


def _set_palette(length):

    if length <= 10:
        vega_10 = list(map(colors.to_hex, cm.tab10.colors))
        vega_10_scanpy = vega_10.copy()
        vega_10_scanpy[2] = '#279e68'  # green
        vega_10_scanpy[4] = '#aa40fc'  # purple
        vega_10_scanpy[8] = '#b5bd61'  # kakhi
        default_10 = vega_10_scanpy
        palette = default_10
    elif length <= 20:
        vega_20 = list(map(colors.to_hex, cm.tab20.colors))
        vega_20_scanpy = [
            *vega_20[0:14:2],
            *vega_20[16::2],
            *vega_20[1:15:2],
            *vega_20[17::2],
            '#ad494a',
            '#8c6d31',
        ]
        vega_20_scanpy[2] = vega_10_scanpy[2]
        vega_20_scanpy[4] = vega_10_scanpy[4]
        vega_20_scanpy[7] = vega_10_scanpy[8] 
        palette = vega_20_scanpy
    elif length <= 28:
        zeileis_28 = [
            "#023fa5", "#7d87b9", "#bec1d4","#d6bcc0","#bb7784","#8e063b","#4a6fe3","#8595e1",
            "#b5bbe3", "#e6afb9", "#e07b91","#d33f6a","#11c638","#8dd593","#c6dec7","#ead3c6",
            "#f0b98d", "#ef9708","#0fcfc0","#9cded6", "#d5eae7", "#f3e1eb", "#f6c4e1", "#f79cd4",
            '#7f7f7f',"#c7c7c7", "#1CE6FF", "#336600",]
        palette = zeileis_28
    # elif length <= 57:
    #     palette = palettes.default_57
    # elif length <= len(palettes.default_102):  # 103 colors
    #     palette = palettes.default_102
    else:
        palette = ['grey' for _ in range(length)]
        print(
            'the obs value has more than 103 categories. Uniform '
            "'grey' color will be used for all categories."
        )

    return palette


def _melt_df(df: DataFrame, library_key: str, select_niche: Optional[list] = None, order: Optional[list] = None, ):
    if select_niche is not None:
        df = df[df['scNiche'].isin(select_niche)]
    if order is not None:
        df['scNiche'] = pd.Categorical(df['scNiche'], categories=order)
    df_melt = pd.melt(df, id_vars=[library_key, 'scNiche', 'Niche_ratio'])
    return df_melt


def stacked_barplot(adata: AnnData, x_axis: str, y_axis: str, mode: str = 'proportion', palette: Optional[list] = None,
                    save: bool = False, save_dir: str = '', kwargs: dict = {}):
    assert (mode.lower() in ['proportion', 'absolute']), 'mode should be either `proportion` or `absolute`!'

    length = len(adata.obs[y_axis].astype('category').cat.categories)
    if palette is None:
        palette = _set_palette(length=length)
    df = adata.obs.groupby([x_axis, y_axis]).size().unstack().fillna(0)
    if mode.lower() == 'proportion':
        df = df.div(df.sum(axis=1), axis=0)

    # plot
    ax = df.plot(kind='bar', stacked=True, width=0.75, color=palette, linewidth=0, **kwargs)
    ax.legend(bbox_to_anchor=(1, 0.5), loc='center left',
              ncol=(1 if length <= 14 else 2 if length <= 30 else 3), frameon=False)
    if save:
        plt.savefig(save_dir, format='svg')


def enrichment_heatmap(cell_type_abundance,
                       pval_adjust,
                       binarized: bool = False,
                       show_pval: bool = False,
                       col_order: Optional[list] = None,
                       row_order: Optional[list] = None,
                       anno_key: Optional[str] = None,
                       anno_palette: Optional[list] = None,
                       save: bool = False,
                       save_dir: str = '',
                       kwargs: dict = {},
                       alpha: float = 0.05,
                       filter_nonsig: bool = True,   # 是否过滤非显著
                       xtick_rotation: int = 0       # 👈 新增参数（默认45°）
                       ):
    fc = cell_type_abundance.T.copy()
    pval = pval_adjust.T.copy()
    kwargs['vmin'] = 0

    # set order
    if col_order is not None:
        fc = fc[col_order]
        pval = pval[col_order]
        kwargs['col_cluster'] = False
    else:
        kwargs['col_cluster'] = True
    if row_order is not None:
        fc = fc.loc[row_order]
        pval = pval.loc[row_order]
        kwargs['row_cluster'] = False
    else:
        kwargs['row_cluster'] = True

    anno = fc.index
    length = len(anno.unique())
    if anno_palette is None:
        anno_palette = _set_palette(length=length)
    row_colors = dict(zip(anno.unique(), anno_palette))

    fc_plot = fc.copy()
    nonsig_mask = pval > alpha

    # 根据参数决定是否过滤
    if filter_nonsig:
        if binarized:
            fc_plot = fc_plot.applymap(lambda x: 0 if x <= 0 else 1)
            fc_plot = fc_plot.mask(nonsig_mask, -1e-9)
            kwargs['vmax'] = 1
            cmap = ListedColormap(['white', 'green'])
            cmap.set_under('white')
            cmap.set_bad('white')
            kwargs['cmap'] = cmap
        else:
            cmap_name = kwargs.get('cmap', 'magma')
            cmap = sns.color_palette(cmap_name, as_cmap=True)
            cmap.set_under('white')
            cmap.set_bad('white')
            kwargs['cmap'] = cmap
            fc_plot = fc_plot.mask(nonsig_mask, -1e-9)
    else:
        # 不过滤
        if binarized:
            fc_plot = fc_plot.applymap(lambda x: 0 if x <= 0 else 1)
            kwargs['vmax'] = 1
            cmap = ListedColormap(['white', 'green'])
            kwargs['cmap'] = cmap
        else:
            cmap_name = kwargs.get('cmap', 'magma')
            cmap = sns.color_palette(cmap_name, as_cmap=True)
            kwargs['cmap'] = cmap

    # 显示显著性星号
    if show_pval:
        pval_str = pval.applymap(_convert_pval_to_asterisks)
        kwargs['annot'] = pval_str
        kwargs['fmt'] = ''

    # 绘制聚类热图
    ax = sns.clustermap(fc_plot,
                        method='complete',
                        row_colors=anno.map(row_colors),
                        **kwargs)

    # 图例
    for label, color in row_colors.items():
        ax.ax_col_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)
    ax.ax_col_dendrogram.legend(
        bbox_to_anchor=(1, 0.5), loc='center left',
        ncol=(1 if length <= 14 else 2 if length <= 30 else 3),
        frameon=False
    )

    # 旋转坐标轴标签
    for tick in ax.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    for tick in ax.ax_heatmap.get_xticklabels():
        tick.set_rotation(xtick_rotation)

    if save:
        plt.savefig(save_dir, bbox_inches='tight', dpi=300)

    return ax

######################### enrichment_heatmap ##################################




def normalize_then_clip(
    adata,
    clip: Union[float, int, Tuple[Union[float, int], Union[float, int]]],
    *,
    target_sum: float = 1e4,
    use_layer: str = None,               # None: 用 X；否则在该 layer 上做
    store_norm_layer: str = 'norm',      # 保存 normalize+log1p 后的矩阵（log 空间）
    store_out_layer: str = 'norm_clipped',   # 保存裁剪/缩放后的矩阵（仍在 log 空间，且已缩放到[0,1]）
    set_X_to_out: bool = True,           # 是否把 X 指向输出矩阵
    skip_zeros: bool = True              # 分位计算是否只基于非零
):
    """
    流程：per-cell normalize (线性域) → log1p (log 域) → 按基因分位 clip/缩放到 [0,1]（仍在 log 域的相对刻度）
    注意：clip 在 log 空间执行；如果你想要线性域的 clip，请在 normalize_total 后、log1p 前处理。
    """

    # ---------- 0) 解析 clip 参数为 (q_low, q_high) in [0,1] ----------
    def _to_qpair(c):
        if isinstance(c, (tuple, list)) and len(c) == 2:
            lo, hi = c
        else:
            lo, hi = None, c
        def _q(v):
            if v is None:
                return None
            v = float(v)
            return v if 0.0 <= v <= 1.0 else v / 100.0
        return _q(lo), _q(hi)

    q_low, q_high = _to_qpair(clip)
    if q_high is None:
        raise ValueError("clip 需要至少提供上分位（如 0.95 或 95），或 (low, high)。")

    # ---------- 1) per-cell normalization（线性域） ----------
    # 在 X 或指定 layer 上做归一化
    sc.pp.normalize_total(adata, target_sum=target_sum, layer=use_layer, inplace=True)

    # ---------- 2) log1p（log 域） ----------
    # 同样在相同来源上做 log1p
    sc.pp.log1p(adata, layer=use_layer, copy=False)

    # 取出“已 normalize+log 的矩阵”
    Xn = adata.layers[use_layer] if use_layer is not None else adata.X
    # 存一份到 store_norm_layer
    adata.layers[store_norm_layer] = Xn.copy() if not sp.issparse(Xn) else Xn.copy().tocsr()

    # ---------- 3) 在 log 空间做按基因分位 clip/缩放到 [0,1] ----------
    if sp.issparse(Xn):
        Xc = Xn.tocsc(copy=True).astype(np.float32)
        indptr = Xc.indptr
        data = Xc.data
        n_genes = Xc.shape[1]
        n_cells = Xc.shape[0]

        for j in range(n_genes):
            start, end = indptr[j], indptr[j+1]
            col = data[start:end]         # 非零值（log 空间）
            if q_low is None:
                # 仅上分位：x / hi，然后裁到 [0,1]
                if col.size == 0:
                    continue
                if skip_zeros:
                    hi = np.percentile(col, q_high * 100.0)
                else:
                    # 把隐式 0 也考虑进来（近似）：把 hi 至少设为 >0
                    hi = np.percentile(np.concatenate([col, np.zeros(n_cells - col.size, dtype=col.dtype)]),
                                       q_high * 100.0)
                hi = 1.0 if hi <= 0 else float(hi)
                col /= hi
                np.clip(col, 0.0, 1.0, out=col)
            else:
                # 下上分位： (x - lo) / (hi - lo) → [0,1]
                if skip_zeros:
                    if col.size == 0:
                        continue
                    lo = np.percentile(col, q_low * 100.0)
                    hi = np.percentile(col, q_high * 100.0)
                else:
                    dense_col = np.concatenate([col, np.zeros(n_cells - col.size, dtype=col.dtype)])
                    lo = np.percentile(dense_col, q_low * 100.0)
                    hi = np.percentile(dense_col, q_high * 100.0)
                if hi <= lo:
                    hi = lo + 1e-6
                col -= lo
                col /= (hi - lo)
                np.clip(col, 0.0, 1.0, out=col)

        Xout = Xc.tocsr()
    else:
        Xarr = np.asarray(Xn, dtype=np.float32).copy()
        n_genes = Xarr.shape[1]

        if q_low is None:
            if skip_zeros:
                for j in range(n_genes):
                    col = Xarr[:, j]
                    nz = col[col > 0]
                    hi = np.percentile(nz, q_high * 100.0) if nz.size > 0 else 1.0
                    hi = 1.0 if hi <= 0 else float(hi)
                    col /= hi
                    np.clip(col, 0.0, 1.0, out=col)
            else:
                hi = np.percentile(Xarr, q_high * 100.0, axis=0)
                hi[hi <= 0] = 1.0
                Xarr /= hi[None, :]
                np.clip(Xarr, 0.0, 1.0, out=Xarr)
        else:
            if skip_zeros:
                for j in range(n_genes):
                    col = Xarr[:, j]
                    nz = col[col > 0]
                    if nz.size == 0:
                        continue
                    lo = np.percentile(nz, q_low * 100.0)
                    hi = np.percentile(nz, q_high * 100.0)
                    if hi <= lo:
                        hi = lo + 1e-6
                    col -= lo
                    col /= (hi - lo)
                    np.clip(col, 0.0, 1.0, out=col)
            else:
                lo = np.percentile(Xarr, q_low * 100.0, axis=0)
                hi = np.percentile(Xarr, q_high * 100.0, axis=0)
                span = np.maximum(hi - lo, 1e-6)
                Xarr = (Xarr - lo[None, :]) / span[None, :]
                np.clip(Xarr, 0.0, 1.0, out=Xarr)

        Xout = Xarr

    # ---------- 4) 写回 ----------
    adata.layers[store_out_layer] = Xout
    if set_X_to_out:
        adata.X = Xout


def _dense_quantile_with_zeros(col_nonzero, n_total, q):
    """
    给定一列的非零值 col_nonzero 和总行数 n_total，
    估计包含隐式 0 的总体 q 分位（q in [0,1]）。
    """
    nnz = col_nonzero.size
    n_zero = n_total - nnz
    if nnz == 0:
        return 0.0 if q <= 1.0 else 0.0
    frac_zero = n_zero / n_total
    if q <= frac_zero:
        return 0.0
    q_eff = (q - frac_zero) / (1 - frac_zero)
    q_eff = min(max(q_eff, 0.0), 1.0)
    return float(np.percentile(col_nonzero, q_eff * 100.0))



def lower_to_upper(char_vec):
    tmp = [s.lower().upper() for s in char_vec]
    # upper_value = set(map(str, tmp))
    return tmp


def search_index(arr1, arr2):
    index_map = {value: idx for idx, value in enumerate(arr1)}
    indices_in_arr2 = [i for i, x in enumerate(arr2) if x in index_map]
    return set(indices_in_arr2)

def integrate_set(set1, set2):
    arr1 = np.array(list(set1))
    arr2 = np.array(list(set2))
    
    intersection = np.intersect1d(arr1, arr2)
    # arr1_diff = np.setdiff1d(arr1, arr2)
    # arr2_diff = np.setdiff1d(arr2, arr1)
    # result = list(np.concatenate((intersection, arr1_diff, arr2_diff)))
    result = list(intersection)
    return result

def scale_ad_mat(A):
    row_sums = A.sum(axis=1)
    row_sums[row_sums == 0] = 1
    inv_row_sums = csr_matrix(1.0 / row_sums)
    A_normalized = A.multiply(inv_row_sums)
    return(A_normalized)

def diag_with_one(A):
    n_rows, n_cols = A.shape
    identity_diagonal = diags([1] * min(n_rows, n_cols), offsets=0, shape=(n_rows, n_cols))
    A_with_identity_diag = A.copy()
    A_with_identity_diag.setdiag(identity_diagonal.diagonal())
    return A_with_identity_diag

def cal_Delaunay(cell_sort_loc):
    tri = Delaunay(cell_sort_loc)


    all_distances = [np.linalg.norm(cell_sort_loc[simplex[i]] - cell_sort_loc[simplex[j]])
                     for simplex in tri.simplices for i in range(3) for j in range(i + 1, 3)]

    mean_distance = np.mean(all_distances)
    std_distance = np.std(all_distances)
    threshold = mean_distance + 3 * std_distance
    
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                p1, p2 = simplex[i], simplex[j]
                distance = np.linalg.norm(cell_sort_loc[p1] - cell_sort_loc[p2])
                if distance <= threshold:
                    edges.add((p1, p2, distance))


    rows, cols, data = zip(*[(u, v, weight) for u, v, weight in edges])

    n_nodes = cell_sort_loc.shape[0]

    ad_matrix = sp.csc_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), 
                               shape=(n_nodes, n_nodes), dtype=np.float32)


    ad_matrix = ad_matrix + ad_matrix.T
    ad_matrix[ad_matrix > 1] = 1

    ad_matrix_diag_one = diag_with_one(ad_matrix)

    dist_weight_matrix = sp.csc_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
    dist_weight_matrix = dist_weight_matrix + dist_weight_matrix.T
    
    return ad_matrix, ad_matrix_diag_one, dist_weight_matrix

   

def cal_K_neighboorhood(
    cell_sort_loc,
    k: int = 10,
    cell_types=None,
    normalization: bool = False,
    sigma_factor: float = 1 / 3,   # 分位数 q ∈ (0,1)
    eps: float = 1e-8,
    return_format: str = "csc",    # "csc" or "csr"
):
    """
    Returns
    -------
    ad_matrix, ad_matrix_diag_one, weight_matrix,
    ad_matrix_same, ad_matrix_same_diag_one, weight_matrix_same
    (scipy.sparse matrix)
    """
    X = np.asarray(cell_sort_loc)
    n = X.shape[0]
    if k < 1:
        raise ValueError("k must be >= 1")
    if n <= k:
        raise ValueError(f"Need N > k. Got N={n}, k={k}")
    if normalization and not (0.0 < sigma_factor < 1.0):
        raise ValueError("sigma_factor must be in (0,1) when normalization=True")
    if return_format not in ("csc", "csr"):
        raise ValueError("return_format must be 'csc' or 'csr'")

    labels = None
    if cell_types is not None:
        labels = np.asarray(cell_types)
        if labels.shape[0] != n:
            raise ValueError("cell_types length must equal number of cells (N).")

    # ---------- KNN（dense 输出：distances/indices 为 (N,k)；后续全 sparse） ----------
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = distances[:, 1:].astype(np.float32, copy=False)  # (n,k)
    indices = indices[:, 1:].astype(np.int32, copy=False)        # (n,k)

    # ---------- sigma_i：每个细胞的 KNN 非零距离的 q 分位数（完全向量化） ----------
    sigma = None
    if normalization:
        nz = distances > 0
        m = nz.sum(axis=1).astype(np.int32)  # 每行非零个数（<=k）

        all_nonzero = distances[nz]
        global_fallback = float(np.quantile(all_nonzero, sigma_factor)) if all_nonzero.size else eps
        global_fallback = max(global_fallback, eps)

        # 用 +inf 填充 0，排序后每行前 m_i 个有效
        tmp = np.where(nz, distances, np.inf)
        tmp_sorted = np.sort(tmp, axis=1)  # (n,k)

        sigma = np.empty(n, dtype=np.float32)
        sigma[m == 0] = global_fallback

        one_mask = (m == 1)
        if np.any(one_mask):
            sigma[one_mask] = np.maximum(tmp_sorted[one_mask, 0], eps)

        many_mask = (m >= 2)
        if np.any(many_mask):
            mm = m[many_mask].astype(np.float32)
            p = sigma_factor * (mm - 1.0)  # 位置 [0, m-1]
            lo = np.floor(p).astype(np.int32)
            hi = np.ceil(p).astype(np.int32)
            frac = (p - lo).astype(np.float32)

            idx = np.where(many_mask)[0]
            v_lo = tmp_sorted[idx, lo]
            v_hi = tmp_sorted[idx, hi]
            sig = (1.0 - frac) * v_lo + frac * v_hi
            sigma[many_mask] = np.maximum(sig, eps)

    # ---------- 距离阈值（mean + 3*std） ----------
    threshold = float(np.mean(distances) + 3.0 * np.std(distances))

    # ---------- 展平 KNN 边（避免 edges list） ----------
    rows_full = np.repeat(np.arange(n, dtype=np.int32), k)
    cols_full = indices.reshape(-1)
    d_full = distances.reshape(-1)

    keep = (d_full > 0) & (d_full <= threshold)
    rows = rows_full[keep]
    cols = cols_full[keep]
    d = d_full[keep]

    # ---------- 工具：对角补1 ----------
    def diag_with_one(A):
        A = A.copy()
        A.setdiag(0)
        A.eliminate_zeros()
        I = sp.identity(A.shape[0], format="csr", dtype=np.float32)
        return (A.tocsr() + I)

    # ---------- 原始图：A（0/1），W（距离或相似度），均对称 ----------
    if rows.size == 0:
        A = sp.csr_matrix((n, n), dtype=np.float32)
        W = sp.csr_matrix((n, n), dtype=np.float32)
    else:
        # 有向邻接
        A_dir = sp.coo_matrix(
            (np.ones(rows.size, dtype=np.float32), (rows, cols)),
            shape=(n, n),
            dtype=np.float32,
        ).tocsr()
        A_dir.sum_duplicates()

        # 对称并集（0/1）
        A = A_dir.maximum(A_dir.T)
        A.sum_duplicates()
        A.data[:] = 1.0

        # 有向权重 D
        if normalization:
            w = np.exp(-d / sigma[rows]).astype(np.float32)
        else:
            w = d.astype(np.float32, copy=False)

        D = sp.coo_matrix((w, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
        D.sum_duplicates()

        # 对称化：W = (D + D^T)/2
        W = (D + D.T) * 0.5
        W.sum_duplicates()
        W.eliminate_zeros()

    A1 = diag_with_one(A).tocsr()

    # ---------- 同类子图 ----------
    if labels is None or rows.size == 0:
        A_same = sp.csr_matrix((n, n), dtype=np.float32)
        W_same = sp.csr_matrix((n, n), dtype=np.float32)
    else:
        same_mask = (labels[rows] == labels[cols])
        rs = rows[same_mask]
        cs = cols[same_mask]
        ds = d[same_mask]

        if rs.size == 0:
            A_same = sp.csr_matrix((n, n), dtype=np.float32)
            W_same = sp.csr_matrix((n, n), dtype=np.float32)
        else:
            A_same_dir = sp.coo_matrix(
                (np.ones(rs.size, dtype=np.float32), (rs, cs)),
                shape=(n, n),
                dtype=np.float32,
            ).tocsr()
            A_same_dir.sum_duplicates()

            A_same = A_same_dir.maximum(A_same_dir.T)
            A_same.sum_duplicates()
            A_same.data[:] = 1.0

            if normalization:
                ws = np.exp(-ds / sigma[rs]).astype(np.float32)
            else:
                ws = ds.astype(np.float32, copy=False)

            D_same = sp.coo_matrix((ws, (rs, cs)), shape=(n, n), dtype=np.float32).tocsr()
            D_same.sum_duplicates()

            W_same = (D_same + D_same.T) * 0.5
            W_same.sum_duplicates()
            W_same.eliminate_zeros()

    A_same1 = diag_with_one(A_same).tocsr()

    # ---------- 输出格式 ----------
    if return_format == "csc":
        return (
            A.tocsc(), A1.tocsc(), W.tocsc(),
            A_same.tocsc(), A_same1.tocsc(), W_same.tocsc()
        )
    else:
        return (
            A.tocsr(), A1.tocsr(), W.tocsr(),
            A_same.tocsr(), A_same1.tocsr(), W_same.tocsr()
        )



def lr_to_spar_mat(lr_network):
    df = lr_network
    ligands = df['from'].unique()
    receptors = df['to'].unique()
    ligand_to_idx = {lig: idx for idx, lig in enumerate(ligands)}
    receptor_to_idx = {rec: idx for idx, rec in enumerate(receptors)}

    row_indices = df['from'].map(ligand_to_idx).values 
    col_indices = df['to'].map(receptor_to_idx).values

    data = np.ones(len(df))
    L = csr_matrix((data, (row_indices, col_indices)), shape=(len(ligands), len(receptors)))
    return ligands, receptors, L


def sparse_to_dense(sparse_mat):
    return sparse_mat.todense()

def dense_to_sparse(dense_mat):
    return coo_matrix(dense_mat)

def adjust_mat(matrix_trs):
    if isspmatrix(matrix_trs):
        return matrix_trs
    elif isinstance(matrix_trs, np.ndarray):
        return coo_matrix(matrix_trs)
    else:
        raise ValueError("Input must be a NumPy matrix or a scipy sparse matrix.")




def multiple_list_to_single(list_list):
    combined_list = [item for sublist in list_list for item in sublist]
    return combined_list

def lr_calculation(X_l, R_weighted, ligands, receptors, L):
    L_coo = L.tocoo()
    
    li = L_coo.row
    rj = L_coo.col
    
    lr_pair = [f'{ligands[l]}_{receptors[r]}' for l, r in zip(li, rj)]
    
    X_l_selected = X_l[:,li]
    R_weighted_selected = R_weighted[:,rj]
    
    if sp.issparse(X_l_selected) or sp.issparse(R_weighted_selected):
        M_tmp = X_l_selected.multiply(R_weighted_selected)
        M = M_tmp.toarray()
    else:
        M = X_l_selected * R_weighted_selected
    
    return M, lr_pair


def cell_type_abundace(adata, ct_vec, dist_weight_matrix):
    n_cells = adata.shape[0]
    le = LabelEncoder()
    label_ids = le.fit_transform(ct_vec)
    k_ct = len(le.classes_)
    M = csr_matrix(
        (np.ones(n_cells, dtype=np.float32), (np.arange(n_cells), label_ids)),
        shape=(n_cells, k_ct), dtype=np.float32)

    # 3) 距离 -> 1/d 权重（防 0，用 eps；可选裁剪极端大权重）
    eps = 1e-8
    W = dist_weight_matrix.tocsr().astype(np.float32).copy()  # 现为“距离”的稀疏矩阵
    W.data = 1.0 / np.maximum(W.data, eps)                    # 变成 1/d 权重


    ct_weighted_counts = W @ M   # (N x C)

# 5) 行归一化 -> 变成“加权频率/比例”（每行和为 1）
    row_sums = np.asarray(ct_weighted_counts.sum(axis=1)).ravel()
    row_sums[row_sums == 0.0] = 1.0
    ct_weighted_freq = sp.diags(1.0 / row_sums) @ ct_weighted_counts
    
    return ct_weighted_freq
    
    

def cal_cell_L_R(
    adata,
    lr_prior,
    cell_type_col,
    k=10,
    weights_joint=1,
    smoothing_ct=False,
    smooth_para=0.5,
    normalization_dist=False,
    sigma_factor_dist=1/3,
    log_trans=True,
    mode="sender",
    eps=1e-12,
):
    """
    计算基于空间邻域的 ligand-receptor 通讯张量，并支持不同 mode。

    Parameters
    ----------
    mode : str
        - "sender"   : 当前细胞作为 ligand 发送者，邻域某类细胞作为 receptor 接收者
                       输出 O_tensor_trs shape = (n_cells, K, m)
        - "receiver" : 当前细胞作为 receptor 接收者，邻域某类细胞作为 ligand 发送者
                       输出 O_tensor_trs shape = (n_cells, K, m)
        - "both"     : 同时计算 sender + receiver，并在最后一维拼接
                       输出 O_tensor_trs shape = (n_cells, K, 2m)

    Returns
    -------
    O_tensor_trs
        - sender / receiver: (n_cells, K, m)
        - both:             (n_cells, K, 2m)

    ct_abundace_mat
        细胞类型 abundance 矩阵

    O_tensor_final
        - sender / receiver: (n_cells, K, m+1)
        - both:             (n_cells, K, 2m+1)

    cell_names
    cell_type_names
    lr_pair
        - sender / receiver: ["L_R", ..., "ct_abundace"]
        - both:             ["L_R_sender", ..., "L_R_receiver", ..., "ct_abundace"]
    """

    # ---------------- 0) 基本信息 ----------------
    cell_names = list(adata.obs_names.values)
    gene_names = list(adata.var_names.values)

    n_cells = adata.shape[0]
    cell_type_names = np.unique(adata.obs[cell_type_col])
    cell_type = np.array(adata.obs[cell_type_col])
    K = len(cell_type_names)

    mode = mode.lower().strip()
    if mode not in ("sender", "receiver", "both"):
        raise ValueError('mode must be "sender", "receiver", or "both"')

    # ---------------- 1) 邻域图 ----------------
    print("Calculation of cell-cell neighborhood matrix....")
    cell_sort_loc = adata.obsm["spatial"]

    ad_matrix, ad_matrix_diag_one, dist_weight_matrix, _, _, _ = cal_K_neighboorhood(
        cell_sort_loc,
        k,
        normalization=normalization_dist,
        sigma_factor=sigma_factor_dist,
    )

    # ---------------- 2) 表达平滑 ----------------
    print("Smoothing the gene expression data....")
    if smoothing_ct:
        _, _, _, ad_matrix_ct, _, _ = cal_K_neighboorhood(
            cell_sort_loc,
            k,
            cell_types=cell_type,
            normalization=normalization_dist,
            sigma_factor=sigma_factor_dist,
        )
        scale_exp = adata.X
        scale_ad_matrix = scale_ad_mat(ad_matrix_ct)
        smooth_exp = scale_exp * (1 - smooth_para) + scale_ad_matrix.multiply(smooth_para) @ scale_exp
    else:
        smooth_exp = adata.X

    # ---------------- 3) 距离/相似度矩阵 ----------------
    print("Calculation of CCI....")
    D_tmp = dist_weight_matrix.multiply(ad_matrix)
    D_inv = D_tmp.copy()

    if not normalization_dist:
        
        D_inv.data = 1.0 / (D_inv.data + eps)
    # 若 normalization_dist=True，则 dist_weight_matrix 已经是 exp(-d/sigma)，直接用

    # ---------------- 4) ligand / receptor 子矩阵 ----------------
    print("Calculation of cell ligands and receptor signal")
    ligands, receptors, L = lr_to_spar_mat(lr_prior)

    l_index = [np.where(np.array(gene_names) == x)[0][0] for x in ligands]
    X_l = smooth_exp[:, l_index]   # (n x p)

    r_index = [np.where(np.array(gene_names) == x)[0][0] for x in receptors]
    X_r = smooth_exp[:, r_index]   # (n x u)

    num_LR_used = int(np.sum(L.todense()))

    # ---------------- 5) 核心：按 mode 计算 (K x n x m) ----------------
    def _compute_mode_tensor(running_mode):
        """
        running_mode: "sender" or "receiver"

        Returns
        -------
        O_mode : np.ndarray
            shape = (K, n_cells, num_LR_used)
        lr_pair_base : list[str]
            长度为 m 的 LR pair 名称，只在 i=0 时生成
        """
        O_mode = np.zeros((K, n_cells, num_LR_used), dtype=np.float32)
        lr_pair_base = None

        for i, class_id in enumerate(cell_type_names):

            cell_mask = (cell_type == class_id).astype(np.float32)
            M_k = sp.diags(cell_mask)   # (n x n)


            D_masked = D_inv @ M_k      # (n x n)

            if running_mode == "sender":

                R_weighted = D_masked @ X_r   # (n x u)


                if i == 0:
                    tmp, lr_pair_base = lr_calculation(X_l, R_weighted, ligands, receptors, L)
                else:
                    tmp, _ = lr_calculation(X_l, R_weighted, ligands, receptors, L)

            elif running_mode == "receiver":

                L_weighted = D_masked @ X_l   # (n x p)


                if i == 0:
                    tmp, lr_pair_base = lr_calculation(L_weighted, X_r, ligands, receptors, L)
                else:
                    tmp, _ = lr_calculation(L_weighted, X_r, ligands, receptors, L)

            else:
                raise ValueError("running_mode must be 'sender' or 'receiver'")

            O_mode[i, :, :] = tmp

        return O_mode, lr_pair_base

    # ---------------- 6) 根据 mode 组装输出 ----------------
    if mode == "sender":
        O_sender, lr_pair_base = _compute_mode_tensor("sender")   # (K, n, m)
        O_tensor_trs = O_sender.transpose(1, 0, 2)                # (n, K, m)
        lr_pair = lr_pair_base[:]

    elif mode == "receiver":
        O_receiver, lr_pair_base = _compute_mode_tensor("receiver")
        O_tensor_trs = O_receiver.transpose(1, 0, 2)
        lr_pair = lr_pair_base[:]

    else:  # mode == "both"
        O_sender, lr_pair_base = _compute_mode_tensor("sender")
        O_receiver, _ = _compute_mode_tensor("receiver")

        O_sender_trs = O_sender.transpose(1, 0, 2)    # (n, K, m)
        O_receiver_trs = O_receiver.transpose(1, 0, 2)  # (n, K, m)

        O_tensor_trs = np.concatenate([O_sender_trs, O_receiver_trs], axis=2)  # (n, K, 2m)
        lr_pair = [f"{x}_sender" for x in lr_pair_base] + [f"{x}_receiver" for x in lr_pair_base]

    # ---------------- 7) log transform ----------------
    if log_trans:
        if sp.issparse(O_tensor_trs):
            data = O_tensor_trs.data
            O_tensor_trs.data = np.log1p(data).astype(np.float32, copy=False)
            O_tensor_trs.eliminate_zeros()
        else:
            O_tensor_trs = np.asarray(O_tensor_trs)
            O_tensor_trs = np.maximum(O_tensor_trs, 0.0)
            O_tensor_trs = np.log1p(O_tensor_trs).astype(np.float32, copy=False)

    # ---------------- 8) cell type abundance ----------------
    print("Calculation of cell type abundace")
    ct_abundace_mat = cell_type_abundace(adata, cell_type, dist_weight_matrix)

    print("Joint the tensor and matrix")
    denom_tensor = O_tensor_trs.sum()
    O_tensor_normal = O_tensor_trs / (denom_tensor + 1e-12)

    denom_mat = ct_abundace_mat.sum()
    ct_abundace_norm = ct_abundace_mat / (denom_mat + 1e-12) * weights_joint

    ct_abundace_tensor = ct_abundace_norm.toarray()[..., None]

    O_tensor_final = np.concatenate([O_tensor_normal, ct_abundace_tensor], axis=2)
    lr_pair.append("ct_abundace")

    return (
        O_tensor_trs,
        ct_abundace_mat,
        O_tensor_final,
        cell_names,
        cell_type_names,
        lr_pair,
    )


    

def select_ct_LR(O, num_common_select = 200):
    ct_L_R = O.columns.tolist()
    ct_vector = [s.split('_')[0] for s in ct_L_R]
    ct_names = np.unique(ct_vector)
    LR_vector = ['_'.join(s.split('_')[1:]) for s in ct_L_R]
    uni_LR, num_LR = np.unique(LR_vector, return_counts=True)
    common_LR = uni_LR[num_LR == len(ct_names)]
    
    index_common = pd.Series(LR_vector).isin(common_LR)
    common_ct_LR = np.array(ct_L_R)[index_common].tolist()
    unique_ct_LR = pd.Series(ct_L_R)[~pd.Series(ct_L_R).isin(common_ct_LR)].tolist()
    
    O_common = O[common_ct_LR]
    O_unique = O[unique_ct_LR]
    
    ###################################### common genes selection #####################################x
    O_common_tmp = np.zeros((len(ct_names), O.shape[0], len(common_LR)))
    cell_type_tmp = []
    target_sum = 1e3
    for i, ct in enumerate(ct_names):
        index_ct = [f'{ct}_{i}' for i in common_LR]
        matrix_created = np.array(O_common[index_ct])
        # total_row = np.sum(matrix_created, axis = 1)
        # matrix_filter = matrix_created[total_row != 0, :]
        total_col = np.sum(matrix_created, axis = 0)
        matrix_normalized = (matrix_created / total_col) * target_sum
        O_common_tmp[i,:,:]= matrix_normalized
        cell_type_tmp.append(np.repeat(ct, matrix_created.shape[0]))
        
    
    O_common_tmp = O_common_tmp.reshape(len(ct_names) * O.shape[0], len(common_LR))
    total_row = np.sum(O_common_tmp, axis=1)
    O_common_created = O_common_tmp[total_row != 0, :]
    cell_type_tmp = np.concatenate(cell_type_tmp)
    cell_type_created = np.array(cell_type_tmp)[total_row != 0].tolist()
    
    #### select cell type unique LR
    adata_common = ad.AnnData(O_common_created, var = pd.DataFrame(index = common_LR))
    sc.pp.log1p(adata_common)
    adata_common.obs['cell_type'] = cell_type_created
    



    sc.tl.rank_genes_groups(adata = adata_common, groupby = 'cell_type', method = 'wilcoxon')


    result = adata_common.uns['rank_genes_groups']


    groups = result['names'].dtype.names  # 获取所有组名
    all_genes = []

    for group in groups:
        # 获取每个组的基因信息
        group_genes = pd.DataFrame({
            'gene': result['names'][group],
            'pval': result['pvals'][group],
            'adj_pval': result['pvals_adj'][group],
            'group': group  # 添加组名信息
        })
        # 筛选条件：pval < 0.05 且 adj_pval < 0.1
        filtered_genes = group_genes[(group_genes['pval'] < 0.05) & (group_genes['adj_pval'] < 0.1)]
        if filtered_genes.shape[0] <= 200:
            top_genes_tmp = filtered_genes['gene'].values
            top_genes = [f'{group}_{i}' for i in top_genes_tmp]
        else:
            top_genes_tmp = filtered_genes.sort_values(by='adj_pval').head(200)
            top_genes = [f'{group}_{i}' for i in top_genes_tmp['gene'].values]
        all_genes.append(top_genes)
    
    all_genes_common =  np.concatenate(all_genes).tolist()
    ###################################### common genes selection #####################################
    ###################################### unique genes selection #####################################
    num_unique = len(all_genes_common)
    adata_unique = ad.AnnData(O_unique)
    sc.pp.normalize_total(adata_unique, target_sum = 1e3)
    sc.pp.log1p(adata_unique)
    sc.pp.highly_variable_genes(adata_unique, flavor="seurat_v3", n_top_genes=num_unique, subset=False) 
    all_genes_unique = adata_unique.var_names[adata_unique.var['highly_variable'].values].tolist()
    
    ###################################### unique genes selection #####################################
    all_genes = all_genes_common + all_genes_unique
    
    return all_genes
    


#####
def louvain_clustering_adata(mat, setting_k, random_state = 2025, resolution = None, neigh = 50, min_res = 0.01,
                             max_res = 8, max_step = 15, tolerance = 0):
    adata_louvain = sc.AnnData(mat)
    sc.pp.neighbors(adata_louvain, n_neighbors=neigh)
    
    if resolution is None:
        this_step = 0
        this_min = float(min_res)
        this_max = float(max_res)
        
        while this_step < max_step:
            this_resolution = this_min + ((this_max - this_min) / 2)
            sc.tl.leiden(adata_louvain, resolution = this_resolution, random_state = random_state)
            res = adata_louvain.obs["leiden"].astype(int)
            this_cluster = len(np.unique(res))
            print(f"Step K: {this_step}, Louvain resolution: {this_resolution}, "
                  f"Number of clusters: {this_cluster}, Ideal of clusters: {setting_k}")
            
            if this_cluster > setting_k + tolerance:
                this_max = this_resolution
            elif this_cluster < setting_k - tolerance:
                this_min = this_resolution
            else:
                print(f'Succeeded in finding clusters: {setting_k} with resolution: {this_resolution}')
                break

            this_step += 1

        if this_step >= max_step:
            print("Cannot find the desired number of clusters.")
            
    else:
        sc.tl.louvain(adata_louvain, resolution = resolution)
        
    
    label = adata_louvain.obs['leiden']
    del adata_louvain
    return label, this_resolution


def leiden_clustering_adata(mat, setting_k, random_state = 2025, resolution = None, neigh = 50, min_res = 0.01,
                             max_res = 8, max_step = 20, tolerance = 0):
    adata_louvain = sc.AnnData(mat)
    sc.pp.neighbors(adata_louvain, n_neighbors=neigh)
    
    if resolution is None:
        this_step = 0
        this_min = float(min_res)
        this_max = float(max_res)
        
        while this_step < max_step:
            this_resolution = this_min + ((this_max - this_min) / 2)
            sc.tl.leiden(adata_louvain, resolution = this_resolution, random_state = random_state)
            res = adata_louvain.obs["leiden"].astype(int)
            this_cluster = len(np.unique(res))
            print(f"Step K: {this_step}, Leiden resolution: {this_resolution}, "
                  f"Number of clusters: {this_cluster}, Ideal of clusters: {setting_k}")
            
            if this_cluster > setting_k + tolerance:
                this_max = this_resolution
            elif this_cluster < setting_k - tolerance:
                this_min = this_resolution
            else:
                print(f'Succeeded in finding clusters: {setting_k} with resolution: {this_resolution}')
                break

            this_step += 1

        if this_step >= max_step:
            print("Cannot find the desired number of clusters.")
            
    else:
        sc.tl.leiden(adata_louvain, resolution = resolution)
        
    
    label = adata_louvain.obs['leiden']
    del adata_louvain
    return label, this_resolution





def _cut_height_for_K(Z, K, N):
    """给定 linkage Z 与样本数 N，返回得到 K 个簇时的一个切割高度（用于画水平线）。"""
    h = np.sort(Z[:, 2])  # 合并高度，长度 N-1
    if N <= 1:
        return 0.0
    eps = (h.max() - h.min()) * 1e-6 if len(h) else 1e-6
    if K <= 1:
        return h[-1] + eps
    if K >= N:
        return h[0] - eps
    idx = N - K - 1
    return (h[idx] + h[idx + 1]) / 2.0    
    


def merge_small_clusters_by_Z(Z, labels, min_size=10, max_passes=3, verbose=False):
    """
    基于层次聚类树 Z，把 size < min_size 的簇合并到 Z 中最近的“兄弟簇”。

    说明：
      - 不改 Z，不重新聚类，只做 post-process
      - 合并依据来自 Z 的合并顺序 => 可解释、可复现
      - max_passes 允许多轮合并（避免合并一次后仍存在小簇）

    参数
    ----
    Z : ndarray, shape (N-1, 4)
        scipy linkage matrix
    labels : ndarray, shape (N,)
        fcluster 得到的 1-based 标签
    min_size : int
        最小允许簇大小
    max_passes : int
        最多合并轮次
    verbose : bool
        打印合并信息

    返回
    ----
    new_labels : ndarray, shape (N,)
        合并后重新编号的 1-based 标签
    """
    labels = np.asarray(labels).astype(int).copy()
    N = len(labels)

    if N <= 1:
        return labels

    # 预构建 node_members：叶子(0..N-1) + 内部节点(N..2N-2)
    node_members = {i: {i} for i in range(N)}
    for i, (a, b, _, _) in enumerate(Z):
        a, b = int(a), int(b)
        node_members[N + i] = node_members[a] | node_members[b]

    def _relabel_compact(lab):
        uniq = np.unique(lab)
        remap = {old: i + 1 for i, old in enumerate(uniq)}
        return np.array([remap[x] for x in lab], dtype=int)

    new_labels = _relabel_compact(labels)

    for p in range(max_passes):
        counts = Counter(new_labels)
        small = sorted([c for c, sz in counts.items() if sz < min_size], key=lambda x: counts[x])

        if len(small) == 0:
            break

        if verbose:
            print(f"[merge pass {p+1}] small clusters:", [(c, counts[c]) for c in small])

        
        cluster_nodes = {}
        for c in np.unique(new_labels):
            members = set(np.where(new_labels == c)[0])
            # 找最小覆盖节点：按节点id从小到大扫（内部节点id越小通常越早合并）
            # 这里用一个简单策略：遍历 node_members 的 key 顺序（叶子+内部）
            for node in range(0, N + (N - 1)):
                if members.issubset(node_members[node]):
                    cluster_nodes[c] = node
                    break

        # 从Z合并顺序中，为每个 cluster 找一个“最近兄弟 cluster”
        # 兄弟的定义：在某次合并中，一侧子树包含 cluster c，而另一侧子树包含某个 cluster d
        cluster_merge_target = {}

        for i, (a, b, _, _) in enumerate(Z):
            a, b = int(a), int(b)

            # 哪些 cluster 完全落在子树 a / b
            clusters_in_a = [c for c, node in cluster_nodes.items()
                             if node_members[node].issubset(node_members[a])]
            clusters_in_b = [c for c, node in cluster_nodes.items()
                             if node_members[node].issubset(node_members[b])]

            # 建立兄弟关系（只记录第一次出现的兄弟 => 最近）
            for ca in clusters_in_a:
                if ca not in cluster_merge_target and len(clusters_in_b) > 0:
                    # 选一个兄弟：这里取 b 侧中“最大簇”更稳（避免小簇合并到另一个小簇）
                    cb = max(clusters_in_b, key=lambda x: counts.get(x, 0))
                    cluster_merge_target[ca] = cb
            for cb in clusters_in_b:
                if cb not in cluster_merge_target and len(clusters_in_a) > 0:
                    ca = max(clusters_in_a, key=lambda x: counts.get(x, 0))
                    cluster_merge_target[cb] = ca

        # 执行合并：小簇 -> 兄弟簇
        for c in small:
            tgt = cluster_merge_target.get(c, None)
            if tgt is None or tgt == c:
                continue
            if verbose:
                print(f"  merge cluster {c} (size={counts[c]}) -> {tgt} (size={counts.get(tgt,0)})")
            new_labels[new_labels == c] = tgt

        new_labels = _relabel_compact(new_labels)

    return new_labels





def louvain_clustering_O_CC(
    adata_filter, O, K_list, n_pc=20, neighborhood=50,
    resolution=None, min_res=0.01, max_res=2,
    interval=0.01, times_random_running=10,
    method_hclust='average', labels=None, truncate=None,
    plot=True, save_pdf=True, save_pdf_name=None,
    # ✅ 使用你们已有程序构建arr
    build_res_by_K=True,
    k_band=(0.7, 1.3),
    min_window=0.05,
    max_window=0.60,
    # ✅ 新增：小簇合并
    merge_small=True,
    min_cluster_size=10,
    merge_passes=3,
    merge_verbose=False,
):
    """
    返回：
      - Z: linkage
      - labels_by_K: 指定K的扁平聚类结果
      - similarity_mat: 共识相似度矩阵
      - (可选) hierarchy_3levels: 三层结构（labels + cuts + edges）
    """

    # -------------------------
    # 原逻辑：参数检查
    # -------------------------
    if any(v is None for v in (min_res, max_res, interval)):
        raise ValueError("Please provide the resolution range (min_res and max_res) and the interval.")
    if min_res <= 0 or max_res <= 0 or max_res <= min_res:
        raise ValueError("min_res and max_res must be positive and max_res > min_res.")

    # -------------------------
    # 原逻辑：预处理
    # -------------------------
    scale_O = StandardScaler().fit_transform(O)
    X = PCA(n_components=n_pc).fit_transform(scale_O) if O.shape[1] > n_pc + 100 else scale_O

    # ✅ 修正：不要 min_res - interval，避免包含0/负值和范围偏移
    arr = np.arange(min_res, max_res + 1e-12, interval)

    adata = sc.AnnData(X)
    sc.pp.neighbors(adata, n_neighbors=neighborhood)

    N = adata_filter.shape[0]
    if X.shape[0] != N:
        raise ValueError(
            f"Row mismatch: O/X has {X.shape[0]} rows but adata_filter has {N} rows. "
            "Make sure O and adata_filter are aligned and ordered consistently."
        )
    
    K_list_clean = sorted(set(int(k) for k in K_list if 1 < int(k) <= N))
    
    def _build_arr_using_your_findres(
        mat, K_list_local, neigh_local, interval_local,
        band=(0.7, 1.3), min_res_local=0.01, max_res_local=2.0,
        min_window_local=0.05, max_window_local=0.60
    ):
        """
        对每个K：
          1) 找 anchor: res_K
          2) 定义簇数扰动范围 [Klo, Khi] = [floor(0.7K), ceil(1.3K)]
          3) 用相同的二分搜索分别找 res_{Klo}, res_{Khi}
          4) 窗口 = [min(res_{Klo}, res_{Khi}), max(...)]
          5) 对窗口做兜底限制：至少 ±min_window，至多 ±max_window
        最后把所有窗口取并集，并用 interval 采样为 arr。
        """
        anchors = {}
        windows = {}

        # 为了避免重复调用过多，可加一个简单cache
        res_cache = {}

        def _find_res_for_K(k_target: int) -> float:
            k_target = int(k_target)
            if k_target in res_cache:
                return res_cache[k_target]
            _, r = louvain_clustering_adata(
                mat=mat, setting_k=k_target, neigh=neigh_local,
                min_res=min_res_local, max_res=max_res_local
            )
            res_cache[k_target] = float(r)
            return float(r)

        arr_list = []

        for K in sorted(set(int(k) for k in K_list_local if int(k) > 1)):
            Klo = max(2, int(np.floor(band[0] * K)))
            Khi = min(N, int(np.ceil(band[1] * K)))

            rK  = _find_res_for_K(K)
            rLo = _find_res_for_K(Klo)
            rHi = _find_res_for_K(Khi)

            lo = float(min(rLo, rHi))
            hi = float(max(rLo, rHi))

            # 裁剪到 [min_res_local, max_res_local]
            lo = max(float(min_res_local), lo)
            hi = min(float(max_res_local), hi)

            # ---- 兜底限制窗口宽度（以 anchor 为中心更稳）
            # 如果窗口太窄，扩到至少 ±min_window
            
            if (hi - lo) < 2.0 * min_window_local:
                lo = max(float(min_res_local), rK - min_window_local)
                hi = min(float(max_res_local), rK + min_window_local)

            # 如果窗口太宽，收缩到至多 ±max_window
            if (hi - lo) > 2.0 * max_window_local:
                lo = max(float(min_res_local), rK - max_window_local)
                hi = min(float(max_res_local), rK + max_window_local)

            anchors[K] = {
                "res_anchor": float(rK),
                "Klo": int(Klo),
                "Khi": int(Khi),
            }
            windows[K] = {
                "lo": float(lo),
                "hi": float(hi),
                "res_for_Klo": float(rLo),
                "res_for_Khi": float(rHi),
            }

            arr_list.append(np.arange(lo, hi + 1e-12, interval_local))

        if len(arr_list) == 0:
            arr = np.arange(min_res_local, max_res_local + 1e-12, interval_local)
        else:
            arr = np.unique(np.round(np.concatenate(arr_list), 10))

        return arr, anchors, windows    
    
    
    # pdb.set_trace()
    windows = {}
    if build_res_by_K and len(K_list_clean) > 0:
        # 这里 mat 直接用 O（与找res阶段一致）。neigh 建议和后面一致。
        arr, anchors, windows = _build_arr_using_your_findres(
            mat=O,
            K_list_local=K_list_clean,
            neigh_local=neighborhood,
            interval_local=interval,
            band=k_band,
            min_res_local=min_res,
            max_res_local=max_res
        )
    else:
        anchors = {}
        windows = {}
        arr = np.arange(min_res, max_res + 1e-12, interval)
    
    
    
    
    co_occurrence_matrix = csr_matrix((N, N), dtype=np.float32)

    for res_tmp in arr:
        print(f'Running resolution:{res_tmp}')
        seeds = random.sample(range(0, 2024), times_random_running)
        for seed in seeds:
            ad = adata.copy()
            ad.obsp = adata.obsp.copy()
            sc.tl.louvain(ad, resolution=float(res_tmp), random_state=int(seed))
            memb = np.asarray(ad.obs['louvain'])
            co_occurrence_matrix += ct_np_co_matrix(memb)

    total_runs = len(arr) * times_random_running
    similarity_mat = co_occurrence_matrix / float(total_runs)

    # -------------------------
    # 距离 + 层次聚类
    # -------------------------
    distance_mat = _to_distance_from_similarity(similarity_mat)
    y = squareform(distance_mat, checks=True)

    print('Staring hierarchical clustering...')
    Z = linkage(y, method=method_hclust)

    # 指定K切割（同一棵Z上切 => 嵌套保证）
    labels_by_K = {K: fcluster(Z, t=K, criterion="maxclust") for K in K_list_clean}

    labels_by_K_merged = {}
    if merge_small:
        for K, lab in labels_by_K.items():
            labels_by_K_merged[K] = merge_small_clusters_by_Z(
                Z, lab, min_size=min_cluster_size, max_passes=merge_passes, verbose=merge_verbose
            )

    # -------------------------
    # plot
    # -------------------------
    if plot:
        fig, ax = plt.subplots(figsize=(10, 4))
        dendrogram(
            Z,
            labels=labels,
            no_labels=(labels is None),
            leaf_rotation=90,
            leaf_font_size=8,
            truncate_mode=("level" if truncate else None),
            p=truncate
        )
        ax.set_title("Consensus hierarchical clustering (dendrogram)")
        ax.set_ylabel("Distance (1 - similarity)")

        for K in K_list_clean:
            t = _cut_height_for_K(Z, K, N)
            ax.axhline(t, linestyle="--", linewidth=1)
            ax.text(ax.get_xlim()[1], t, f"  K={K}", va="center")

        fig.tight_layout()
        if save_pdf and save_pdf_name is not None:
            fig.savefig(save_pdf_name, bbox_inches="tight")
        plt.show()

    extra = {
        "arr_used": arr,
        "anchors": anchors,
        "windows": windows,
        "k_list_clean": K_list_clean,
        "total_runs": int(total_runs),
        "merge_small": bool(merge_small),
        "min_cluster_size": int(min_cluster_size),
        "merge_passes": int(merge_passes),
        "labels_by_K_merged": labels_by_K_merged if merge_small else None,
    }

    return Z, labels_by_K, similarity_mat, extra





########################################## clustering for large dataset #######################


# -*- coding: utf-8 -*-
"""
Mini-Batch KMeans → 固定个数 meta-cells（支持最小簇大小）
↑（细胞层，不建图）
↓
meta 层：构建 kNN（仅 meta 级），用 Louvain 做多分辨率×多种子共识 → 共识矩阵
↓
在 (1 - 共识相似度) 上做层次聚类，按给定 K_list 切树
"""




# -------------------------
# 工具函数
# -------------------------
def _group_mean_sparse(X_sparse, labels, n_meta):
    """对稀疏表达矩阵按簇求均值；返回 (n_meta, n_genes) 的 np.ndarray。"""
    means = []
    for c in range(n_meta):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            means.append(np.zeros((X_sparse.shape[1],), dtype=np.float32))
        else:
            m = X_sparse[idx, :].mean(axis=0)  # 稀疏安全
            means.append(np.asarray(m).ravel().astype(np.float32))
    return np.vstack(means)


def _group_mean_dense(X_dense, labels, n_meta):
    """对稠密特征按簇求均值；返回 (n_meta, d)。"""
    X_dense = np.asarray(X_dense)
    d = X_dense.shape[1]
    out = np.zeros((n_meta, d), dtype=np.float32)
    for c in range(n_meta):
        idx = np.where(labels == c)[0]
        if idx.size > 0:
            out[c, :] = X_dense[idx, :].mean(axis=0).astype(np.float32)
    return out


def _compute_centroids(X, labels, n_meta, use_cosine=True):
    """
    计算每个簇的中心（均值）。use_cosine=True 时采用球面几何（行 L2 归一）。
    X: ndarray 或稀疏矩阵 (n_cells, d)
    """
    # 行归一（逐行缩放，不会计算 NxN）
    if use_cosine:
        Xn = normalize(X, norm='l2', axis=1, copy=True)
    else:
        Xn = X if isinstance(X, np.ndarray) else X.toarray()

    if issparse(Xn):
        cents = []
        for c in range(n_meta):
            idx = np.where(labels == c)[0]
            if idx.size == 0:
                cents.append(np.zeros((Xn.shape[1],), dtype=np.float32))
            else:
                m = Xn[idx, :].mean(axis=0)
                cents.append(np.asarray(m).ravel().astype(np.float32))
        C = np.vstack(cents)
    else:
        C = _group_mean_dense(np.asarray(Xn), labels, n_meta)

    # 余弦几何下把中心也 L2 归一
    if use_cosine:
        C = normalize(C, norm='l2', axis=1, copy=False)
    return C


def _merge_small_clusters_by_centroid(O, labels, min_size=10, use_cosine=True, max_pass=5):
    """
    无图版本：基于簇中心相似度把小簇（<min_size）并到“最近的大簇”。
    - O: 原始表征 (n_cells, d)，可稀疏
    - labels: 初始簇标签（0..C-1）
    - 返回：合并后重新压缩为 0..C'-1 的标签
    """
    labels = pd.Categorical(labels).codes.astype(int)
    n_meta = labels.max() + 1
    for _ in range(int(max_pass)):
        sizes = np.bincount(labels, minlength=n_meta)
        small = np.where(sizes < int(min_size))[0]
        if small.size == 0:
            break

        C = _compute_centroids(O, labels, n_meta, use_cosine=use_cosine)
        sizes = np.bincount(labels, minlength=n_meta)  # 更新一次
        big_mask = sizes >= int(min_size)
        if not np.any(big_mask):
            break  # 极端情况：无“大簇”可并

        for c in small:
            idx_c = np.where(labels == c)[0]
            if idx_c.size == 0:
                continue
            candidate = np.where(big_mask & (np.arange(n_meta) != c))[0]
            if candidate.size == 0:
                candidate = np.where(np.arange(n_meta) != c)[0]

            if use_cosine:
                sims = (C[c][None, :] @ C[candidate].T).ravel()
                to = candidate[int(np.argmax(sims))]
            else:
                diffs = C[candidate] - C[c][None, :]
                d2 = np.einsum('ij,ij->i', diffs, diffs)
                to = candidate[int(np.argmin(d2))]
            labels[idx_c] = int(to)

        labels = pd.Categorical(labels).codes.astype(int)
        n_meta = labels.max() + 1

    return labels


def ct_np_co_matrix(labels):
    """
    给定长度为 n 的簇标签（0..C-1），返回 n×n 的同簇共现矩阵（对角为1）。
    仅用于 meta 层（n≈几百到几千），不会在细胞层使用。
    """
    labels = pd.Categorical(labels).codes
    n = labels.shape[0]
    C = labels.max() + 1
    H = np.zeros((n, C), dtype=np.float32)
    H[np.arange(n), labels] = 1.0
    co = H @ H.T  # 同簇对为1，否则为0
    co[co > 0] = 1.0
    np.fill_diagonal(co, 1.0)
    return co.astype(np.float32)


def _to_distance_from_similarity(S):
    """将相似度矩阵转距离矩阵；确保对称、对角为0。"""
    S = np.asarray(S, dtype=np.float32)
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1.0)
    D = 1.0 - S
    D[D < 0] = 0.0
    np.fill_diagonal(D, 0.0)
    return D


def _find_res_for_K(ad_meta, K_target, seed=2025, lo=0.1, hi=5.0, tol=0.02, max_iter=25):
    """
    在同一张 meta 图上用二分搜索找到导致簇数 ~ K_target 的 Louvain 分辨率。
    """
    lo = float(lo)
    hi = float(hi)
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        tmp = ad_meta.copy()
        tmp.obsp = ad_meta.obsp.copy()
        sc.tl.louvain(tmp, resolution=mid, random_state=int(seed))
        k = int(tmp.obs['louvain'].nunique())
        if abs(k - K_target) <= 1:
            return mid
        if k < K_target:
            lo = max(lo, mid)
        else:
            hi = min(hi, mid)
        if (hi - lo) < tol:
            break
    return (lo + hi) / 2.0


# -------------------------
# 主函数
# -------------------------
def louvain_clustering_O_CC_large_data(
    adata_filter,                      # AnnData：细胞×基因，仅用于计算 meta-cell 平均表达
    O,                                 # ndarray/稀疏：细胞×特征（用于聚类的嵌入）
    K_list,
    # —— 与旧接口兼容的参数（细胞层不再建图、不会使用） —— #
    n_pc=20,
    neighborhood=50,                   # ★在 meta 层建图时使用的 n_neighbors（前 1～3 步保持不变）
    # 共识扫描（Louvain 在 meta 层）
    min_res=None, max_res=None, interval=0.05, times_random_running=3,
    method_hclust='average',           # ← 该入参保留，但在 Step 4 我们会改用 'complete'
    labels=None, truncate=None,
    plot=False, save_pdf=False, save_pdf_name=None,
    # —— 固定簇数的 Mini-Batch k-means（细胞层） —— #
    use_mbkmeans=True,                 # 必须 True
    mbk_n_clusters=2000,               # ★固定 meta-cells 数量（默认 2000）
    mbk_batch_size=4096,
    mbk_max_iter=200,
    mbk_n_init=5,
    mbk_use_cosine=True,               # True: 余弦几何（行 L2 归一）
    mbk_random_state=0,
    min_meta_size=10,                  # ★每个 meta-cell 至少 10 个细胞
    # —— meta 层构造相似度的空间 —— #
    meta_use_O=True,                   # True: 用 O 的簇均值；False: 用表达均值
    consensus_metric_cosine=True,      # meta 层建图时的度量（True→cosine, False→euclidean）
    consensus_seed=2025
):
    """
    返回：
    Z_meta: meta-cell 层面的层次聚类 linkage（complete）
    labels_by_K_meta: {K: meta-cell 的标签（长度 = n_meta，1-based 连续整数）}
    similarity_mat_meta: 共识相似度矩阵 (n_meta×n_meta)
    label1_cells: 初始/修正后的 meta-cells 标签（长度 = n_cells）
    final_cell_labels_by_K: {K: 回映射到细胞层面的最终标签（长度 = n_cells）}
    """
   
    # -------------------------
    K_list = sorted(set(int(k) for k in K_list))
    if len(K_list) == 0:
        raise ValueError("K_list must contain at least one integer > 1.")
    N = adata_filter.shape[0]
    if N != O.shape[0]:
        raise ValueError("adata_filter.n_obs must match O.shape[0] (same number of cells).")
    if not use_mbkmeans:
        raise ValueError("本版本细胞层仅支持 Mini-Batch KMeans。")

    # -------------------------
    # Step 1: 细胞层 - 固定 M 的 Mini-Batch KMeans（球面/余弦可选）
    # -------------------------
    if mbk_n_clusters is None or int(mbk_n_clusters) < 2:
        raise ValueError("请设置 mbk_n_clusters 为固定的 meta-cells 数量（例如 2000）。")
    M = int(mbk_n_clusters)
    M = max(M, max(K_list))  # 确保后续能切到最大 K

    # 行 L2 归一（不会计算 NxN）
    if mbk_use_cosine:
        O_proc = normalize(O, norm='l2', axis=1, copy=True)
    else:
        O_proc = np.asarray(O) if not issparse(O) else O.toarray()
        
    print('Staring Kmeans running----')

    mbk = MiniBatchKMeans(
        n_clusters=M,
        init='k-means++',
        batch_size=int(mbk_batch_size),
        max_iter=int(mbk_max_iter),
        n_init=int(mbk_n_init),
        random_state=int(mbk_random_state),
        verbose=False
    )
    
    print('Kmeans running successfully....')
    label1_cells = mbk.fit_predict(O_proc).astype(int)
    label1_cells = pd.Categorical(label1_cells).codes.astype(int)
    n_meta = int(label1_cells.max()) + 1
    print(f"[Init-MBK] 固定 {M} 个 meta-cells，得到 {n_meta} 个（去重后）。")

    # -------------------------
    # Step 1.5: 最小簇约束（无图，基于中心相似度合并）
    # -------------------------
    if (min_meta_size is not None) and (int(min_meta_size) > 0):
        before = n_meta
        label1_cells = _merge_small_clusters_by_centroid(
            O_proc, label1_cells, min_size=int(min_meta_size),
            use_cosine=mbk_use_cosine, max_pass=5
        )
        n_meta = int(label1_cells.max()) + 1
        if n_meta != before:
            print(f"[MinSize] 小簇并入后：{before} → {n_meta} 个 meta-cells。")
    if n_meta < max(K_list):
        raise ValueError(
            f"小簇合并后 meta-cells 数 {n_meta} 小于所需的最大 K={max(K_list)}，"
            f"请提高 mbk_n_clusters 或降低 min_meta_size。"
        )

    # -------------------------
    # Step 2: 计算 meta 表达与 meta-O（表达用于注释，O 用于构造相似度）
    # -------------------------
    X_exp = adata_filter.X  # 稀疏友好
    meta_expr = _group_mean_sparse(X_exp, label1_cells, n_meta)
    O_dense = O if isinstance(O, np.ndarray) else np.asarray(O.todense())
    meta_O = _group_mean_dense(O_dense, label1_cells, n_meta)

    # -------------------------
    # Step 3: meta 层建图（仅 meta 级） + Louvain 共识
    # -------------------------
    meta_feat = meta_O if meta_use_O else meta_expr
    meta_feat_proc = normalize(meta_feat, norm='l2', axis=1, copy=True) if consensus_metric_cosine else meta_feat

    ad_meta = sc.AnnData(meta_feat_proc)
    sc.pp.neighbors(
        ad_meta,
        n_neighbors=int(neighborhood),
        metric='cosine' if consensus_metric_cosine else 'euclidean'
    )

    minK, maxK = min(K_list), max(K_list)
    if (min_res is None) or (max_res is None):
        min_res = _find_res_for_K(ad_meta, minK, seed=int(consensus_seed), lo=0.1, hi=5.0, tol=0.02)
        max_res = _find_res_for_K(ad_meta, maxK, seed=int(consensus_seed), lo=min_res, hi=max(5.0, min_res + 2.0), tol=0.02)
    print(f"[Meta] Resolution range for scan: [{min_res:.3f}, {max_res:.3f}], step={interval}")

    res_values = np.arange(min_res - interval * 5, max_res + interval * 5, float(interval), dtype=np.float32)
    if res_values.size == 0:
        res_values = np.array([float(min_res)], dtype=np.float32)

    rng = np.random.default_rng(int(consensus_seed))
    co_mat = np.zeros((n_meta, n_meta), dtype=np.float32)
    total_runs = 0

    for res_tmp in res_values:
        seeds = rng.integers(0, 2**31 - 1, size=int(max(1, times_random_running)))
        for sd in seeds:
            ad = ad_meta.copy()
            ad.obsp = ad_meta.obsp.copy()
            sc.tl.louvain(ad, resolution=float(res_tmp), random_state=int(sd))
            memb = pd.Categorical(ad.obs['louvain']).codes.astype(int)
            co_mat += ct_np_co_matrix(memb)
            total_runs += 1

    similarity_mat_meta = co_mat / float(total_runs)
    similarity_mat_meta = 0.5 * (similarity_mat_meta + similarity_mat_meta.T)
    np.fill_diagonal(similarity_mat_meta, 1.0)

    # -------------------------
    # Step 4: 层次聚类（E 策略：基于“1-共识相似度”的 complete linkage + cut_tree）
    # -------------------------
    distance_mat = _to_distance_from_similarity(similarity_mat_meta)
    print('[Meta] Start hierarchical clustering on meta-cells (consensus, complete linkage, no cell-level graph)...')
    y = squareform(distance_mat, checks=False)
    Z_meta = linkage(y, method='complete')  # E: 用 'complete' 替代 'average'

    # 用 cut_tree 直接得到嵌套分割（比 fcluster(maxclust=K) 稳）
    K_list_valid = [k for k in K_list if 1 < int(k) <= n_meta]
    labels_by_K_meta = {int(K): (cut_tree(Z_meta, n_clusters=int(K)).reshape(-1) + 1).astype(int)
                        for K in K_list_valid}

    # -------------------------
    # Step 4.5: 最小簇合并（C 策略，对每个 K 的 meta 标签进行二次保障）
    # -------------------------
    def merge_tiny_meta_clusters(labels_meta, min_size, distance_mat):
        """
        将规模 < min_size 的 meta-簇，按照与其它簇的平均距离，合并到“最近”的簇里。
        输入 labels_meta 可为 1-based 或 0-based，函数会统一并最终返回 1-based 连续整数。
        """
        labels = np.asarray(labels_meta, dtype=int).copy()
        labels = pd.Categorical(labels).codes.astype(int)  # 压缩为 0..C-1
        if labels.size == 0:
            return labels

        while True:
            sizes = np.bincount(labels)
            tiny_clusters = np.where(sizes < int(min_size))[0]
            if tiny_clusters.size == 0 or sizes.size <= 1:
                break

            changed = False
            for c in tiny_clusters:
                idx_c = np.where(labels == c)[0]
                # 候选目标簇
                other_clusters = [t for t in np.unique(labels) if t != c]
                if len(other_clusters) == 0:
                    continue

                best_t, best_d = None, np.inf
                for t in other_clusters:
                    idx_t = np.where(labels == t)[0]
                    d = distance_mat[np.ix_(idx_c, idx_t)].mean()  # 平均距离
                    if d < best_d:
                        best_d, best_t = d, t

                if best_t is not None:
                    labels[idx_c] = best_t
                    changed = True

            if not changed:
                break

            labels = pd.Categorical(labels).codes.astype(int)  # 合并后重新压缩

        return (pd.Categorical(labels).codes.astype(int) + 1).astype(int)  # 返回 1-based

    # 经验阈值：至少 3 个 meta 点，或按 n_meta/200 动态设定
    min_meta_cluster_size = max(10, n_meta // 200)
    for K in list(labels_by_K_meta.keys()):
        labels_by_K_meta[K] = merge_tiny_meta_clusters(
            labels_by_K_meta[K],
            min_size=min_meta_cluster_size,
            distance_mat=distance_mat
        )

    # -------------------------
    # Step 5: 将 meta-cell 的标签映射回单细胞
    # -------------------------
    final_cell_labels_by_K = {
        K: np.asarray(meta_labels, dtype=int)[label1_cells]
        for K, meta_labels in labels_by_K_meta.items()
    }

    return Z_meta, labels_by_K_meta, similarity_mat_meta, label1_cells, final_cell_labels_by_K





def louvain_clustering_O_CC_large_data_hi(
    adata_filter,                      # AnnData：细胞×基因，仅用于计算 meta-cell 平均表达
    O,                                 # ndarray/稀疏：细胞×特征（用于聚类的嵌入）
    K_list,                            # ★必须给定；用于输出这些 K 的结果，并定义 [min..max] 的扫描范围
    # —— 与旧接口兼容的参数（细胞层不再建图、不会使用） —— #
    n_pc=20,
    neighborhood=50,                   # ★在 meta 层建图时使用的 n_neighbors
    # 共识扫描（Louvain 在 meta 层）
    min_res=None, max_res=None, interval=0.05, times_random_running=3,
    method_hclust='average',
    labels=None, truncate=None,
    plot=False, save_pdf=False, save_pdf_name=None,
    # —— 固定簇数的 Mini-Batch k-means（细胞层） —— #
    use_mbkmeans=True,                 # 必须 True
    mbk_n_clusters=2000,               # ★固定 meta-cells 数量（默认 2000）
    mbk_batch_size=4096,
    mbk_max_iter=100,
    mbk_n_init=1,
    mbk_use_cosine=True,               # True: 余弦几何（行 L2 归一）
    mbk_random_state=0,
    min_meta_size=10,                  # ★每个 meta-cell 至少 10 个细胞
    # —— meta 层构造相似度的空间 —— #
    meta_use_O=True,                   # True: 用 O 的簇均值；False: 用表达均值
    consensus_metric_cosine=True,      # meta 层建图时的度量（True→cosine, False→euclidean）
    consensus_seed=2025,
    # —— 最佳 K 甄选（默认：PAC + silhouette 决胜） —— #
    bestK_metric='pac',               
    pac_l=0.1, pac_u=0.9,             # PAC 模糊区间 (l,u)
    tiebreaker_silhouette=True,       # 分数接近时用 silhouette 决胜
    tiebreaker_tol=1e-4               # 决胜阈值
):
    """
    统一返回（共 11 项）：
    1) Z_meta: linkage（meta 层）
    2) labels_by_K_meta: {K: meta 层标签（长度 = n_meta）}（严格只含 K_list 中的 K）
    3) similarity_mat_meta: 共识相似度矩阵 (n_meta×n_meta)
    4) label1_cells: 初始/修正后的 meta-cells（长度 = n_cells）
    5) final_cell_labels_by_K: {K: 细胞层标签（长度 = n_cells）}（K ∈ K_list）
    6) labels_by_level_meta: {}  # 兼容位
    7) final_cell_labels_by_level: {}
    8) best_K: 在 K∈{min(K_list)..max(K_list)}（步长=1）中按 bestK_metric 选出的最佳 K
    9) best_labels_meta: 最佳 K 的 meta 层 1-based 标签
    10) best_labels_cells: 最佳 K 回映射到细胞层的 1-based 标签
    11) bestK_scores: 扫描范围内各 K 的分数字典 {K: score}

    行为：
    - 始终输出 K_list 中各 K 的结果（2,5）。
    - 额外在 [min(K_list)..max(K_list)] 上扫描并返回最佳 K（8-11）。
    """
    

    # -------------------------
    # 参数校验
    # -------------------------
    if K_list is None:
        raise ValueError("K_list 不能为空：用于输出固定 K 的结果，并定义最佳K扫描范围[min..max]。")
    K_list = sorted(set(int(k) for k in K_list))
    if len(K_list) == 0 or any(k <= 1 for k in K_list):
        raise ValueError("K_list 必须包含 >1 的整数。")

    N = adata_filter.shape[0]
    if N != O.shape[0]:
        raise ValueError("adata_filter.n_obs must match O.shape[0] (same number of cells).")
    if not use_mbkmeans:
        raise ValueError("本版本细胞层仅支持 Mini-Batch KMeans。")

    # -------------------------
    # Step 1: 细胞层 - 固定 M 的 Mini-Batch KMeans
    # -------------------------
    if mbk_n_clusters is None or int(mbk_n_clusters) < 2:
        raise ValueError("请设置 mbk_n_clusters 为固定的 meta-cells 数量（例如 2000）。")
    M = int(max(mbk_n_clusters, max(K_list)))  # 确保能切到最大 K

    if mbk_use_cosine:
        O_proc = normalize(O, norm='l2', axis=1, copy=True)
    else:
        O_proc = np.asarray(O) if not issparse(O) else O.toarray()

    mbk = MiniBatchKMeans(
        n_clusters=M,
        init='k-means++',
        batch_size=int(mbk_batch_size),
        max_iter=int(mbk_max_iter),
        n_init=int(mbk_n_init),
        random_state=int(mbk_random_state),
        verbose=False
    )
    label1_cells = mbk.fit_predict(O_proc).astype(int)
    label1_cells = pd.Categorical(label1_cells).codes.astype(int)
    n_meta = int(label1_cells.max()) + 1
    print(f"[Init-MBK] 固定 {M} 个 meta-cells，得到 {n_meta} 个（去重后）。")

    # 最小簇约束
    if (min_meta_size is not None) and (int(min_meta_size) > 0):
        before = n_meta
        label1_cells = _merge_small_clusters_by_centroid(
            O_proc, label1_cells, min_size=int(min_meta_size),
            use_cosine=mbk_use_cosine, max_pass=5
        )
        n_meta = int(label1_cells.max()) + 1
        if n_meta != before:
            print(f"[MinSize] 小簇并入后：{before} → {n_meta} 个 meta-cells。")
    if n_meta < max(K_list):
        raise ValueError(
            f"小簇合并后 meta-cells 数 {n_meta} 小于所需的最大 K={max(K_list)}，"
            f"请提高 mbk_n_clusters 或降低 min_meta_size。"
        )

    # -------------------------
    # Step 2: meta 表达与 meta-O
    # -------------------------
    X_exp = adata_filter.X
    meta_expr = _group_mean_sparse(X_exp, label1_cells, n_meta)
    O_dense = O if isinstance(O, np.ndarray) else np.asarray(O.todense())
    meta_O = _group_mean_dense(O_dense, label1_cells, n_meta)

    # -------------------------
    # Step 3: 共识建图（meta 级）
    # -------------------------
    meta_feat = meta_O if meta_use_O else meta_expr
    meta_feat_proc = normalize(meta_feat, norm='l2', axis=1, copy=True) if consensus_metric_cosine else meta_feat

    ad_meta = sc.AnnData(meta_feat_proc)
    sc.pp.neighbors(
        ad_meta,
        n_neighbors=int(neighborhood),
        metric='cosine' if consensus_metric_cosine else 'euclidean'
    )

    minK, maxK = min(K_list), max(K_list)
    if (min_res is None) or (max_res is None):
        min_res = _find_res_for_K(ad_meta, minK, seed=int(consensus_seed), lo=0.1, hi=5.0, tol=0.02)
        max_res = _find_res_for_K(ad_meta, maxK, seed=int(consensus_seed), lo=min_res, hi=max(5.0, min_res + 2.0), tol=0.02)
    print(f"[Meta] Resolution range for scan: [{min_res:.3f}, {max_res:.3f}], step={interval}")

    res_values = np.arange(min_res - interval * 5, max_res + interval * 5, float(interval), dtype=np.float32)
    if res_values.size == 0:
        res_values = np.array([float(min_res)], dtype=np.float32)

    rng = np.random.default_rng(int(consensus_seed))
    co_mat = np.zeros((n_meta, n_meta), dtype=np.float32)
    total_runs = 0

    for res_tmp in res_values:
        seeds = rng.integers(0, 2**31 - 1, size=int(max(1, times_random_running)))
        for sd in seeds:
            ad = ad_meta.copy()
            ad.obsp = ad_meta.obsp.copy()
            sc.tl.louvain(ad, resolution=float(res_tmp), random_state=int(sd))
            memb = pd.Categorical(ad.obs['louvain']).codes.astype(int)
            co_mat += ct_np_co_matrix(memb)
            total_runs += 1

    similarity_mat_meta = co_mat / float(total_runs)
    similarity_mat_meta = 0.5 * (similarity_mat_meta + similarity_mat_meta.T)
    np.fill_diagonal(similarity_mat_meta, 1.0)

    # -------------------------
    # Step 4: 层次聚类（1-共识相似度）
    # -------------------------
    distance_mat = _to_distance_from_similarity(similarity_mat_meta)
    print('[Meta] Start hierarchical clustering on meta-cells (consensus, no cell-level graph)...')
    y = squareform(distance_mat, checks=False)
    Z_meta = linkage(y, method=method_hclust)  # 'average'/'complete'

    # A) 严格输出 K_list 中每个 K 的标签
    K_list_valid = [k for k in K_list if 1 < int(k) <= n_meta]
    if len(K_list_valid) == 0:
        raise ValueError(f"K_list has no valid K in (1, {n_meta}].")
    labels_by_K_meta = {
        int(K): (cut_tree(Z_meta, n_clusters=int(K)).reshape(-1) + 1).astype(int)
        for K in K_list_valid
    }

    # B) 在[min..max]步长=1的范围内扫描，选择 best_K（默认：PAC）
    K_scan = list(range(minK, maxK + 1))
    K_scan = [k for k in K_scan if 1 < k <= n_meta]
    if len(K_scan) == 0:
        raise ValueError(f"No valid K to scan in range [{minK},{maxK}] with n_meta={n_meta}.")
    cand_labels_by_K = {
        int(K): (cut_tree(Z_meta, n_clusters=int(K)).reshape(-1) + 1).astype(int)
        for K in K_scan
    }
    best_K, bestK_scores = _select_best_K(
        distance_mat=distance_mat,
        labels_by_K_meta=cand_labels_by_K,
        similarity_mat_meta=similarity_mat_meta,
        method=bestK_metric,
        pac_l=pac_l, pac_u=pac_u,
        tiebreaker_silhouette=tiebreaker_silhouette,
        tiebreaker_tol=tiebreaker_tol
    )
    best_labels_meta = cand_labels_by_K[int(best_K)]

    # -------------------------
    # Step 5: 回映射到细胞层
    # -------------------------
    final_cell_labels_by_K = {
        K: np.asarray(meta_labels, dtype=int)[label1_cells]
        for K, meta_labels in labels_by_K_meta.items()
    }
    best_labels_cells = np.asarray(best_labels_meta, dtype=int)[label1_cells]

    labels_by_level_meta = {}
    final_cell_labels_by_level = {}

    # -------------------------
    # 可选绘图
    # -------------------------
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        dendrogram(
            Z_meta,
            labels=labels,
            no_labels=(labels is None),
            leaf_rotation=90,
            leaf_font_size=8,
            truncate_mode=("level" if truncate else None),
            p=truncate
        )
        ax.set_title("Hierarchical clustering on meta-cells (Louvain consensus @ meta level)")
        ax.set_ylabel("Distance (1 - consensus similarity)")

        # 标注 K 线（K_list 与 best_K）
        mark_Ks = sorted(set(list(labels_by_K_meta.keys()) + [int(best_K)]))
        for K in mark_Ks:
            n_leaf = n_meta
            idx = max(0, min(n_leaf - K - 1, len(Z_meta) - 1))
            h = float(Z_meta[idx, 2]) if len(Z_meta) else 0.0
            ax.axhline(h, linestyle='--', linewidth=1)
            ax.text(ax.get_xlim()[1], h, f"  K={K}{' (best)' if K==int(best_K) else ''}", va='center')

        fig.tight_layout()
        if save_pdf:
            if save_pdf_name is None:
                save_pdf_name = 'meta_dendrogram_consensus_meta_graph.pdf'
            fig.savefig(save_pdf_name, bbox_inches='tight')
        plt.show()

    return (
        Z_meta,                       # 1
        labels_by_K_meta,             # 2
        similarity_mat_meta,          # 3
        label1_cells,                 # 4
        final_cell_labels_by_K,       # 5
        labels_by_level_meta,         # 6
        final_cell_labels_by_level,   # 7
        int(best_K),                  # 8
        best_labels_meta,             # 9
        best_labels_cells,            # 10
        bestK_scores                  # 11
    )


# ========================
# 最佳 K 选择器（PAC + silhouette 决胜）
# ========================

def _select_best_K(distance_mat, labels_by_K_meta, similarity_mat_meta=None,
                    method='pac', pac_l=0.1, pac_u=0.9,
                    tiebreaker_silhouette=True, tiebreaker_tol=1e-4):
    Ks = sorted(labels_by_K_meta.keys())
    scores = {}

    if method == 'silhouette':
        from sklearn.metrics import silhouette_score
        for K in Ks:
            labs = np.asarray(labels_by_K_meta[K], dtype=int) - 1
            if len(np.unique(labs)) < 2:
                scores[K] = -1.0
            else:
                try:
                    scores[K] = float(silhouette_score(distance_mat, labs, metric='precomputed'))
                except Exception:
                    scores[K] = -1.0
        rank_key = lambda k: (-scores[k], k)

    elif method == 'consensus':
        if similarity_mat_meta is None:
            raise ValueError("method='consensus' 需要 similarity_mat_meta。")
        S = similarity_mat_meta.copy()
        np.fill_diagonal(S, 1.0)
        for K in Ks:
            labs = np.asarray(labels_by_K_meta[K], dtype=int)
            same = labs[:, None] == labs[None, :]
            np.fill_diagonal(same, False)
            intra = S[same]
            inter = S[~same]
            if intra.size == 0 or inter.size == 0:
                scores[K] = -np.inf
            else:
                scores[K] = float(intra.mean() - inter.mean())
        rank_key = lambda k: (-scores[k], k)

    elif method == 'pac':
        if similarity_mat_meta is None:
            raise ValueError("method='pac' 需要 similarity_mat_meta。")
        l, u = float(pac_l), float(pac_u)
        if not (0.0 <= l < u <= 1.0):
            raise ValueError("pac_l, pac_u 需满足 0 ≤ l < u ≤ 1。")
        S = similarity_mat_meta
        for K in Ks:
            labs = np.asarray(labels_by_K_meta[K], dtype=int)
            same = labs[:, None] == labs[None, :]
            diff = ~same
            same[np.eye(same.shape[0], dtype=bool)] = False
            diff[np.eye(diff.shape[0], dtype=bool)] = False
            # 模糊对：共识值落在 (l,u)
            amb_within = ((S > l) & (S < u) & same).sum()
            amb_between = ((S > l) & (S < u) & diff).sum()
            total_pairs = same.sum() + diff.sum()
            scores[K] = 1.0 - (amb_within + amb_between) / max(1, total_pairs)  # 1-PAC，越大越好
        rank_key = lambda k: (-scores[k], k)

    elif method == 'cdf_auc':
        if similarity_mat_meta is None:
            raise ValueError("method='cdf_auc' 需要 similarity_mat_meta。")
        S = similarity_mat_meta
        grid = np.linspace(0.0, 1.0, 201)
        for K in Ks:
            labs = np.asarray(labels_by_K_meta[K], dtype=int)
            same = labs[:, None] == labs[None, :]
            same[np.eye(same.shape[0], dtype=bool)] = False
            within_vals = S[same]
            between_vals = S[~same]
            if within_vals.size == 0 or between_vals.size == 0:
                scores[K] = -np.inf
                continue
            def cdf_auc(vals):
                vals = np.sort(vals)
                cdf = np.searchsorted(vals, grid, side='right') / max(1, vals.size)
                return np.trapz(cdf, grid)
            auc_within = cdf_auc(within_vals)
            auc_between = cdf_auc(between_vals)
            scores[K] = float(auc_within - auc_between)  # 越大越好
        rank_key = lambda k: (-scores[k], k)

    else:
        raise ValueError("bestK_metric 仅支持 'pac'、'silhouette'、'consensus' 或 'cdf_auc'.")

    best_K = sorted(Ks, key=rank_key)[0]

    # 分数接近时用 silhouette 决胜
    if tiebreaker_silhouette and (method != 'silhouette'):
        from sklearn.metrics import silhouette_score
        bests = [k for k in Ks if abs(scores[k] - scores[best_K]) <= float(tiebreaker_tol)]
        if len(bests) > 1:
            sil_scores = {}
            for k in bests:
                labs0 = np.asarray(labels_by_K_meta[k], dtype=int) - 1
                if len(np.unique(labs0)) < 2:
                    sil_scores[k] = -1.0
                else:
                    try:
                        sil_scores[k] = float(silhouette_score(distance_mat, labs0, metric='precomputed'))
                    except Exception:
                        sil_scores[k] = -1.0
            best_K = sorted(bests, key=lambda k: (-sil_scores[k], k))[0]

    return int(best_K), scores







########################################## clustering for large dataset ####################### 

###################################### calculation LR score ######################################
def compute_pair(idx, i1, i2, distances, X_L, X_R, z_row, z_col):
    cell_1 = i1[idx]
    cell_2 = i2[idx]
    distance = distances[idx]

    # 提取相邻细胞的 ligand 和 receptor 表达，保持稀疏格式
    ligand_expr = X_L.getrow(cell_1)[:, z_row]  # 只取 Z 中配对的 ligand
    receptor_expr = X_R.getrow(cell_2)[:, z_col]  # 只取 Z 中配对的 receptor

    # 计算 ligand-receptor 对的表达乘积并归一化
    product_values = ligand_expr.multiply(receptor_expr).toarray().flatten() * distance

    # 仅保留非零的计算结果
    nonzero_idx = np.nonzero(product_values)[0]
    
    if nonzero_idx.size > 0:
        return {
            'data': product_values[nonzero_idx],
            'rows': [idx] * nonzero_idx.size,
            'cols': nonzero_idx  
        }
    else:
        return None
    
def compute_M_sparse_with_parallel(A, Z, X_L, X_R, n_jobs=-4):
    # 确保 A 是稀疏 coo 矩阵格式
    if not isinstance(A, coo_matrix):
        A = A.tocoo()

    # 获取相邻细胞对的信息
    i1, i2 = A.row, A.col
    distances = A.data

    # 获取 Z 矩阵中有效的 ligand-receptor 配对
    z_row, z_col = Z.nonzero()

    # 将 X_L 和 X_R 转换为 csr 矩阵，以提高逐行访问的效率
    X_L = X_L.tocsr()
    X_R = X_R.tocsr()

    # 使用 joblib 进行并行处理
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(compute_pair)(idx, i1, i2, distances, X_L, X_R, z_row, z_col)
        for idx in tqdm(range(len(i1)), desc="Processing cell pairs")
    )

    # 提取非空结果
    nnz = sum(len(result['data']) for result in results if result is not None)

    # 预分配合适大小的数组以避免多次扩展
    data = np.empty(nnz)
    rows = np.empty(nnz, dtype=int)
    cols = np.empty(nnz, dtype=int)

    idx = 0  # 用于跟踪插入位置
    for result in results:
        if result is not None:
            data_len = len(result['data'])
            data[idx:idx + data_len] = result['data']
            rows[idx:idx + data_len] = result['rows']
            cols[idx:idx + data_len] = result['cols']  # 使用有效的 ligand-receptor 对索引
            idx += data_len

    # 构建最终的稀疏矩阵
    M_sparse = coo_matrix((data, (rows, cols)), shape=(len(i1), len(z_row)))  # 这里的列数为有效的 ligand-receptor 对数

    # 创建 AnnData 对象
    obs_names = [f"cell_{i}-cell_{j}" for i, j in zip(i1, i2)]
    var_names = [f"ligand_{li}-receptor_{ri}" for li, ri in zip(z_row, z_col)]

    adata_score = ad.AnnData(X=M_sparse, obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))

    # 删除 var 中全为零的列
    # adata_score = adata_score[:, adata_score.X.getnnz(axis=0) > 0]

    return adata_score, z_row, z_col


def calculate_score_zone(adata, index_zone, X_smooth, D_inv, L, ligands, receptors, parallel = False):
    gene_names = adata.var_names.values
    # cell_names = adata.obs_names.values
    #### construct X_l, X_r
    l_index = [np.where(gene_names == x)[0][0] for x in ligands if np.any(gene_names == x)]
    X_l = X_smooth[:,l_index]

    
    r_index = [np.where(gene_names == x)[0][0] for x in receptors if np.any(gene_names == x)]
    X_r = X_smooth[:,r_index]

    # check the mat and transfer to sparse.matrix
    X_l_trs = adjust_mat(X_l)
    X_r_trs = adjust_mat(X_r)
    D_inv_trs = adjust_mat(D_inv)
    L_trs = adjust_mat(L)
    
    if index_zone is None:
        D_inv_trs_zone = D_inv_trs
    else:
        D_inv_trs_zone = D_inv_trs[index_zone, :]
    
    
    adata_score, z_row, z_col = compute_M_sparse_with_parallel(A = D_inv_trs_zone, 
                                                               Z = L_trs,
                                                               X_L = X_l_trs, 
                                                               X_R = X_r_trs)
    
    
    
   
    
    # adata_score.var_names = LR_names
    ligand_names = gene_names[z_row]
    receptor_names = gene_names[z_col]
    LR_names = [f"{v1}_{v2}" for v1, v2 in zip(ligand_names, receptor_names)]
    adata_score.var_names = LR_names
    return adata_score



def cal_cell_L_R_score(adata, lr_prior, index_zone = None, ad_matrix = None,
                       dist_weight_matrix = None, smooth_para = 0.5):
    
    # cell_names = list(adata.obs_names.values)
    # first step, Delaunay algorithm
    print('Calculation of cell-cell Delaunay matrix....')
    cell_sort_loc = adata.obsm['spatial']
    
    if (ad_matrix is None) or (dist_weight_matrix is None):
        ad_matrix, ad_matrix_diag_one, dist_weight_matrix = cal_Delaunay(cell_sort_loc)
    # second step, smoothing the gene expression
    print('Smoothing the gene expression data....')
    scale_exp = adata.X
    scale_ad_matrix = scale_ad_mat(ad_matrix)
    
    smooth_exp = scale_exp * (1-smooth_para) + scale_ad_matrix.multiply(smooth_para) @ scale_exp 
    
    # third setep, calculate cell-cell inverse distance matrix
    print('Calculation of CCI....')
    D_tmp = dist_weight_matrix.multiply(ad_matrix)
    # D = diag_with_one(D_tmp)
    D_inv = D_tmp.copy()
    D_inv.data = 1/D_inv.data
    
    # fourth step, construct n^2 * Ligand-Receptor matrix
    ligands, receptors, L = lr_to_spar_mat(lr_prior)
    
    # fifth step, calculate the total signal of ligands and receptors of cells
    print('Calculation of cell ligands and receptor signal ')
    adata_score =  calculate_score_zone(adata, index_zone, smooth_exp, D_inv, L, ligands, receptors)
    
    return adata_score


def percentage_factor(vec, percentence = 0.8):
    sorted_indices = np.argsort(vec)[::-1]
    sorted_vec = vec[sorted_indices]
    cumsum_ratio = np.cumsum(sorted_vec) / sorted_vec.sum()
    cutoff_index = np.argmax(cumsum_ratio >= percentence) + 1
    top_original_indices = sorted_indices[:cutoff_index]
    return top_original_indices

def select_cor_Rank(adata, U_l, V_l, W_l, cell_type, lr_names, domain_index, special_domain,
                    U_l_threshold = 1e-7, p_value_threshold = 0.005, q_value_threshold = 0.01,
                    num_selected = 3, percentence = 0.8):
    ct_vector = adata.obs[domain_index] == special_domain
    U_l_filter = np.where(U_l < U_l_threshold, 0, U_l)
    U_l_domain = U_l_filter[np.array(ct_vector)]
    U_l_other = U_l_filter[~np.array(ct_vector)]
    mean_values = []
    fold_change = []
    p_values = []
    for i in range(U_l_domain.shape[1]):
        mean_domain = np.mean(U_l_domain[:,i])
        mean_other = np.mean(U_l_other[:,i])
        fold_change_tmp = np.log2((mean_domain + 1e-6)/(mean_other + 1e-6))
        # _, p_value_tmp = stats.mannwhitneyu(U_l_domain[:,i], U_l_other[:,i])
        _, p_value_tmp = stats.ttest_ind(U_l_domain[:,i], U_l_other[:,i])
        mean_values.append(mean_domain)
        fold_change.append(fold_change_tmp)
        p_values.append(p_value_tmp)
    
    _, p_values_corrected, _, _ = multipletests(p_values, method = 'bonferroni')
    
    mean_values = np.array(mean_values)
    fold_change = np.array(fold_change)
    p_values = np.array(p_values)
    p_values_corrected = np.array(p_values_corrected)
    
    
    index_p_values = np.where(p_values < p_value_threshold)[0]
    index_p_values_corrected = np.where(p_values_corrected < q_value_threshold)[0]
    valid_indexes = np.intersect1d(index_p_values, index_p_values_corrected)
    print(f'The number of differentially expression Ranks is: {len(valid_indexes)}')
    if len(valid_indexes) <= num_selected:
        top_indexes = valid_indexes
    else:
        sorted_indexes = valid_indexes[np.argsort(p_values_corrected[valid_indexes])]
        selected_local_indexes = percentage_factor(vec = mean_values[sorted_indexes], percentence = percentence)
        top_indexes = sorted_indexes[selected_local_indexes]
        # sorted_indexes = valid_indexes[np.argsort(-mean_values[valid_indexes])]
        # top_indexes = sorted_indexes[:num_selected]
        
    
    cell_type = np.array(cell_type)
    lr_names = np.array(lr_names)
    results = {}
    for i in range(len(top_indexes)):
        top_ct_index = percentage_factor(vec = V_l[:,top_indexes[i]], percentence = percentence)
        top_lr_index = percentage_factor(vec = W_l[:,top_indexes[i]], percentence = percentence)
        results[f'Rank_{top_indexes[i]}'] = {}
        results[f'Rank_{top_indexes[i]}'][f'Rank_{top_indexes[i]}_ct'] = cell_type[top_ct_index]
        results[f'Rank_{top_indexes[i]}'][f'Rank_{top_indexes[i]}_lr'] = lr_names[top_lr_index]
    
    return results, mean_values, fold_change



def plot_rank_importance(results_list_loading, V_l_loading, Cluster_label_loading, 
                         color_dict_loading, save=False, file_names='',
                         bar_width=0.2, gap_between_bars=0.05):
    cell_type_names_loading = list(color_dict_loading.keys())
    cell_type_names_loading = np.array(cell_type_names_loading).astype(str)
    data_loading = []
    ct_list_loading = []
    rank_list_loading = []

    # 构建数据
    for i in range(len(results_list_loading)):
        results_tmp = results_list_loading[i]
        rank_used = list(results_tmp.keys())
        rank_return = ['_'.join(s.split('_')[1:]) for s in rank_used]
        rank_list_loading.append(rank_return)
        rank_cluster_data = [int(tmp) for tmp in rank_return]
        data_tmp = []
        ct_tmp = []
        for j in rank_cluster_data:
            ct_index_tmp = results_tmp[f'Rank_{j}'][f'Rank_{j}_ct'].tolist()
            ct_index_tmp = np.array(ct_index_tmp).astype(str)
            ct_position = [np.where(cell_type_names_loading == x)[0][0] for x in ct_index_tmp]
            vec_tmp = list(V_l_loading[np.array(ct_position), np.array(j)])
            data_tmp.append(vec_tmp)
            ct_tmp.append(ct_index_tmp)
        data_loading.append(data_tmp)
        ct_list_loading.append(ct_tmp)

    n_groups = len(data_loading)
    group_labels = Cluster_label_loading

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_groups)

    plotted_cell_types = set()  # 用于控制图例唯一性

    for i, group_data in enumerate(data_loading):
        n_ranks = len(group_data)
        group_width = n_ranks * (bar_width + gap_between_bars)

        for j, bar_stack in enumerate(group_data):
            bar_x = x[i] - group_width / 2 + j * (bar_width + gap_between_bars)
            bottom = 0
            for k, height in enumerate(bar_stack):
                cell_type = ct_list_loading[i][j][k]
                color = color_dict_loading.get(cell_type, 'gray')  # 若无颜色，用灰色

                label = cell_type if cell_type not in plotted_cell_types else None
                plotted_cell_types.add(cell_type)

                ax.bar(bar_x, height, width=bar_width, bottom=bottom, color=color, label=label)
                bottom += height

    # 设置 X 轴标签
    ax.set_xticks([])  # 移除默认 xtick
    transform = ax.get_xaxis_transform()

    # 上层：Rank 标签
    for i, group_data in enumerate(data_loading):
        n_ranks = len(group_data)
        group_width = n_ranks * (bar_width + gap_between_bars)
        for j in range(n_ranks):
            bar_x = x[i] - group_width / 2 + j * (bar_width + gap_between_bars)
            ax.text(bar_x, -0.03, rank_list_loading[i][j], ha='center', va='top', fontsize=9, transform=transform)

    # 下层：Cluster 标签
    for i in range(n_groups):
        ax.text(x[i], -0.10, group_labels[i], ha='center', va='top', fontsize=10, fontweight='bold', transform=transform)

    ax.set_ylabel("Cell type cumsum loading")
    ax.set_title("Ranks factor loading for selected niches")

    # 设置图例
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), title="Cell Types", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig(file_names)
    plt.show()
       
        
def plot_ligand_receptor_network(pairs, layout='kamada', figsize=(10, 8), save=False, file_name=''):
    """
    绘制配体-受体有向网络图。
    
    参数：
        pairs: list of str, 如 ['A_B', 'C_D'] 表示配体_受体对。
        layout: str, 可选布局方式 ['kamada', 'spring', 'shell', 'spectral']。
        figsize: tuple, 图像尺寸。
        save: bool, 是否保存图像。
        file_name: str, 保存的文件名。
    """
    # 拆分为边
    edges = [pair.split('_') for pair in pairs]

    # 创建有向图
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # 布局方式
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:  # 默认 kamada-kawai
        pos = nx.kamada_kawai_layout(G)

    # 绘图
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1400, edgecolors='black')
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=40, edge_color='black', width=2)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')

    plt.title("Ligand–Receptor Directed Network", fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    if save:
        plt.savefig(file_name, dpi=300)
    plt.show()    


def drop_lastrow_and_col_l2norm(W_l):
    # 1) 扔掉最后一行
    W = W_l[:-1, :].astype(np.float32, copy=False)

    # 2) 每列 L2 范数
    col_norm = np.linalg.norm(W, axis=0)  # shape: (k,)

    # 3) 防止除以 0（全零列保持为零列）
    col_norm_safe = np.where(col_norm == 0, 1.0, col_norm)

    # 4) 列归一化
    W_norm = W / col_norm_safe
    return W_norm



######################################### visualization ################################
def is_color_similar_to_gray(color, gray="#D3D3D3", threshold=0.1):
    """
    Check if a given color is similar to gray using RGB distance.
    """
    rgb_color = np.array(mcolors.to_rgb(color))
    rgb_gray = np.array(mcolors.to_rgb(gray))
    return np.linalg.norm(rgb_color - rgb_gray) < threshold

def get_non_gray_colors(palette, gray="#D3D3D3", threshold=0.1):
    """
    Filter out colors from the palette that are too similar to gray.
    """
    return [color for color in palette if not is_color_similar_to_gray(color, gray, threshold)]

    

def remove_outliers_iqr(data, threshold=1.5):
    """
    Remove outliers using the IQR method for a 2D numpy array.

    Parameters:
        data (np.ndarray): shape (n_samples, 2)
        threshold (float): IQR multiplier, default 1.5

    Returns:
        np.ndarray: filtered coordinates
    """
    x, y = data[:, 0], data[:, 1]

    def iqr_filter(coord):
        q1, q3 = np.percentile(coord, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (coord >= lower) & (coord <= upper)

    mask = iqr_filter(x) & iqr_filter(y)
    return data[mask]

def plot_clusters_in_one_figure(adata, cluster_col, save_path, save_file, size_scatter=0.6,
                                 library_id="151673", spatial_key="spatial", n_col=4, dpi=100,
                                 threshold_filtered = 1.0):
    """
    Plot spatial cluster maps with convex hull and strict outlier removal (IQR-based).

    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with spatial coordinates and cluster labels.
    cluster_col : str
        Column name in `adata.obs` containing cluster annotations.
    save_path : str
        Directory to save the resulting plot.
    save_file : str
        File name of the saved plot (e.g., 'clusters.png').
    size_scatter : float
        Size of the scatter points.
    library_id : str
        Spatial library ID used in spatial plots.
    spatial_key : str
        Key in `.obsm` storing spatial coordinates.
    n_col : int
        Number of columns in subplot grid.
    dpi : int
        Resolution (dots per inch) for the output figure.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    unique_clusters = adata.obs[cluster_col].cat.categories
    num_clusters = len(unique_clusters)
    
    area_list = []

    # Prepare subplot grid
    n_rows = int(np.ceil(num_clusters / n_col))
    fig, axs = plt.subplots(n_rows, n_col, figsize=(n_col * 4, n_rows * 4))
    axs = np.ravel(axs)

    for idx, cluster in enumerate(unique_clusters):
        ax = axs[idx]

        # Highlight current cluster
        adata.obs['highlight'] = adata.obs[cluster_col].apply(
            lambda x: cluster if x == cluster else 'Other'
        ).astype('category')

        # Set fixed colors
        adata.uns['highlight_colors'] = ["#d62728", "#D3D3D3"]  # red and gray

        # Plot spatial
        sc.pl.spatial(adata, img_key="hires", color='highlight', size=size_scatter,
                      ax=ax, show=False, legend_loc=None, title=cluster)

        # Remove axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Get spatial coordinates
        coords = adata.obsm["spatial"]
        cluster_mask = adata.obs['highlight'] == cluster
        cluster_coords = coords[cluster_mask.values]

        # Remove spatial outliers using IQR method
        try:
            filtered_coords = remove_outliers_iqr(cluster_coords, threshold=threshold_filtered)

            # Draw convex hull
            if filtered_coords.shape[0] >= 3:
                hull = ConvexHull(filtered_coords)
                for simplex in hull.simplices:
                    ax.plot(
                        filtered_coords[simplex, 0],
                        filtered_coords[simplex, 1],
                        linestyle='--', color='black', linewidth=1
                    )
                tmp = hull.area / (filtered_coords.shape[0])
                area_list.append(tmp)
        except Exception as e:
            print(f"[Warning] ConvexHull failed for cluster {cluster}: {e}")

    # Remove unused axes
    for j in range(num_clusters, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    save_file_final = os.path.join(save_path, save_file)
    print(f"Figure saved to: {save_file_final}")
    plt.savefig(save_file_final, dpi=dpi)
    plt.close()
    
    return unique_clusters, area_list
    
### cell type abundace    





    

############### cell type abundace and cell-cell contact ###################
def colormap(cell_type_name):
    palette = sns.color_palette("colorblind", len(cell_type_name))
    color_ct_dic =  {cell_type: palette[i] for i, cell_type in enumerate(cell_type_name)}
    return color_ct_dic
    



def inter_label(label_1, label_2):
    
    unique_labels_1 = np.unique(label_1) 
    unique_labels_2 = np.unique(label_2)
    
    count_df = pd.DataFrame(0, index=unique_labels_1, columns=unique_labels_2)
    ratio_df = pd.DataFrame(0.0, index=unique_labels_1, columns=unique_labels_2)
    
    
    for label in unique_labels_1:
        mask = label_1 == label  
        label_2_subset = label_2[mask]  
    
        unique_label_2, counts = np.unique(label_2_subset, return_counts=True) 
        total = len(label_2_subset)  


        count_df.loc[label, unique_label_2] = counts  
        # count_df.columns.values[0] = 'index'
        ratio_df.loc[label, unique_label_2] = counts / total 
        # ratio_df.columns.values[0] = 'index'
        
    return count_df, ratio_df

def neighborhood_ration(neighborhood_ct_pd, domain_label_vec):
    
    neighborhood_ct_mat = np.array(neighborhood_ct_pd)
    row_sums = np.sum(neighborhood_ct_mat, axis = 1) + 1e-10
    neighborhood_ct_ratio = neighborhood_ct_mat / row_sums[:, np.newaxis]
    ct_label = np.unique(np.array(domain_label_vec))
    means = []
    for label in ct_label:
        class_data = neighborhood_ct_ratio[domain_label_vec == label]
        class_mean = np.mean(class_data, axis=0)
        means.append(class_mean)
    means = np.array(means)
    row_sum_mean = np.sum(means, axis = 1) + 1e-20
    
    means_final = means / row_sum_mean[:, np.newaxis]
    
    df_means = pd.DataFrame(means_final, index = ct_label, 
                            columns = neighborhood_ct_pd.columns)
    return df_means



def plot_clustered_heatmap(data_matrix, similarity_metric='cosine', clustering_method='average', cmap='RdBu_r', figsize=(10, 10), save_figure = None):
    """
    绘制行和列聚类的热图，矩阵对角线呈现块状结构。

    参数:
    data_matrix : np.ndarray
        输入矩阵 (n x m)，行为样本，列为低维表示特征。
    similarity_metric : str
        用于计算相似性的度量方法，例如 'cosine' 或 'correlation'。
    clustering_method : str
        层次聚类方法，默认 'average'。
    cmap : str
        热图的颜色映射，默认 'RdBu_r'。
    figsize : tuple
        热图的尺寸，默认 (10, 10)。

    返回:
    None
    """
    if similarity_metric == 'cosine':
        similarity_matrix = cosine_similarity(data_matrix)
    elif similarity_metric == 'correlation':
        similarity_matrix = np.corrcoef(data_matrix)
    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

    # Step 2: 计算行和列的聚类顺序
    row_linkage = linkage(1 - similarity_matrix, method=clustering_method)
    col_linkage = linkage(1 - similarity_matrix.T, method=clustering_method)
    row_order = leaves_list(row_linkage)  # 行聚类顺序
    col_order = leaves_list(col_linkage)  # 列聚类顺序

    # Step 3: 根据聚类顺序重新排列矩阵
    clustered_matrix = similarity_matrix[np.ix_(row_order, col_order)]

    # Step 4: 绘制热图
    plt.figure(figsize=figsize)
    sns.heatmap(
        clustered_matrix,
        cmap=cmap,
        square=False,
        cbar=False,
        xticklabels=False,  # 不显示列标签
        yticklabels=False,  # 不显示行标签
        annot=False,        # 不显示数值
    )
    if save_figure:
        plt.savefig(save_figure, bbox_inches='tight', dpi=50)  # 高质量保存
        print(f"Heatmap saved to {save_figure}")
    
    plt.show()

    
def enrichment_analysis(adata, id_key, val_key):
    print(f"Calculating the enrichment of each cluster ({id_key}) in group ({val_key})...")
    obs = adata.obs.copy()
    id_list = sorted(list(set(obs[id_key])))
    val_list = sorted(list(set(obs[val_key])))
    
    df_count = obs.groupby([id_key, val_key]).size().unstack().fillna(0)

    # 计算比例，避免误差
    MIN_NUM = 20
    df_count.loc[:, df_count.sum() < MIN_NUM] = 0
    df_normalized = df_count.div(df_count.sum(axis=0), axis=1)
        
    
    
    pval = []
    pval_adj = []
    N = adata.shape[0]
    for idx in id_list:
        K = df_count.loc[idx].sum()
        
        pval_tmp = []

        for val in val_list:
            n = df_count[val].sum()
            k = df_count.loc[idx,val]
            
            p_value = stats.hypergeom.sf(k-1, N, K, n)
            pval_tmp.append(p_value)
        
        _, p_adj_tmp, _, _ = multipletests(pval_tmp, method = 'fdr_bh')
        pval.append(pval_tmp)
        pval_adj.append(p_adj_tmp) 
    
    pval = pd.DataFrame(pval)
    pval_adj = pd.DataFrame(pval_adj)
    pval.columns = pval_adj.columns = val_list
    pval.index = pval_adj.index = id_list
    
    df_normalized = df_normalized.reindex(index=pval.index, columns=pval.columns)
    
    return df_normalized, pval, pval_adj

def spatial_contigue_analysis(location, class_labels, save = True, save_fig = ''):
    unique_classes = np.unique(class_labels)
    
    x_coords = np.array(location[:,0])
    y_coords = np.array(location[:,1])
    area_dict = {}
    median_distance_dict = {}
    
    
    for cls in unique_classes:
        class_points = np.array([(x_coords[i], y_coords[i]) for i in range(len(class_labels)) if class_labels[i] == cls])

        # 计算两两点的欧式距离
        distances = pdist(class_points)
        median_distance = np.median(distances)

        # 计算 KDE 熵
        hull = ConvexHull(class_points)
        area_all = (hull.area)/class_points.shape[0]
        # area_all = (hull.area)

        # 存储
        area_dict[cls] = area_all
        median_distance_dict[cls] = median_distance
        
    color_map = plt.get_cmap("tab20")  # tab10 提供 10 种高区分度颜色
    colors = {cls: color_map(i % 10) for i, cls in enumerate(unique_classes)}
    
    plt.figure(figsize=(8,6))
    for cls in unique_classes:
        plt.scatter(median_distance_dict[cls], area_dict[cls],
                    color=colors[cls], label=cls, s=100, edgecolors='black')
        plt.text(median_distance_dict[cls] + 0.4, area_dict[cls] + 0.2,
                        cls, fontsize=8, verticalalignment='center')
    
    plt.xlabel("Median distance")
    plt.ylabel("Area/Sample")
    # plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Classes")
    plt.grid(True)
    if save:
        plt.savefig(save_fig)
    plt.show()
    return median_distance_dict, area_dict
        

def neighboor_enrichment_analysis_single(
    adata,
    domain_index,
    special_spatial_domain,
    cell_type_index,
    k=10,
    permutation=1000,
    eps=1e-6,
    random_state=None,
    use_expanded_localization=False,
    alpha=0.0,
    precomputed_adj=None  # 新增：可传入(已计算的)邻接矩阵
):
    """
    返回：fold_change, p_value_pd, ratio_pd, localization_A_df
    （与你当前版本一致，只是多了 precomputed_adj 这个可选参数）
    """
    rng = np.random.default_rng(random_state)

    # 1) 邻接：如果外面已算好就复用
    if precomputed_adj is None:
        ad_matrix, _, _, _, _, _ = cal_K_neighboorhood(
            adata.obsm["spatial"], k=k
        )
    else:
        ad_matrix = precomputed_adj
    ad_matrix = ad_matrix.tocsr()

    # 2) 选 domain 及其一阶邻居
    obs = adata.obs
    domain_mask = (obs[domain_index].values == special_spatial_domain)
    domain_idx = np.where(domain_mask)[0]

    A_dom_all = ad_matrix[domain_idx, :]
    neighbors_bool = (A_dom_all.getnnz(axis=0) > 0)
    neighbor_idx = np.where(neighbors_bool)[0]
    selected_idx = np.union1d(domain_idx, neighbor_idx)

    # 子邻接：domain 行 × (domain∪邻居) 列
    A = ad_matrix[domain_idx, :][:, selected_idx]

    # 3) 真实计数（类型×类型）
    ct_all = obs.iloc[selected_idx][cell_type_index].astype(str).values
    ct_dom = obs.iloc[domain_idx][cell_type_index].astype(str).values
    types_all = np.unique(ct_all)
    types_dom = np.unique(ct_dom)

    
    def _one_hot_sparse(labels, categories):
        cat2idx = {c:i for i,c in enumerate(categories)}
        row = np.arange(labels.size, dtype=int)
        col = np.fromiter((cat2idx[x] for x in labels), dtype=int, count=labels.size)
        data = np.ones(labels.size, dtype=np.float64)
        return csr_matrix((data, (row, col)), shape=(labels.size, categories.size))

    D_all = _one_hot_sparse(ct_all, types_all)
    D_dom = _one_hot_sparse(ct_dom, types_dom)

    truth_counts = (D_dom.T @ A @ D_all).toarray()

    # 4) 置换（富集单侧）
    ge_count = np.zeros_like(truth_counts, dtype=np.int64)
    sum_perm = np.zeros_like(truth_counts, dtype=np.float64)

    sel_pos = {ix: i for i, ix in enumerate(selected_idx)}
    dom_pos = np.fromiter((sel_pos[ix] for ix in domain_idx), dtype=int, count=domain_idx.size)

    for _ in range(permutation):
        perm_ct_all = ct_all.copy()
        rng.shuffle(perm_ct_all)
        perm_ct_dom = perm_ct_all[dom_pos]

        P_all = _one_hot_sparse(perm_ct_all, types_all)
        P_dom = _one_hot_sparse(perm_ct_dom, types_dom)
        perm_counts = (P_dom.T @ A @ P_all).toarray()

        ge_count += (perm_counts >= truth_counts).astype(np.int64)
        sum_perm += perm_counts

    pvals = (ge_count + 1.0) / (permutation + 1.0)
    expected = sum_perm / permutation

    ratio = (truth_counts + eps) / (expected + eps)
    fc = np.log2(np.maximum(ratio, 1.0))

    fold_change   = pd.DataFrame(fc,     index=types_dom, columns=types_all)
    p_value_pd    = pd.DataFrame(pvals,  index=types_dom, columns=types_all)
    ratio_pd      = pd.DataFrame(ratio,  index=types_dom, columns=types_all)

    # 5) 方案A：localization（0–1）
    ct_global = obs[cell_type_index].astype(str).value_counts()
    all_types_sorted = ct_global.index.astype(str)

    if use_expanded_localization:
        expanded_bool = (A_dom_all.getnnz(axis=0) > 0)
        expanded_idx = np.union1d(domain_idx, np.where(expanded_bool)[0])
        in_set = pd.Index(obs.index[expanded_idx])
    else:
        in_set = pd.Index(obs.index[domain_idx])

    x_counts  = obs.loc[in_set, cell_type_index].astype(str).value_counts()
    x_aligned = x_counts.reindex(all_types_sorted,  fill_value=0).astype(float)
    N_aligned = ct_global.reindex(all_types_sorted, fill_value=0).astype(float)

    K = 1
    localization_values = (x_aligned + alpha) / (N_aligned + alpha * K)
    localization_A_df = pd.DataFrame(
        localization_values.values,
        index=all_types_sorted,
        columns=[str(special_spatial_domain)],
        dtype=float
    )

    return fold_change, p_value_pd, ratio_pd, localization_A_df

def neighboor_enrichment_analysis_multi(
    adata,
    domain_index,
    cell_type_index,
    domains=None,      # None 则自动用 obs[domain_index].unique()
    k=10,
    permutation=1000,
    eps=1e-6,
    random_state=None,
    use_expanded_localization=False,
    alpha=0.0
):
    """
    对多个 domain 依次计算结果，返回:
    results = {
      domain_name: {
        "fold_change": DataFrame,
        "p_value_pd": DataFrame,
        "ratio_pd": DataFrame,
        "localization_A_df": DataFrame
      },
      ...
    }
    """
    # 预先计算一次邻接，供每个域复用（省时）
    adj, _, _, _, _, _ = cal_K_neighboorhood(adata.obsm["spatial"], k=k)
    adj = adj.tocsr()

    # 域列表
    if domains is None:
        domains = list(pd.Series(adata.obs[domain_index].values).unique())

    results = {}
    for d in domains:
        print(f'Dealing with {d}')
        fc, pv, rt, locA = neighboor_enrichment_analysis_single(
            adata=adata,
            domain_index=domain_index,
            special_spatial_domain=d,
            cell_type_index=cell_type_index,
            k=k,
            permutation=permutation,
            eps=eps,
            random_state=random_state,
            use_expanded_localization=use_expanded_localization,
            alpha=alpha,
            precomputed_adj=adj  # 复用邻接
        )
        results[str(d)] = {
            "fold_change": fc,
            "p_value_pd": pv,
            "ratio_pd": rt,
            "localization_A_df": locA
        }
    return results


    
    
def jaccrad_sim_rank(vec1, vec2):
    set1 = set(vec1)
    set2 = set(vec2)
    jaccard = len(set1 & set2) / len(set1 | set2)
    return jaccard
    
    



def select_ranks(anndata_mat, U, niche_idx,
                 threshold_filter=0.1,
                 pval_threshod=0.05,
                 pval_adjust_threshod=0.1,
                 col_order=None,
                 row_order=None,
                 plot_fig=True,
                 save=True,
                 save_fig=None,
                 filter_nonsig = True,
                 transfer: bool = False  # 👈 新增参数
                 ):
    """
    根据富集分析选择显著的rank。
    transfer=True 时，在绘图时矩阵转置，并自动交换 row_order 与 col_order。
    """

    # --- 归一化 ---
    rows_sum = np.sum(U, axis=1, keepdims=True)
    U_normalized = U / rows_sum

    # --- 找出每个样本最大rank ---
    max_columns = np.argmax(U_normalized, axis=1)
    anndata_tmp = anndata_mat.copy()
    anndata_tmp.obs['rank_idx'] = max_columns
    
    # pdb.set_trace()

    # --- 富集分析 ---
    df_normalized, pval, pval_adj = enrichment_analysis(
        anndata_tmp,
        id_key='rank_idx',
        val_key=niche_idx
    )

    # --- 阈值过滤 ---
    mask = df_normalized < threshold_filter
    pval[mask] = 1
    pval_adj[mask] = 1

    # --- 绘图部分 ---
    if plot_fig:
        kwargs = {
            'figsize': (8, 8),
            'vmax': 1,
            'cmap': 'YlOrBr',
            'linewidths': 0,
            'linecolor': 'white'
        }

        if transfer:
            df_plot = df_normalized.T
            pval_plot = pval_adj.T
            row_order_new = col_order  # ✅ 交换
            col_order_new = row_order
        else:
            df_plot = df_normalized
            pval_plot = pval_adj
            row_order_new = row_order
            col_order_new = col_order
            
        
        # pdb.set_trace()

        enrichment_heatmap(
            cell_type_abundance=df_plot,
            pval_adjust=pval_plot,
            save=save,
            show_pval=plot_fig,
            kwargs=kwargs,
            filter_nonsig = filter_nonsig,
            col_order=col_order_new,
            row_order=row_order_new,
            save_dir=save_fig
        )

    # --- 筛选显著索引 ---
    condition = (pval <= pval_threshod) & (pval_adj <= pval_adjust_threshod)
    significant_indices = {}

    for col_name in condition.columns:
        row_indices = condition.index[condition[col_name]].tolist()
        if len(row_indices) > 0:
            filtered_row_indices = [
                idx for idx in row_indices
                if df_normalized.loc[idx, col_name] >= threshold_filter
            ]
            if len(filtered_row_indices) > 0:
                significant_indices[col_name] = filtered_row_indices

    return df_normalized, pval, pval_adj, significant_indices





def Bubble_plot(df, scatter_size=300, fig_width=8, fig_length=6, legend_avail = True, save_fig = None, transfer = False):
    # 设置气泡图的大小
    # df = df.applymap(lambda x: x if x >= 0.1 else 0)  # 小于0.1的元素置为0
    # df = df.loc[~(df.sum(axis=1) == 0)]  # 删除所有行值和为0的行
    if transfer:
        df = df.T
        df = df.loc[:, ~(df.sum(axis=0) == 0)]  # 过滤全为0的列
    else:
        df = df.loc[~(df.sum(axis=1) == 0), :]  # 过滤全为0的行
    
    
    plt.figure(figsize=(fig_width, fig_length))
    
    q = df.where(df.ne(0)).stack().quantile([0.1, 0.5, 0.8])  # 返回一个 Series，索引是 0.333333 和 0.666667
    q_r = q.round(2)
    q_1_5 = q_r.loc[0.1]
    q_1_3 = q_r.loc[0.5]
    q_2_3 = q_r.loc[0.8]

    # 自定义颜色映射
    def get_color(value):
        if value > q_2_3:
            return (0.8, 0, 0)  # 深红色
        elif q_1_3 <= value <= q_2_3:
            return (0, 0, 1)  # 蓝色
        elif q_1_5 <= value < q_1_3:
            return (0.5, 0.5, 0.5)  # 灰色
        else:
            return (0, 0, 0, 0)  # 小于0.1不显示（透明）

    # 绘制气泡图
    for i, ligand in enumerate(df.index):
        for j, cell_group in enumerate(df.columns):
            value = df.iloc[i, j]  # 获取数值
            size = value * scatter_size  # 气泡的大小
            color = get_color(value)  # 获取颜色
            plt.scatter(j, i, s=size, alpha=0.6, c=[color])  # 绘制气泡，交换i和j的位置

    # 设置标签和标题
    plt.yticks(range(len(df.index)), df.index, fontsize = 8)  # 设置y轴标签为行名
    plt.xticks(range(len(df.columns)), df.columns, fontsize = 8, rotation=90)  # 设置x轴标签为列名
    plt.title('Bubble Plot with Custom Colors')

    # 创建颜色图例
    legend_labels = [f'> {q_2_3}', f'[{q_1_3}, {q_2_3}]', f'[{q_1_5}, {q_1_3}]']
    legend_colors = [(0.8, 0, 0), (0, 0, 1), (0.5, 0.5, 0.5)]

    # 使用Line2D来创建圆形标记的图例
    legend_handles = [mlines.Line2D([], [], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
                      for color, label in zip(legend_colors, legend_labels)]

    # 添加图例
    if legend_avail:
        plt.legend(handles=legend_handles, loc='upper left')

    # 展示图形
    plt.tight_layout()
    if save_fig is not None:
        plt.savefig(save_fig)
    plt.show()
    
    return df











# 定义自定义圆形图例 handler
class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = (xdescent + width / 2, ydescent + height / 2)
        p = Circle(xy=center, radius=min(width, height) / 3,
                   facecolor=orig_handle.get_facecolor(),
                   edgecolor=orig_handle.get_edgecolor(),
                   transform=trans)
        return [p]





def plot_core_network_gradient_ribbon(
    df_adj,
    palette_dict,
    core_color,
    core_node=None,

    figsize=(6.2, 6.6),

    # 节点大小
    outer_node_radius=0.25,
    core_node_radius=0.35,

    # 边样式
    edge_color="#A9C4EB",
    min_width=2.0,
    max_width=8.0,
    alpha_min=0.10,
    alpha_max=0.95,
    n_segments=80,

    # 箭头头参数
    arrow_head_length=0.2,
    arrow_head_width=0.1,
    arrow_head_alpha=None,

    # 自动布局参数
    radius=2.8,
    start_angle=140,
    end_angle=-140,
    y_compress=0.95,

    # 可选：手动覆盖
    node_positions=None,
    curve_offsets=None,

    title=None,
    bg_color="white",
    add_core_glow=True,
    glow_color=None,
    save=None,
    dpi=300
):
    """
    通用版：
    1. 默认自动布局
    2. 可用 node_positions 手动覆盖布局
    3. 默认自动曲率
    4. 可用 curve_offsets 手动覆盖曲率
    """

    # -------------------------
    # 0. 输入检查
    # -------------------------
    if not isinstance(df_adj, pd.DataFrame):
        raise TypeError("df_adj must be a pandas DataFrame.")
    if df_adj.shape[0] != df_adj.shape[1]:
        raise ValueError("df_adj must be a square adjacency matrix.")
    if list(df_adj.index) != list(df_adj.columns):
        raise ValueError("df_adj index and columns must match in the same order.")

    df = df_adj.copy().astype(float)

    for n in df.index:
        df.loc[n, n] = 0.0

    if core_node is None:
        core_node = df.sum(axis=1).idxmax()

    if core_node not in df.index:
        raise ValueError(f"core_node '{core_node}' not found in df_adj.")

    weights_series = df.loc[core_node]
    weights_series = weights_series[weights_series > 0]

    if len(weights_series) == 0:
        raise ValueError(f"No positive outgoing edges found from core node '{core_node}'.")

    target_nodes = weights_series.index.tolist()
    weights = weights_series.values.astype(float)
    n_targets = len(target_nodes)

    # -------------------------
    # 1. 自动布局 / 手动布局
    # -------------------------
    pos = {}

    # 核心节点
    if node_positions is not None and core_node in node_positions:
        pos[core_node] = node_positions[core_node]
    else:
        pos[core_node] = (0.0, 0.0)

    # 外围节点
    if node_positions is not None:
        # 优先使用用户传入的位置
        missing_nodes = [n for n in target_nodes if n not in node_positions]

        for n in target_nodes:
            if n in node_positions:
                pos[n] = node_positions[n]

        # 对没有传位置的节点，自动补位
        if missing_nodes:
            auto_angles = np.linspace(start_angle, end_angle, len(missing_nodes))
            for n, ang in zip(missing_nodes, auto_angles):
                rad = np.deg2rad(ang)
                x = radius * np.cos(rad)
                y = radius * np.sin(rad) * y_compress
                pos[n] = (x, y)
    else:
        # 全自动布局
        auto_angles = np.linspace(start_angle, end_angle, n_targets)
        for n, ang in zip(target_nodes, auto_angles):
            rad = np.deg2rad(ang)
            x = radius * np.cos(rad)
            y = radius * np.sin(rad) * y_compress
            pos[n] = (x, y)

    # -------------------------
    # 2. 权重映射
    # -------------------------
    w_min, w_max = weights.min(), weights.max()

    def scale_weight(w, out_min, out_max):
        if np.isclose(w_min, w_max):
            return (out_min + out_max) / 2
        return out_min + (w - w_min) * (out_max - out_min) / (w_max - w_min)

    # -------------------------
    # 3. Bezier工具
    # -------------------------
    def bezier_quad(p0, p1, p2, t):
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

    def bezier_quad_tangent(p0, p1, p2, t):
        return 2 * (1 - t) * (p1 - p0) + 2 * t * (p2 - p1)

    # -------------------------
    # 4. 自动曲率 / 手动曲率
    # -------------------------
    if curve_offsets is None:
        # 自动生成：从正到负，居中对称
        auto_offsets = np.linspace(0.35, -0.35, n_targets)
        curve_offset_map = {n: v for n, v in zip(target_nodes, auto_offsets)}
    else:
        # 支持 dict 或 list
        if isinstance(curve_offsets, dict):
            curve_offset_map = {n: curve_offsets.get(n, 0.0) for n in target_nodes}
        else:
            curve_offsets = list(curve_offsets)
            if len(curve_offsets) != n_targets:
                raise ValueError("If curve_offsets is a list, its length must match the number of target nodes.")
            curve_offset_map = {n: v for n, v in zip(target_nodes, curve_offsets)}

    if arrow_head_alpha is None:
        arrow_head_alpha = alpha_max

    # -------------------------
    # 5. 画布
    # -------------------------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(bg_color)

    # -------------------------
    # 6. 渐变丝带边
    # -------------------------
    for node, w in zip(target_nodes, weights):
        p0 = np.array(pos[core_node], dtype=float)
        p2 = np.array(pos[node], dtype=float)

        mid = (p0 + p2) / 2.0
        vec = p2 - p0
        dist = np.linalg.norm(vec)
        if dist == 0:
            continue

        perp = np.array([-vec[1], vec[0]]) / dist
        offset = curve_offset_map.get(node, 0.0)
        p1 = mid + perp * offset

        width = scale_weight(w, min_width, max_width)

        ts = np.linspace(0.10, 0.87, n_segments)
        pts = np.array([bezier_quad(p0, p1, p2, t) for t in ts])

        for i in range(len(ts) - 1):
            xseg = [pts[i, 0], pts[i + 1, 0]]
            yseg = [pts[i, 1], pts[i + 1, 1]]
            alpha = alpha_min + (alpha_max - alpha_min) * (i / (len(ts) - 2)) ** 1.15

            ax.plot(
                xseg, yseg,
                color=edge_color,
                alpha=alpha,
                linewidth=width,
                solid_capstyle="round",
                zorder=1
            )

        # 箭头头部与边宽解耦
        t_head = 0.91
        tip = bezier_quad(p0, p1, p2, t_head)
        tan = bezier_quad_tangent(p0, p1, p2, t_head)
        tan_norm = np.linalg.norm(tan)
        if tan_norm == 0:
            continue

        direction = tan / tan_norm
        normal = np.array([-direction[1], direction[0]])

        base_center = tip - direction * arrow_head_length
        left = base_center + normal * arrow_head_width
        right = base_center - normal * arrow_head_width

        arrow_head = Polygon(
            [tip, left, right],
            closed=True,
            facecolor=edge_color,
            edgecolor=edge_color,
            alpha=arrow_head_alpha,
            linewidth=0,
            zorder=2
        )
        ax.add_patch(arrow_head)

    # -------------------------
    # 7. 核心节点光晕
    # -------------------------
    if glow_color is None:
        glow_color = core_color

    if add_core_glow:
        cx, cy = pos[core_node]
        ax.scatter(cx, cy, s=5800, color=glow_color, alpha=0.08, zorder=2)
        ax.scatter(cx, cy, s=4100, color=glow_color, alpha=0.12, zorder=2)
        ax.scatter(cx, cy, s=2900, color=glow_color, alpha=0.10, zorder=2)

    # -------------------------
    # 8. 节点
    # -------------------------
    ax.add_patch(
        Circle(
            pos[core_node],
            radius=core_node_radius,
            facecolor=core_color,
            edgecolor="black",
            linewidth=1.3,
            zorder=3
        )
    )

    for node in target_nodes:
        ax.add_patch(
            Circle(
                pos[node],
                radius=outer_node_radius,
                facecolor=palette_dict.get(node, "#D9D9D9"),
                edgecolor="black",
                linewidth=1.1,
                zorder=3
            )
        )

    # -------------------------
    # 9. 标签
    # -------------------------
    ax.text(
        *pos[core_node], core_node,
        ha="center", va="center",
        fontsize=12.5, fontweight="bold",
        color="white", zorder=4
    )

    for node in target_nodes:
        ax.text(
            *pos[node], node,
            ha="center", va="center",
            fontsize=11, fontweight="bold",
            color="black",
            zorder=4
        )

    # -------------------------
    # 10. 坐标范围自动适配
    # -------------------------
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]

    ax.set_xlim(min(xs) - 0.9, max(xs) + 0.9)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    ax.set_aspect("equal")
    ax.axis("off")

    if title is not None:
        plt.title(title, fontsize=13, fontweight="bold", pad=8)

    plt.tight_layout()

    if save is not None:
        plt.savefig(f"{save}.pdf", bbox_inches="tight")
        plt.savefig(f"{save}.svg", bbox_inches="tight")
        plt.savefig(f"{save}.png", dpi=600, bbox_inches="tight")

    plt.show()


def gene_network_plot_single(df, column_index, node_size=500,  
                             node_font_size=10,  
                             edge_scater = 4, fig_width=4, fig_high=3, save_fig = None):
    # pdb.set_trace()
    gene_interactions = {
    f"{row}": df.at[row, column_index]
    for row in df.index
    if df.at[row, column_index] > 0  # 只保留大于0的值
    }
    
    
    G = nx.DiGraph()
    for interaction, weight in gene_interactions.items():
        ligand, receptor = interaction.split('_')  # 拆分ligand和receptor
        G.add_edge(ligand, receptor, weight=weight)  # 添加有向边，权重为该向量的值


    # 绘制基因网络图
    plt.figure(figsize=(fig_width, fig_high))

    # 使用circular_layout布局（圆形布局）
    # if len(G.nodes) == 3:
    #     pos = nx.spring_layout(G)  # 只有两个节点时使用 shell_layout
    # else:
    #     pos = nx.circular_layout(G)  # 否则使用 circular_layout
    pos = nx.spring_layout(G)

    # 获取边的权重
    edges = G.edges(data=True)

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size= node_size, node_color='lightblue')

    # 绘制边，边的粗细根据权重值来决定
    edge_widths = [d['weight'] * edge_scater for u, v, d in edges]  # 权重值 * 10 来决定边的宽度

    # 绘制边，使用弧线连接，并设置黑色线条，增加箭头大小
    nx.draw_networkx_edges(
        G, 
        pos, 
        edgelist=edges, 
        width=edge_widths, 
        alpha=0.6, 
        edge_color='black',  # 黑色线条
        arrows=True,  # 开启箭头
        arrowsize=25,  # 增大箭头
        connectionstyle='arc3, rad=0.1'  # 使用弧线
    )

    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size = node_font_size)

    # 设置图标题

    # 隐藏坐标轴
    plt.axis('off')

    # 调整布局并显示图形
    plt.tight_layout()
    if save_fig is not None:
        plt.savefig(save_fig)
    plt.show()
    plt.close()
    
    # return fig


def gene_network_plot_based(df, column_index, ax, node_size=500,  
                             node_font_size=10,  
                             edge_scater = 4, fig_width=4, fig_high=3, save_fig = None):
    
    gene_interactions = {
    f"{row}": df.at[row, column_index]
    for row in df.index
    if df.at[row, column_index] > 0  # 只保留大于0的值
    }
    
    
    G = nx.DiGraph()
    for interaction, weight in gene_interactions.items():
        ligand, receptor = interaction.split('_')  # 拆分ligand和receptor
        G.add_edge(ligand, receptor, weight=weight)  # 添加有向边，权重为该向量的值

    # 使用circular_layout布局（圆形布局）
    # if len(G.nodes) == 2:
    #     pos = nx.spring_layout(G)  # 只有两个节点时使用 shell_layout
    # else:
    #     pos = nx.circular_layout(G)  # 否则使用 circular_layout
    pos = nx.spring_layout(G)

    # 获取边的权重
    edges = G.edges(data=True)

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size= node_size, node_color='lightblue', ax = ax)

    # 绘制边，边的粗细根据权重值来决定
    edge_widths = [d['weight'] * edge_scater for u, v, d in edges]  # 权重值 * 10 来决定边的宽度

    # 绘制边，使用弧线连接，并设置黑色线条，增加箭头大小
    nx.draw_networkx_edges(
        G, 
        pos, 
        edgelist=edges, 
        width=edge_widths, 
        alpha=0.6, 
        edge_color='black',  # 黑色线条
        arrows=True,  # 开启箭头
        arrowsize=15,  # 增大箭头
        connectionstyle='arc3, rad=0.1',  # 使用弧线
        ax = ax
    )

    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size = node_font_size, ax = ax)

    # 设置图标题

    # 隐藏坐标轴
    plt.axis('off')

    # 调整布局并显示图形
    plt.tight_layout()
    if save_fig is not None:
        plt.savefig(save_fig)


def plot_multiple_gene_networks(df, columns, rows=1,  node_size=500, node_font_size=10, 
                                edge_scater=4, fig_width=10, fig_high=3, save_fig=None):
    # 创建一个大图，指定子图的布局
    len_columns = len(columns)
    fig_width = len_columns * 4
    fig, axes = plt.subplots(rows, len_columns, figsize=(fig_width, fig_high))
    axes = axes.flatten()  # 将2D数组的子图变成1D，便于索引

    # 遍历每一个列，绘制基因网络图
    for idx, column_index in enumerate(columns):
        ax = axes[idx]
        gene_network_plot_based(df, column_index, ax, node_size=node_size, node_font_size=node_font_size,
                          edge_scater=edge_scater, fig_width=fig_width, fig_high=fig_high, save_fig=save_fig)

    # 调整子图间距，确保不会重叠
    plt.tight_layout()

    # 保存图像
    if save_fig is not None:
        plt.savefig(save_fig)
    
    plt.show()
    plt.close()        
       
                
        
def keep_until_pct(
    df: pd.DataFrame,
    threshod: float = 0.9,              # 第1阶段：累计阈值（默认90%）
    cols=None,
    inplace: bool = False,
    post_top_n: int | dict | None = None,  # 第2阶段：在“已保留集合”中再取Top-N（int或{列名:N}）
    include_ties: bool = False,            # True时包含与第N名同值的并列
) -> pd.DataFrame:
    """
    两阶段列内筛选（只处理数值列或手动指定的列）：
      阶段1：按值从大到小累计，累计和达到/超过 threshod*总和 为止，保留到该项（含），其余置0；
      阶段2（可选）：在阶段1“已保留”的集合中，再保留最大的前N个（post_top_n），其余置0。
    NaN 不参与排序/累计与Top-N，原位保持为 NaN。
    """

    out = df if inplace else df.copy()
    if cols is None:
        cols = out.select_dtypes(include=[np.number]).columns

    # 将 post_top_n 归一化成映射
    post_map: dict = {}
    if isinstance(post_top_n, int):
        post_map = {c: int(post_top_n) for c in cols}
    elif isinstance(post_top_n, dict):
        post_map = {c: int(v) for c, v in post_top_n.items() if c in cols}

    for c in cols:
        s = out[c]
        m = s.notna()
        if not m.any():
            continue

        vals = s[m]
        total = vals.sum()
        if not np.isfinite(total) or total <= 0:
            # 总和非正或无有效值：不改动该列
            continue

        # ---------- 阶段1：累计阈值 ----------
        idx_sorted = vals.sort_values(ascending=False, kind="mergesort").index
        cs = vals.loc[idx_sorted].cumsum().to_numpy()
        thr = float(threshod) * total
        hit = np.nonzero(cs >= thr)[0]
        pos = (hit[0] + 1) if hit.size > 0 else len(cs)   # 保留到该项（含）
        keep_idx_stage1 = idx_sorted[:pos]
        drop_idx_stage1 = idx_sorted[pos:]
        # 阶段1之外全部置0（仅作用于非NaN行）
        out.loc[drop_idx_stage1, c] = 0

        # ---------- 阶段2：在“已保留集合”里再取Top-N（可选） ----------
        if c in post_map:
            N = max(0, int(post_map[c]))
            if N == 0:
                # 阶段1保留的也全部清0
                out.loc[keep_idx_stage1, c] = 0
                continue

            # 在阶段1保留集合内按值降序再排一次
            sub_vals = out.loc[keep_idx_stage1, c].sort_values(ascending=False, kind="mergesort")

            if include_ties and N < len(sub_vals):
                # 包含第N名的并列
                nth = sub_vals.iloc[N-1]
                keep_idx_stage2 = sub_vals.index[sub_vals >= nth]
            else:
                keep_idx_stage2 = sub_vals.index[:min(N, len(sub_vals))]

            # 阶段1保留但未进入Top-N的 -> 置0
            drop_idx_stage2 = np.setdiff1d(keep_idx_stage1.to_numpy(), keep_idx_stage2.to_numpy(), assume_unique=False)
            out.loc[drop_idx_stage2, c] = 0

    return out
def _transform_unit_interval(x: pd.Series, mode: str, **kw) -> pd.Series:
    """
    对 [0,1] 区间做单调增强，仅用于后续筛选/归一化。NaN 原位保留。
    mode: "power" | "softmax" | "sigmoid"
    """
    x = x.astype(float).clip(0.0, 1.0)
    if mode == "power":
        p = float(kw.get("p", 2.0))
        y = np.power(x, p)
        return pd.Series(y, index=x.index)

    if mode == "softmax":
        tau = float(kw.get("tau", 0.7))
        z = (x - np.nanmax(x.values)) / max(tau, 1e-6)  # 数值稳定
        ez = np.exp(z)
        ez[np.isnan(ez)] = 0.0
        return pd.Series(ez, index=x.index)             # 归一化在外面做

    if mode == "sigmoid":
        m = float(kw.get("m", 0.5))
        s = float(kw.get("s", 8.0))
        y = 1.0 / (1.0 + np.exp(-s * (x - m)))
        return pd.Series(y, index=x.index)

    raise ValueError(f"Unknown mode: {mode}")



def keep_until_pct_transformed(
    df: pd.DataFrame,
    cols=None,
    # —— 先整体变换（可选）——
    enable_transform: bool = True,
    transform: str = "power",          # "power" | "softmax" | "sigmoid" | "none"
    transform_kwargs: dict | None = None,
    # —— 阶段一/二（在【变换后的矩阵】上进行）——
    threshold: float = 0.9,            # 阶段一累计阈值（相对变换后之和）
    post_top_n: int | dict | None = None,
    include_ties: bool = False,
    tie_atol: float = 0.0,
    # —— 最后可选：把【变换后】矩阵按列归一化为1 —— 
    normalize_colsum1: bool = True,
    norm_eps: float = 1e-12,
    # —— 结果控制 —— 
    mask_original: bool = False,       # True: 同步把原始 df 对应置零并一并返回
    inplace: bool = False,             # 仅对 mask_original=True 时原 df 有意义
):
    """
    流程：X(0..1) 先可选做单调增强 T -> 在 T 上做阶段1(累计阈值)与阶段2(Top-N) -> 对 T 做列和=1（可选）
    返回：
      - 默认：W（变换+筛选+可归一化后的权重矩阵，数值>=0，NaN保留）
      - 若 mask_original=True：返回 (W, 原始X按相同掩码置零后的 DataFrame)
    注意：
      - 阈值与 Top-N 的排序/累计全部在“变换后的值”上进行。
      - 若 enable_transform=False 或 transform in {"none", None}，则跳过变换，直接用原列值。
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    tkw = transform_kwargs or {}

    def _apply_transform(series: pd.Series) -> pd.Series:
        """按需把列映射到 (0,1] 或非负区间，保持单调。需要你已有的 _transform_unit_interval。"""
        if (not enable_transform) or (transform is None) or (str(transform).lower() == "none"):
            return series  # 不做变换
        return _transform_unit_interval(series, transform, **tkw)

    # 1) 先对选中列整体做变换（得到权重矩阵 W）
    W = pd.DataFrame(index=df.index)
    for c in cols:
        s = df[c]
        m = s.notna()
        w = pd.Series(np.nan, index=s.index, dtype=float)
        if m.any():
            w.loc[m] = _apply_transform(s.loc[m].astype(float))
        # 小于 0 的数（若存在）截到 0，确保非负
        w = w.where(~w.notna() | (w >= 0), 0.0)
        W[c] = w

    # 2) 阶段一、阶段二都在 W 上进行（置零作用在 W）
    X_out = df if (mask_original and inplace) else (df.copy() if mask_original else None)

    # 归一化 post_top_n 为映射
    if isinstance(post_top_n, int):
        post_map = {c: int(post_top_n) for c in cols}
    elif isinstance(post_top_n, dict):
        post_map = {c: int(v) for c, v in post_top_n.items() if c in cols}
    else:
        post_map = {}

    for c in cols:
        w = W[c]
        m = w.notna()
        if not m.any():
            continue

        vals = w[m]
        total = float(vals.sum())
        if not np.isfinite(total) or total <= 0:
            # 列全 0 或无效，直接清零并跳过后续
            W.loc[m, c] = 0.0
            if mask_original:
                X_out.loc[m, c] = 0.0
            continue

        # —— 阶段1：累计阈值（在 W 上）——
        idx_sorted = vals.sort_values(ascending=False, kind="mergesort").index
        cs = vals.loc[idx_sorted].cumsum().to_numpy()
        thr = float(threshold) * total
        hit = np.nonzero(cs >= thr)[0]
        pos = (hit[0] + 1) if hit.size > 0 else len(cs)
        keep1 = idx_sorted[:pos]
        drop1 = idx_sorted[pos:]

        # 在 W 上置零
        if len(drop1) > 0:
            W.loc[drop1, c] = 0.0
            if mask_original:
                X_out.loc[drop1, c] = 0.0

        # —— 阶段2：在 keep1 内再 Top-N（仍用 W 的值排序）——
        if c in post_map:
            N = max(0, int(post_map[c]))
            if N == 0:
                W.loc[keep1, c] = 0.0
                if mask_original:
                    X_out.loc[keep1, c] = 0.0
            else:
                sub = W.loc[keep1, c].sort_values(ascending=False, kind="mergesort")
                if include_ties and N < len(sub):
                    nth = sub.iloc[N-1]
                    if tie_atol > 0:
                        mask_keep = (sub.values > nth) | np.isclose(sub.values, nth, atol=float(tie_atol))
                        keep2 = sub.index[mask_keep]
                    else:
                        keep2 = sub.index[sub >= nth]
                else:
                    keep2 = sub.index[:min(N, len(sub))]
                drop2 = np.setdiff1d(keep1.to_numpy(), keep2.to_numpy(), assume_unique=False)
                if len(drop2) > 0:
                    W.loc[drop2, c] = 0.0
                    if mask_original:
                        X_out.loc[drop2, c] = 0.0

    # 3) 最后：对 W 做列和=1 归一化（保持 NaN；0 仍为 0）
    if normalize_colsum1:
        for c in cols:
            col = W[c]
            m = col.notna()
            if not m.any():
                continue
            ssum = float(col[m].sum())
            if np.isfinite(ssum) and ssum > norm_eps:
                W.loc[m, c] = col[m] / ssum

    return (W, X_out) if mask_original else W
 
    
    
############## niche neighborhood enrichment analysis


def _bh_fdr(pvals_1d: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals_1d, float)
    n = p.size
    order = np.argsort(p)
    ranks = np.arange(1, n+1, dtype=float)
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate((p[order] * n / ranks)[::-1])[::-1]
    return np.clip(q, 0.0, 1.0)

def niche_niche_interaction(
    adata,
    niche_col: str = "CD_clustering_7",
    k: int = 10,
    permutations: int = 1000,
    *,
    obsm_key: str = "spatial",
    halve_diagonal: bool = True,
    random_state: int | None = None,
):
    """
    计算 niche–niche 交互：
      - C_obs = D^T A D（无向图；对角可选除2）
      - permutation 打乱 niche 标签 → 期望/零分布 → p 值、BH-FDR
      - 额外返回 3 种比例分数：RowFrac、SymFrac(推荐)、GlobalFrac
    """
    rng = np.random.default_rng(random_state)

    # 1) kNN 邻接（请确保 cal_K_neighboorhood 使用 k=k）
    A, _, _, _, _, _ = cal_K_neighboorhood(adata.obsm[obsm_key], k=k)
    if not issparse(A): A = csr_matrix(A)
    A.setdiag(0); A.eliminate_zeros()

    # 2) niche one-hot
    niches = adata.obs[niche_col]
    if not pd.api.types.is_categorical_dtype(niches):
        niches = niches.astype("category")
    cats = list(niches.cat.categories)
    code = niches.cat.codes.to_numpy()
    N, K = code.size, len(cats)
    D = csr_matrix((np.ones(N, dtype=int), (np.arange(N), code)), shape=(N, K))

    # 3) 观测
    C_obs = (D.T @ A @ D).toarray().astype(float)
    if halve_diagonal:
        di = np.diag_indices(K)
        C_obs[di] /= 2.0

    # 4) 置换
    ge = np.zeros_like(C_obs, int)
    exp_sum = np.zeros_like(C_obs, float)
    for _ in tqdm(range(permutations), desc="Permuting"):
        perm = rng.permutation(code)
        Dp = csr_matrix((np.ones(N, int), (np.arange(N), perm)), shape=(N, K))
        Cp = (Dp.T @ A @ Dp).toarray().astype(float)
        if halve_diagonal:
            Cp[di] /= 2.0
        exp_sum += Cp
        ge += (Cp >= C_obs).astype(int)

    pval = (ge + 1) / (permutations + 1)    # +1 平滑
    exp = exp_sum / permutations

    # 5) FDR（对上三角做一次，镜像回去）
    iu = np.triu_indices(K, 0)
    q_upper = _bh_fdr(pval[iu].ravel())
    qval = np.full_like(pval, np.nan, float)
    qval[iu] = q_upper
    qval[(iu[1], iu[0])] = qval[iu]

    # 6) 富集强度
    eps = 1e-9
    ratio = C_obs / (exp + eps)            # obs/exp
    log2fc = np.log2(np.maximum(ratio, 1.0))

    # 7) 比例分数
    # RowFrac：每行和=1
    row_sum = C_obs.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    row_frac = C_obs / row_sum

    # SymFrac（Dice正规化）：2*C_ij / (E_i + E_j)
    E = C_obs.sum(axis=1)                  # 每个 niche 的“度”（内部边算一次）
    denom = (E[:, None] + E[None, :])
    denom[denom == 0] = 1.0
    sym_frac = (2.0 * C_obs) / denom

    # GlobalFrac：占全部边的比例（注意：矩阵总和=2*|E|；但只是比例比较可用）
    total = C_obs.sum()
    gfrac = C_obs / (total if total > 0 else 1.0)

    # 8) 打包 DataFrame
    idx = pd.Index(cats, name=niche_col)
    obs_df     = pd.DataFrame(C_obs, index=idx, columns=idx)
    exp_df     = pd.DataFrame(exp,   index=idx, columns=idx)
    pval_df    = pd.DataFrame(pval,  index=idx, columns=idx)
    qval_df    = pd.DataFrame(qval,  index=idx, columns=idx)
    ratio_df   = pd.DataFrame(ratio, index=idx, columns=idx)
    log2fc_df  = pd.DataFrame(log2fc,index=idx, columns=idx)
    rowfrac_df = pd.DataFrame(row_frac, index=idx, columns=idx)
    symfrac_df = pd.DataFrame(sym_frac, index=idx, columns=idx)
    gfrac_df   = pd.DataFrame(gfrac,    index=idx, columns=idx)

    return {
        "obs": obs_df, "exp": exp_df,
        "pval": pval_df, "qval": qval_df,
        "ratio": ratio_df, "log2fc": log2fc_df,
        "rowfrac": rowfrac_df, "symfrac": symfrac_df, "globalfrac": gfrac_df,
    }


def _convert_pval_to_asterisks(pval):
    """根据 p-value 返回相应的显著性标记"""
    if pval <= 0.005:
        return "***"
    elif pval <= 0.01:
        return "**"
    elif pval <= 0.05:
        return "*"
    return ""

def plot_symfrac_heatmap(res,
                         alpha: float = 0.05,
                         mask_diag: bool = False,
                         cmap: str = "viridis",
                         figsize=(7.5, 6.5),
                         save_fig=None,
                         order: list[str] | None = None  # ✅ 新增参数
                         ):
    """返回 fig, ax, order（order 可用于弦图或固定顺序）"""
    M = res["symfrac"].copy()
    Q = res["qval"].reindex_like(M)

    # ✅ 顺序控制：若未提供，则用层次聚类
    if order is None:
        Z = linkage(pdist(M.values, metric='correlation'), method='average')
        order = leaves_list(Z)
        order = M.index[order].tolist()
    else:
        # 校验：确保给定顺序包含所有索引
        missing = set(M.index) - set(order)
        if missing:
            raise ValueError(f"给定的 order 缺少这些索引: {missing}")

    # 重排矩阵
    M_ord = M.loc[order, order]
    Q_ord = Q.loc[order, order]

    # 对角遮罩
    if mask_diag:
        np.fill_diagonal(M_ord.values, np.nan)

    # 绘制热图
    fig, ax = plt.subplots(figsize=figsize)
    hm = sns.heatmap(
        M_ord, cmap=cmap, vmin=0, vmax=1, square=True,
        xticklabels=True, yticklabels=True,
        cbar_kws=dict(label="Symmetric interaction fraction (0–1)"),
        ax=ax
    )
    ax.set_title("Niche–niche interaction (symfrac)")

    # 添加显著性标记（使用 _convert_pval_to_asterisks 来替代原来的单星号）
    sig = (Q_ord.values < alpha)
    n = sig.shape[0]
    for i in range(n):
        for j in range(n):
            if (i == j and mask_diag) or not sig[i, j]:
                continue
            # 使用 _convert_pval_to_asterisks 来获取显著性标记
            pval = Q_ord.iloc[i, j]
            asterisk = _convert_pval_to_asterisks(pval)
            ax.text(j + 0.5, i + 0.5, asterisk,
                    ha="center", va="center",
                    color="white", fontsize=10,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

    plt.tight_layout()
    if save_fig is not None:
        plt.savefig(save_fig, dpi=300, bbox_inches='tight')
    return fig, ax, order, M_ord.index.tolist()

