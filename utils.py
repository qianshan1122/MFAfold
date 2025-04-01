import torch
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score


def bp2matrix(L, base_pairs):
    """
    :param L: 序列长
    :param base_pairs: 配对
    :return: 配对矩阵
    """

    matrix = torch.zeros((L, L))
    for bp in base_pairs:
        # base pairs are 1-based
        matrix[bp[0] - 1, bp[1] - 1] = 1
        matrix[bp[1] - 1, bp[0] - 1] = 1
    return matrix


NT_DICT = {
    "R": ["G", "A"],
    "Y": ["C", "U"],
    "K": ["G", "U"],
    "M": ["A", "C"],
    "S": ["G", "C"],
    "W": ["A", "U"],
    "B": ["G", "U", "C"],
    "D": ["G", "A", "U"],
    "H": ["A", "C", "U"],
    "V": ["G", "C", "A"],
    "N": ["G", "A", "C", "U"],
}
VOCABULARY = ["A", "C", "G", "U"]


#
def seq2emb(seq, l, pad_token="-"):
    """One-hot representation of seq nt in vocabulary.  Emb is CxL
    Other nt are mapped as shared activations.
    """
    seq = seq.upper().replace("T", "U")  # convert to RNA
    emb_size = len(VOCABULARY)
    emb = torch.zeros((l, emb_size), dtype=torch.float)

    for k, nt in enumerate(seq):
        if nt == pad_token:
            continue
        if nt in VOCABULARY:
            emb[k, VOCABULARY.index(nt)] = 1
        elif nt in NT_DICT:
            v = 1 / len(NT_DICT[nt])
            ind = [VOCABULARY.index(n) for n in NT_DICT[nt]]
            emb[k, ind] = v
        else:
            raise ValueError(f"Unrecognized nucleotide {nt}")

    return emb


def pair_strength(pair):
    if "G" in pair and "C" in pair:
        return 3
    if "A" in pair and "U" in pair:
        return 2
    if "G" in pair and "U" in pair:
        return 0.8

    if pair[0] in NT_DICT and pair[1] in NT_DICT:
        n0, n1 = NT_DICT[pair[0]], NT_DICT[pair[1]]
        # Possible pairs with other bases
        if ("G" in n0 and "C" in n1) or ("C" in n0 and "G" in n1):
            return 3
        if ("A" in n0 and "U" in n1) or ("U" in n0 and "A" in n1):
            return 2
        if ("G" in n0 and "U" in n1) or ("U" in n0 and "G" in n1):
            return 0.8

    return 0


def prob_mat(seq, l_change):
    """Receive sequence and compute local conection probabilities (Ufold paper, optimized version)"""
    Kadd = 30
    window = 3
    N = len(seq)

    mat = np.zeros((l_change, l_change), dtype=np.float32)

    L = np.arange(N)
    pairs = np.array(np.meshgrid(L, L)).T.reshape(-1, 2)
    pairs = pairs[np.abs(pairs[:, 0] - pairs[:, 1]) > window, :]

    for i, j in pairs:
        coefficient = 0
        for add in range(Kadd):
            if (i - add >= 0) and (j + add < N):
                score = pair_strength((seq[i - add], seq[j + add]))
                if score == 0:
                    break
                else:
                    coefficient += score * np.exp(-0.5 * (add ** 2))
            else:
                break
        if coefficient > 0:
            for add in range(1, Kadd):
                if (i + add < N) and (j - add >= 0):
                    score = pair_strength((seq[i + add], seq[j - add]))
                    if score == 0:
                        break
                    else:
                        coefficient += score * np.exp(-0.5 * (add ** 2))
                else:
                    break

        mat[i, j] = coefficient

    return torch.tensor(mat)


def contact_f1(ref_batch, pred_batch, L, th=0.5, reduce=True, method="triangular"):
    """Compute F1 from base pairs. Input goes to sigmoid and then thresholded"""
    f1_list = []

    if type(ref_batch) == float or len(ref_batch.shape) < 3:
        ref_batch = [ref_batch]
        pred_batch = [pred_batch]
        L = [L]

    for ref, pred, l in zip(ref_batch, pred_batch, L):
        # ignore padding
        ind = torch.where(ref != -1)
        pred = pred[ind].view(l, l)
        ref = ref[ind].view(l, l)

        # pred goes from -inf to inf
        pred = torch.sigmoid(pred)
        pred[pred <= th] = 0

        if method == "triangular":
            f1 = f1_triangular(ref, pred > th)
        elif method == "shift":
            ref = mat2bp(ref)
            pred = mat2bp(pred)
            _, _, f1 = f1_shift(ref, pred)
        else:
            raise NotImplementedError

        f1_list.append(f1)

    if reduce:
        return torch.tensor(f1_list).mean().item()
    else:
        return torch.tensor(f1_list)


def f1_triangular(ref, pred):
    """Compute F1 from the upper triangular connection matrix"""
    # get upper triangular matrix without diagonal
    ind = torch.triu_indices(ref.shape[0], ref.shape[1], offset=1)

    ref = ref[ind[0], ind[1]].numpy().ravel()
    pred = pred[ind[0], ind[1]].numpy().ravel()

    return f1_score(ref, pred, zero_division=0)


def f1_shift(ref_bp, pre_bp):
    """F1 score with tolerance of 1 position"""
    # corner case when there are no positives
    if len(ref_bp) == 0 and len(pre_bp) == 0:
        return 1.0, 1.0, 1.0

    tp1 = 0
    for rbp in ref_bp:
        if (
                rbp in pre_bp
                or [rbp[0], rbp[1] - 1] in pre_bp
                or [rbp[0], rbp[1] + 1] in pre_bp
                or [rbp[0] + 1, rbp[1]] in pre_bp
                or [rbp[0] - 1, rbp[1]] in pre_bp
        ):
            tp1 = tp1 + 1
    tp2 = 0
    for pbp in pre_bp:
        if (
                pbp in ref_bp
                or [pbp[0], pbp[1] - 1] in ref_bp
                or [pbp[0], pbp[1] + 1] in ref_bp
                or [pbp[0] + 1, pbp[1]] in ref_bp
                or [pbp[0] - 1, pbp[1]] in ref_bp
        ):
            tp2 = tp2 + 1

    fn = len(ref_bp) - tp1
    fp = len(pre_bp) - tp1

    tpr = pre = f1 = 0.0
    if tp1 + fn > 0:
        tpr = tp1 / float(tp1 + fn)  # sensitivity (=recall =power)
    if tp1 + fp > 0:
        pre = tp2 / float(tp1 + fp)  # precision (=ppv)
    if tpr + pre > 0:
        f1 = 2 * pre * tpr / (pre + tpr)  # F1 score

    return tpr, pre, f1


def mat2bp(x):
    """Get base-pairs from conection matrix [N, N]. It uses upper
    triangular matrix only, without the diagonal. Positions are 1-based. """
    ind = torch.triu_indices(x.shape[0], x.shape[1], offset=1)
    pairs_ind = torch.where(x[ind[0], ind[1]] > 0)[0]

    pairs_ind = ind[:, pairs_ind].T
    # remove multiplets pairs
    multiplets = []
    for i, j in pairs_ind:
        ind = torch.where(pairs_ind[:, 1] == i)[0]
        if len(ind) > 0:
            pairs = [bp.tolist() for bp in pairs_ind[ind]] + [[i.item(), j.item()]]
            best_pair = torch.tensor([x[bp[0], bp[1]] for bp in pairs]).argmax()

            multiplets += [pairs[k] for k in range(len(pairs)) if k != best_pair]

    pairs_ind = [[bp[0] + 1, bp[1] + 1] for bp in pairs_ind.tolist() if bp not in multiplets]

    return pairs_ind


def _relax_ss(ss_mat: np.array) -> np.array:
    # Pad secondary structure (because of cyclical rolling)
    ss_mat = np.pad(ss_mat, ((1, 1), (1, 1)), mode='constant')

    # Create relaxed pairs matrix
    relax_pairs = \
        np.roll(ss_mat, shift=1, axis=-1) + np.roll(ss_mat, shift=-1, axis=-1) + \
        np.roll(ss_mat, shift=1, axis=-2) + np.roll(ss_mat, shift=-1, axis=-2)

    # Add relaxed pairs into original matrix
    relaxed_ss = ss_mat + relax_pairs

    # Ignore cyclical shift and clip values
    relaxed_ss = relaxed_ss[..., 1: -1, 1: -1]
    relaxed_ss = np.clip(relaxed_ss, 0, 1)

    return relaxed_ss


def ss_recall(target_ss: np.array, pred_ss: np.array, allow_flexible_pairings: bool = False) -> float:
    if allow_flexible_pairings:
        pred_ss = _relax_ss(pred_ss)

    seq_len = target_ss.shape[-1]
    upper_tri_idcs = np.triu_indices(seq_len, k=1)

    return recall_score(target_ss[upper_tri_idcs], pred_ss[upper_tri_idcs], zero_division=0.0)


def ss_precision(target_ss: np.array, pred_ss: np.array, allow_flexible_pairings: bool = False) -> float:
    if allow_flexible_pairings:
        target_ss = _relax_ss(target_ss)

    seq_len = target_ss.shape[-1]
    upper_tri_idcs = np.triu_indices(seq_len, k=1)
    return precision_score(target_ss[upper_tri_idcs], pred_ss[upper_tri_idcs], zero_division=0.0)


EPSILON = 1e-5


def ss_f1(target_ss: np.array, pred_ss: np.array, allow_flexible_pairings: bool = False) -> float:
    precision = ss_precision(target_ss, pred_ss, allow_flexible_pairings=allow_flexible_pairings)
    recall = ss_recall(target_ss, pred_ss, allow_flexible_pairings=allow_flexible_pairings)

    # Prevent division with 0.0
    if precision + recall < EPSILON:
        return 0.0

    return (2 * precision * recall) / (precision + recall)


