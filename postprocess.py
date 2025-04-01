import torch
import torch.nn.functional as F
import math
import numpy as np


_SHARP_LOOP_DIST_THRESHOLD = 4


def _generate_sharp_loop_mask(seq_len):
    mask = np.eye(seq_len, k=0, dtype=bool)
    for i in range(1, _SHARP_LOOP_DIST_THRESHOLD):
        mask = mask + np.eye(seq_len, k=i, dtype=bool) + np.eye(seq_len, k=-i, dtype=bool)

    return mask


CANONICAL_PAIRS = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']


def _generate_canonical_pairs_mask(seq: str):
    seq = seq.replace('T', 'U')

    mask = np.zeros((len(seq), len(seq)), dtype=bool)
    for i, nt_i in enumerate(seq):
        for j, nt_j in enumerate(seq):
            if f'{nt_i}{nt_j}' in CANONICAL_PAIRS:
                mask[i, j] = True

    return mask


def _clean_sec_struct(sec_struct: np.array, probs: np.array):
    clean_sec_struct = np.copy(sec_struct)
    tmp_probs = np.copy(probs)
    tmp_probs[sec_struct < 1] = 0.0

    while np.sum(tmp_probs > 0.0) > 0:
        i, j = np.unravel_index(np.argmax(tmp_probs, axis=None), tmp_probs.shape)

        tmp_probs[i, :] = tmp_probs[j, :] = 0.0
        clean_sec_struct[i, :] = clean_sec_struct[j, :] = 0

        tmp_probs[:, i] = tmp_probs[:, j] = 0.0
        clean_sec_struct[:, i] = clean_sec_struct[:, j] = 0

        clean_sec_struct[i, j] = clean_sec_struct[j, i] = 1

    return clean_sec_struct


def prob_mat_to_sec_struct(probs: np.array, seq: str, threshold: float = 0.5, allow_nc_pairs: bool = False,
                           allow_sharp_loops: bool = False):
    assert np.all(np.isclose(probs, np.transpose(probs))), "Probability matrix must be symmetric!"
    seq_len = probs.shape[-1]

    allowed_pairs_mask = np.logical_not(np.eye(seq_len, dtype=bool))

    if not allow_sharp_loops:
        # Prevent pairings that would cause sharp loops
        allowed_pairs_mask = np.logical_and(allowed_pairs_mask, ~_generate_sharp_loop_mask(seq_len))

    if not allow_nc_pairs:
        # Prevent non-canonical pairings
        allowed_pairs_mask = np.logical_and(allowed_pairs_mask, _generate_canonical_pairs_mask(seq))

    probs[~allowed_pairs_mask] = 0.0

    sec_struct = np.greater(probs, threshold).astype(int)
    sec_struct = _clean_sec_struct(sec_struct, probs)

    return sec_struct
