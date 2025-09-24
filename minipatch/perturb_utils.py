import numpy as np
import pandas as pd
from tqdm import tqdm
import torch


# # #trace shape [Ncity,2], the shape of old_trace is [batch_size,200,1],返回值[batch_size,200,1]
def perturb_trace(old_trace, perturbation, highlight=False):
    # print(type(old_trace))
    # print(perturbation)
    
    trace = perturbation.reshape(-1, 2)

    # 对插入位置排序
    insert_loc = np.argsort(trace[:, 0])
    adv_trace = trace[insert_loc]

    # 向量化插入操作
    insert_positions = adv_trace[:, 0].astype(int)
    insert_counts = adv_trace[:, 1].astype(int)

    # 计算总插入偏移
    total_insert = insert_counts.sum()
    if total_insert == 0:
        return old_trace

    # 计算新 trace 的 shape
    batch_size, seq_len, _ = old_trace.shape
    new_seq_len = seq_len + total_insert
    new_trace = np.zeros((batch_size, new_seq_len, 1))
    prev_pos = 0
    pos_offset = 0

    # 逐步插入
    for i in range(len(insert_positions)):
        pos = insert_positions[i]
        insert_num = insert_counts[i]
        # 复制当前 segment
        new_trace[:, prev_pos + pos_offset:pos + pos_offset, :] = old_trace[:, prev_pos:pos, :]

        # 插入元素
        if i == 0:
            insert_val = old_trace[:, pos, :]
        else:
            insert_val = new_trace[:, pos, :]

        insert_val = np.expand_dims(insert_val, 1)
        insert_val = np.tile(insert_val, (1, insert_num, 1))
        new_trace[:, pos + pos_offset: pos + pos_offset + insert_num, :] = insert_val
        prev_pos = pos
        pos_offset += insert_num

    # 复制剩余部分
    new_trace[:, prev_pos + pos_offset:, :] = old_trace[:, prev_pos:, :]
    return new_trace[:, :seq_len, :]


# def perturb_trace(traces, perturbations, highlight=False):
#     """
#     Perturb packet trace(s) according to the given perturbation(s).
#     Not support multiple traces and multiple perturbations at the same time.

#     Parameters
#     ----------
#     traces : array_like
#         A 2-D numpy array [Length x 1] for a single trace or a 3-D numpy
#         array [N x Length x 1] for N traces.
#     perturbations : array_like
#         A 1-D numpy array specifying a single perturbation or a 2-D numpy
#         array specifying multiple perturbations.
#     highlight : optional
#         Highlight perturbations by setting the absolute value to 2.
#     """
#     if type(perturbations) == list:
#         perturbations = np.array(perturbations)

#     # If this function is passed just one perturbation vector,
#     # pack it in a list to keep the computation the same
#     if perturbations.ndim < 2:
#         perturbations = np.array([perturbations])

#     if traces.ndim < 3 or traces.shape[0] == 1:
#         # Copy the trace n == len(perturbations) times so that we can 
#         # create n new perturbed traces
#         traces = np.tile(traces, [len(perturbations), 1, 1])
#     else:
#         # Copy the perturbation n == len(traces) times to 
#         # create n new perturbed traces
#         perturbations = np.tile(perturbations, [len(traces), 1])

#     # Make sure to floor the members of perturbations as int types
#     perturbations = perturbations.astype(int)

#     for trace, perturbation in zip(traces, perturbations):
#         # Split perturbation into an array of patches
#         patches = np.split(perturbation, len(perturbation)//2)
#         length = len(np.where(trace != 0)[0])

#         # Align patch positions
#         for patch in patches:
#             x_pos, n_pkt = patch
#             # Constraint 1: within the trace
#             if x_pos > length:
#                 x_pos = length
#             # Constraint 2: at burst tail with the same direction
#             if x_pos < len(trace):
#                 while trace[x_pos, 0] * n_pkt < 0:
#                     x_pos += 1
#                     if x_pos >= len(trace):
#                         break
#             if x_pos < len(trace):
#                 while trace[x_pos, 0] * n_pkt > 0:
#                     x_pos += 1
#                     if x_pos >= len(trace):
#                         break
#             patch[0] = x_pos

#         # Apply patches
#         positions = []
#         for patch in sorted(patches, key=lambda x: x[0], reverse=True):
#             x_pos, n_pkt = patch
#             direction = 1 if n_pkt > 0 else -1
#             n_pkt = abs(n_pkt)
#             # Constraint 3: at different positions
#             if x_pos in positions:
#                 continue
#             positions.append(x_pos)
#             # Constraint 1: within the trace
#             if x_pos + n_pkt >= len(trace):
#                 n_pkt = len(trace) - x_pos
#             if n_pkt == 0:
#                 continue
#             # Constraint 2: with the same direction
#             if x_pos < length:
#                 assert direction * trace[x_pos-1, 0] > 0

#             # At each trace's position x_pos, insert a patch of n packets
#             trace[x_pos+n_pkt:, 0] = trace[x_pos:-n_pkt, 0]
#             # Outgoing
#             if direction > 0:
#                 if highlight:
#                     trace[x_pos:x_pos+n_pkt, 0] = 2.
#                 else:
#                     trace[x_pos:x_pos+n_pkt, 0] = 1.
#             # Incoming
#             else:
#                 if highlight:
#                     trace[x_pos:x_pos+n_pkt, 0] = -2.
#                 else:
#                     trace[x_pos:x_pos+n_pkt, 0] = -1.

#     # Return shape: [N * Length * 1]
#     # print(traces.shape)
#     return traces


def patch_length(x):
    if type(x) is list:
        patches = np.split(np.array(x), len(x)//2)
        patch_lengths = [abs(patch[1]) for patch in patches]
        return sum(patch_lengths)
    else:
        patches = np.split(np.array(x['perturbation']), len(x['perturbation'])//2)
        patch_lengths = [abs(patch[1]) for patch in patches]
        return sum(patch_lengths) * x['num_valid']


def verify_perturb(model, traces, labels, verbose, filename):
    """
    Verify perturbations against the given model.
    """
    # Original prediction results
    traces_ = []
    labels_ = []
    true = labels.argmax(axis=-1)
    results = pd.read_json('%s.json' % filename, orient='index')
    for site_id in tqdm(results['website']) if verbose > 0 else results['website']:
        traces_.append(traces[true == site_id])
        labels_.append(labels[true == site_id])
    traces = np.vstack(traces_)
    labels = np.vstack(labels_)
    pred = model.predict(traces).argmax(axis=-1)
    valids = pred == labels.argmax(axis=-1)
    result_before = model.evaluate(traces, labels, verbose=verbose)

    # Perturb the traces
    traces_ = []
    labels_ = []
    valids_ = []
    true = labels.argmax(axis=-1)
    site_ids = np.unique(labels.argmax(axis=-1))
    for site_id in tqdm(results['website']) if verbose > 0 else results['website']:
        perturbations = results[results['website'] == site_id]['perturbation'].iloc[0]
        traces_.append(perturb_trace(traces[true == site_id], perturbations))
        labels_.append(labels[true == site_id])
        valids_.append(valids[true == site_id])
    traces = np.vstack(traces_)
    labels = np.vstack(labels_)
    valids = np.hstack(valids_)
    results['num_valid'] = [sum(x) for x in valids_]
    lengths = []
    for site in range(len(site_ids)):
        trace_ids = np.where(valids_[site])[0]
        lengths.append([len(np.argwhere(traces_[site][i] != 0)) for i in trace_ids])
    results['lengths'] = lengths
    num_patch = results['perturbation'].apply(lambda x: len(np.split(np.array(x), len(x)//2))).mean()
    len_patch = (results['perturbation'].apply(patch_length) / num_patch).mean()
    len_perturb = sum(results.apply(patch_length, axis=1))
    len_origin = sum(results['lengths'].apply(sum))

    # Perturbed prediction results
    pred_after = model.predict(traces).argmax(axis=-1)
    stubborns = pred_after == labels.argmax(axis=-1)
    result_after = model.evaluate(traces, labels, verbose=verbose)
    
    
    
    print('Test loss: %.3f -> %.3f' % (
        result_before[0], result_after[0]))
    print('Accuracy: %.2f%% (%d/%d) -> %.2f%% (%d/%d)' % (
        float(result_before[1] * 100), sum(valids), len(labels),
        float(result_after[1] * 100), sum(stubborns), len(labels)))
    print('Success rate: %.2f%% (%d/%d)' % (
        100 * sum((valids) & (~stubborns)) / sum(valids),
        sum((valids) & (~stubborns)), sum(valids)))
    print('Patch Count/Length: %.2f/%.2f' % (
        num_patch, len_patch))
    print('Bandwidth Overhead: %.2f%% (%d/%d)' % (
        100 * len_perturb / len_origin,
        len_perturb / sum(valids), len_origin / sum(valids)))
