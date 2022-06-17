import argparse
import matplotlib.pyplot as plt
import random
import pandas as pd
import pickle
import numpy as np
from joblib import Parallel, delayed
import heapq
from scipy import correlate
from numba import vectorize, cuda
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

# User-specified prediction function

# predict_and_save_results_mstgcn - under lib/utils.py

'''
:param: low_bound -
:param: up_bound -
:param: val -
:return: bool -
'''
def within_range(low_bound, up_bound, val):
    return low_bound <= val and val <= up_bound

'''
:requires: Arrays first_portion and second_portion
should be .
:param: first_portion -
:param: second_portion -
:param: data_vec -
:return: output_vec -
'''
def take_diff_vecs(first_portion,second_portion,data_vec,jth):
    vec_shape_len = len(data_vec.shape)
    #prev_vec = data_vec[:first_portion[0],0] if vec_shape_len == 2\
    #                                         else data_vec[:first_portion[0]]
    between_first_and_second = data_vec[:second_portion[0],0] if vec_shape_len == 2\
            else data_vec[:second_portion[0]]
    # vector after the second portion.
    post_vec = data_vec[(second_portion[-1]+1):, 0] if vec_shape_len == 2\
               else data_vec[(second_portion[-1]+1):]
    # first portion replacing second portion.
    if (len(first_portion) > len(second_portion)):
        new_first_indices = first_portion[:len(second_portion)]
    elif (len(first_portion) < len(second_portion)):
        curr_len = 0
        new_first_indices = []
        while (curr_len+len(first_portion) < len(second_portion)):
            new_first_indices += first_portion
            curr_len += len(first_portion)
        new_first_indices += first_portion[:(len(second_portion)-curr_len)]
    else:
        new_first_indices = second_portion[:jth] + first_portion[jth:]
    replaced_second_portion = np.take(data_vec[:,0] if vec_shape_len == 2 else data_vec, new_first_indices, axis=0)
    output_vec = np.append(np.append(between_first_and_second,replaced_second_portion),post_vec)
    return output_vec

'''
:param: first_portion - first portion of the traffic data that's being resampled.
:param: second_portion -
:param: indep_vec -
:param: dep_vec -
:param: jth -
:return: output_indep, output_dep -
'''

def perturb_x_y(first_portion, second_portion, indep_vec, dep_vec, jth):
    output_indep = take_diff_vecs(first_portion, second_portion, indep_vec, jth)
    output_dep = take_diff_vecs(first_portion, second_portion, dep_vec, jth)
    return output_indep, output_dep

'''
:brief: visualize_resampled_portions finds similar .
:param: vec -
:param: first_portion -
:param: second_portion -
:param: plot_name -
'''

def visualize_resampled_portions(vec, first_portion, second_portion, plot_name):
    plt.plot(list(range(0,first_portion[0])), vec[:first_portion[0]], color="b")
    plt.plot(first_portion,list(np.take(vec, first_portion, axis=0)),color="g")
    plt.plot(list(range(first_portion[-1]+1,second_portion[0])), vec[(first_portion[-1]+1):second_portion[0]], color="b")
    plt.plot(second_portion, list(np.take(vec, second_portion, axis=0)), color="g")
    plt.plot(list(range(second_portion[-1]+1,len(vec))), vec[(second_portion[-1]+1):], color="b")
    plt.savefig(f'./{plot_name}.png')
    plt.close()

'''
:param: indep_vec
:param: dep_vec -
:param: jth -
:param: observations_to_perturb -
:return: output_indep, output_dep - 2-tuple of

'''
def mc_sample_strumbelj(indep_vec, dep_vec, jth, observations_to_perturb):
    blocks = list(range(len(observations_to_perturb)))

    output_vec = []
    first, second = np.random.choice(blocks, size = 2, replace=False)
    assert len(blocks) >= 3, "not enough blocks"
    assert (first < second) or (first > second), "Overlapping intervals."
    assert first != second, "indices same when not supposed to."
    first_portion = observations_to_perturb[first]
    second_portion = observations_to_perturb[second]

    if (first < second):
        #visualize_resampled_portions(dep_vec, first_portion, second_portion)
        output_indep, output_dep = perturb_x_y(first_portion,
                                        second_portion,
                                        indep_vec,
                                        dep_vec,
                                        jth)
    elif (first > second):
        #visualize_resampled_portions(dep_vec, second_portion, first_portion)
        output_indep, output_dep = perturb_x_y(second_portion,
                                               first_portion,
                                               indep_vec,
                                               dep_vec,
                                               jth)
    return output_indep, output_dep

'''
:param: data_vec
:param: jth
:param: ref_portion
:param: sample_type
:return: mc_sample_strumbelj return value -
'''
def sample_similar_segments(i, input_tensor, dep_tensor, jth, ref_portion, sample_type, top_k_similar=15):
    all_and_11_points_input = input_tensor[:,i,0:1,0]
    for idx in range(1,12):
        all_and_11_points_input = np.append(all_and_11_points_input, input_tensor[-1,i,0:1,idx])
    all_and_11_points_dep = dep_tensor[:,i,0]
    for idx in range(1,12):
        all_and_11_points_dep = np.append(all_and_11_points_dep, dep_tensor[-1,i,idx])

    if (sample_type is not None and len(sample_type) != 0):
        sim_segs = most_similar_segments(ref_portion[0],ref_portion[1],
                                         all_and_11_points_input, top_k_similar)
        test_x_vec = all_and_11_points_input
        test_y_vec = all_and_11_points_dep
        return mc_sample_strumbelj(test_x_vec, test_y_vec, jth, sim_segs)
    else:
        print("No specified sample type.")
        return []


'''
:brief:
:param: i -
:param: input_tensor -
'''
def twelve_shifted_resampled(i, input_tensor, dep_tensor, jth, ref_portion, sample_type, top_k=15):
    #print("Input tensor len: ", len(input_tensor))
    output = sample_similar_segments(i, input_tensor, dep_tensor, jth, ref_portion, sample_type, top_k)
    if len(output) == 2:
        res_input = []
        res_output = []
        for i in range(12):
            res_input.append(np.array(output[0][i:(len(output[0])-(11-i) )]))
            res_output.append(np.array(output[1][i:(len(output[1])-(11-i) )]))
        return res_input,res_output
    else:
        print("No specified sample type for sample_similar_segments")
        return []

'''
:brief: the function applies perturbations to the PeMS04 and PeMS08 datasets.
:param: input_tensor - .
:param: dep_tensor - .
:param: region_to_perturb - we can include up to input_tensor.shape[0]
(this excludes the 11 points following because they).
:param: data_type - .
:param: dim - .
:param: jth - .
:param: sample_type - .
:return: perturbed_indep, perturbed_dep - .
'''

def perturb_func(input_tensor, dep_tensor, region_to_perturb, data_type, dim, jth, sample_type):
    if (dim > 4):
        print("Dimensions greater than 4 not handled\n")
    else:
        result=[twelve_shifted_resampled(i, input_tensor, dep_tensor, jth, region_to_perturb, sample_type, 15) \
                for i in range(input_tensor.shape[1])]
    perturbed_indep = flatten_lst(list(map(lambda x: x[0], result)))
    perturbed_dep = flatten_lst(list(map(lambda x: x[1], result)))
    np_indep = np.array(np.concatenate(perturbed_indep,axis=None), dtype=np.float64)
    perturbed_indep = np.reshape(np_indep, input_tensor.shape)
    perturbed_dep = np.reshape(np.array(np.concatenate(perturbed_dep,axis=None), dtype=np.float64), dep_tensor.shape)
    return perturbed_indep, perturbed_dep

## A414 ##

# predicting next point given past 64 points
def produce_x_y(n_trained, n_tested, vec):
    x, y = [],[]
    for i in range(n_trained, len(vec)-n_tested+1):
        x.append(np.array(vec)[i-n_trained:i])
        y.append(np.array(vec)[i:i+n_tested])
    return np.array(x), np.array(y)

def substitute_out_second(first_portion, second_portion, vec, jth):
    up_to_second = vec[:second_portion[0]]
    # vector after the second portion.
    post_vec = vec[(second_portion[-1]+1):]
    # first portion replacing second portion.
    if (len(first_portion) > len(second_portion)):
        new_second_indices = first_portion[:len(second_portion)]
    elif (len(first_portion) < len(second_portion)):
        curr_len = 0
        new_second_indices = []
        while (curr_len+len(first_portion) < len(second_portion)):
            new_second_indices += first_portion
            curr_len += len(first_portion)
        new_second_indices += first_portion[:(len(second_portion)-curr_len)]
    else:
        new_second_indices = second_portion[:jth] + first_portion[jth:]
    replaced_second_portion = np.take(vec, new_second_indices, axis=0)
    output_vec = np.append(np.append(up_to_second,replaced_second_portion),
                                     post_vec)
    return output_vec

def substitute_with_first(portions, vec, jth):
    first_portion_indices = portions[0]
    up_to_first = list(vec[:portions[0][0]])
    # print("Original length of vec: ",len(list(vec)))
    # print(portions)
    substituted_portions = up_to_first + list(np.take(vec,first_portion_indices,axis=0))
    # vector after the second portion.
    for i,portion in enumerate(portions[1:]):
        substituted_portions += list(vec[(portions[i][-1]+1):portions[i+1][0]])
        # print("Substituted Portions:",len(list(vec[(portions[i][-1]+1):portions[i+1][0]])),
        #       portions[i+1][0]-(portions[i][-1]+1))
        if (len(first_portion_indices) > len(portion)):
            new_indices = first_portion_indices[:len(portion)]
            # print("Portion smaller than ref",len(new_indices), len(portion))
        elif (len(first_portion_indices) < len(portion)):
            curr_len = 0
            new_indices = []
            while (curr_len+len(first_portion_indices) < len(portion)):
                new_indices += first_portion_indices
                curr_len += len(first_portion_indices)
            new_indices += first_portion_indices[:(len(portion)-curr_len)]
            # print("Portion larger than ref",len(new_indices), len(portion))
        else:
            new_indices = portion[:jth] + first_portion_indices[jth:]
            # print("Portion same as ref",len(new_indices), len(portion),jth)
        replaced_portion = np.take(vec, new_indices, axis=0)
        substituted_portions += list(replaced_portion)
        # print(len(substituted_portions),len(portion),"and",len(list(replaced_portion)),i)
    substituted_portions += list((vec[(portions[-1][-1]+1):vec.shape[0]]))
    # print("Last Add On:",len(list((vec[(portions[-1][-1]+1):vec.shape[0]]))))
    # print("Length of substituted portions: ",len(substituted_portions))
    return np.array(substituted_portions)
    # first portion replacing second portion.
    # if (len(first_portion) > len(second_portion)):
    #     new_second_indices = first_portion[:len(second_portion)]
    # elif (len(first_portion) < len(second_portion)):
    #     curr_len = 0
    #     new_second_indices = []
    #     while (curr_len+len(first_portion) < len(second_portion)):
    #         new_second_indices += first_portion
    #         curr_len += len(first_portion)
    #     new_second_indices += first_portion[:(len(second_portion)-curr_len)]
    # else:
    #     new_second_indices = second_portion[:jth] + first_portion[jth:]
    # replaced_second_portion = np.take(vec, new_second_indices, axis=0)
    # output_vec = np.append(np.append(up_to_second,replaced_second_portion),
    #                                  post_vec)
    # return output_vec

def mc_sample_strumbelj_a414(vec, jth, observations_to_perturb):
    blocks = list(range(len(observations_to_perturb)))

    first, second, third, fourth = np.random.choice(blocks,
                                                    size = 4,
                                                    replace=False)
    # assert len(blocks) >= 3, "not enough blocks"
    # assert (first < second) or (first > second), "Overlapping intervals."
    # assert first != second, "indices same when not supposed to."
    first_portion = observations_to_perturb[first]
    second_portion = observations_to_perturb[second]
    third_portion = observations_to_perturb[third]
    fourth_portion = observations_to_perturb[fourth]
    ordered_perturbations = sorted([first_portion, second_portion,
                                    third_portion, fourth_portion],
                                    key = lambda x: x[0])

    return substitute_with_first(ordered_perturbations, vec, jth)

def sample_similar_segments_a414(input_tensor, dep_tensor, jth, ref_portion, sample_type, top_k_similar=15):
    all_and_n_points_input = input_tensor[:,0,2]
    for idx in range(1,64):
        all_and_n_points_input = np.append(all_and_n_points_input, input_tensor[-1,idx,2])
    all_and_n_points_input = np.append(all_and_n_points_input, dep_tensor[-1])
    # add on last 64 entries
    # print("Shape:",all_and_n_points_input.shape)
    if (sample_type is not None and len(sample_type) != 0):
        sim_segs = most_similar_segments(ref_portion[0], ref_portion[1],
                                         all_and_n_points_input, top_k_similar)
        return mc_sample_strumbelj_a414(all_and_n_points_input, jth, sim_segs)
    else:
        print("No specified sample type.")
        return []

def n_shifted_resampled(input_tensor, dep_tensor, jth, ref_portion, sample_type, top_k=15):
    #print("Input tensor len: ", len(input_tensor))
    output = sample_similar_segments_a414(input_tensor, dep_tensor, jth, ref_portion, sample_type, top_k)
    return produce_x_y(64, 1, output)

def perturb_func_a414(input_tensor, dep_tensor, region_to_perturb, data_type, dim, jth, sample_type):
    new_input, new_output = n_shifted_resampled(input_tensor, dep_tensor, jth, region_to_perturb, sample_type, 15)
    return new_input, new_output
# We modify the shape of the data after removal of portions of the data.
# Then save the modified data into the same file.

# Rather than perturbing the input data file for predictions
# we can perturb the input tensor of time series observations.
'''
:param: input_tensor
:param: data_type
:param: dim - the dimension of the independent "input" tensor.
:param: **kwargs - key word arguments that can either be data_entry or
:return: result - .
'''
def perturb_data(input_tensor, dep_tensor, region_to_perturb, data_type, dim, M_iterations, jth, sample_type="mc_sample_strumbelj",is_a414=True):
    perturbed_indep, perturbed_dep = (perturb_func_a414(input_tensor,
                                                  dep_tensor,
                                                  region_to_perturb,
                                                  data_type,
                                                  dim,
                                                  jth,
                                                  sample_type) if is_a414\
                                                 else perturb_func(input_tensor,
                                                                   dep_tensor,
                                                                   region_to_perturb,
                                                                   data_type,
                                                                   dim,
                                                                   jth,
                                                                   sample_type))

    return perturbed_indep, perturbed_dep

# #####################################
# Noise Reduction
# #####################################
# Preprocessing on time-series data input
# #####################################

def average(L):
    if len(L) == 0:
        return 0
    return sum(L)/len(L)
def add_on_reciprocal(diff):
    if diff == 0:
        add_on = float('inf')
    else:
        add_on = 1/diff
    return add_on
def linear_smoother(ts, num_neighbors, weight_method="gaussian"):
    averaged_points = []
    # every 20 points
    points_to_include = []
    for i in range(len(ts)):
        closest = []
        heapq.heapify(closest)
        downward, upward = i, i
        if weight_method == "gaussian":
            under_neighbors = i-num_neighbors
            above_neighbors = i+num_neighbors
            while under_neighbors <= downward and under_neighbors >= 0 and\
                  i+num_neighbors >= upward and above_neighbors < len(ts):
                # sorts by the first element of tuple
                if len(closest) >= num_neighbors:
                    heapq.heappop(closest)
                else:
                    heapq.heappush(closest,(add_on_reciprocal(i-downward),ts[downward]))
                if (len(closest) >= num_neighbors):
                    heapq.heappop(closest)
                else:
                    heapq.heappush(closest,(add_on_reciprocal(upward-i), ts[upward]))

                downward -= 1
                upward += 1
            while under_neighbors <= downward and under_neighbors >= 0:
                if len(closest) >= num_neighbors:
                    heapq.heappop(closest)
                else:
                    heapq.heappush(closest,(add_on_reciprocal(i-downward),ts[downward]))
                downward -= 1
            while i+num_neighbors >= upward and above_neighbors < len(ts):
                if len(closest) >= num_neighbors:
                    heapq.heappop(closest)
                else:
                    heapq.heappush(closest,(add_on_reciprocal(upward-i), ts[upward]))
                upward += 1
            heap_items = list(map(lambda x: x[1], [heapq.heappop(closest) \
                                                       for i in range(len(closest))]))
            avg = average(heap_items)
            averaged_points.append(avg)
            # we collect twice as many points as total/num_neighbors
            if (i%(num_neighbors//2) == 0):
                points_to_include.append(avg)
        else:
            # rather than a weighted average of nearest data points,
            # we take ordinary average.
            pass
    return averaged_points, points_to_include

'''
:param: shape
:param: xlab
:param: ylab
:param: plt_title
'''

# Data types
def plot(shape,xlab,ylab,plt_title):
    avg_speed_pems4 = plt.plot(list(range(len(arr['data'][:,0]))),
                                      arr['data'][:,detector,2], color="blue")
    pems4_speed_title = plt.title(plt_title)
    pems4_speed_x = plt.xlabel(xlab)
    pems4_speed_y = plt.ylabel(ylab)


'''
:param: ts
:param: window
:return: ts_points
'''

def ma(ts,window):
    ts_points = []
    for i in range(window,len(ts)):
        ts_points.append(np.sum(ts[(i-window):i])/window)
    return ts_points

'''
:param: vec
:return: np.float64 - norm of the input vector
"vec" is computed.
'''

def norm(vec):
    return np.sqrt(np.sum([x**2 for x in vec]))

'''
:param: vec1
:param: vec2
:return: cosine similarity between the two vectors
vec1, vec2. Using cosine similarity rather than
dot product for similarity due to cosine similarity
not being dependent on vector length.
'''
def cosine_sim(vec1,vec2):
    return np.dot(vec1,vec2)/(norm(vec1)*norm(vec2))


'''
:param: vec -
:return: diffs -
'''
def deltas(vec):
    diffs = []
    for i in range(1,len(vec)):
        diffs.append(vec[i]-vec[i-1])
    return diffs



'''
:brief: Flattens a list down to two levels.

'''

def flatten_lst(L):
    return [elem for lst in L for elem in lst]

'''
Computes the similarity between pairwise difference vectors (x_i-x_{i-1})
:param: start_idx
:param: end_idx
:param: ts
:param: k
:param: compute_diff
:param: avg_diffs

:return: list of lists - each entry is a list of length (end_idx-start_idx) is
one of k segments of the time series data "ts" most similar to
ts[start_idx:end_idx].
'''

def most_similar_segments(start_idx,end_idx,ts,k,compute_diff=False,avg_diff=False):
    if (len(ts[start_idx:end_idx].shape) < 2):
        cmp_window = ts[start_idx:end_idx]
    else:
        cmp_window = flatten_lst(ts[start_idx:end_idx])
        ts = flatten_lst(ts)
    if compute_diff:
        cmp_window = deltas(cmp_window)
    window_size = end_idx-start_idx
    L = []
    heapq.heapify(L)
    num_segments = 0
    seen_segments = []
    sim_to_region = dict()
    lower_upper_bound, upper_lower_bound = start_idx, end_idx
    intervals = []
    curr_size = window_size
    while (lower_upper_bound >= curr_size):
        intervals.append((lower_upper_bound-curr_size,lower_upper_bound))
        lower_upper_bound -= curr_size
        curr_size = np.random.geometric(p=(1/window_size))
        while (curr_size < 11):
            curr_size = np.random.geometric(p=(1/window_size))
    if (lower_upper_bound >= 11):
        intervals.append((0,lower_upper_bound))
    curr_size = window_size
    while (upper_lower_bound+curr_size < len(ts)):
        intervals.append((upper_lower_bound, upper_lower_bound+curr_size))
        upper_lower_bound += curr_size
        curr_size = np.random.geometric(p=(1/window_size))
        while (curr_size < 11):
            curr_size = np.random.geometric(p=(1/window_size))
    if (len(ts)-upper_lower_bound >= 11):
        intervals.append((upper_lower_bound,len(ts)))
    # ..,lower_upper_bound][][upper_lower_bound...
    # print(cmp_window,start_idx,end_idx,ts,len(ts))
    for interval_window in intervals: # we jump by window_size
        # print("bounds:", interval_window[0],interval_window[1])
        segment = ts[interval_window[0]:interval_window[1]]
        if compute_diff:
            segment = deltas(segment)
        if avg_diff:
            assert len(cmp_window) == len(segment), "segment of interest compared to cmp_window"
            sim = np.sum(np.array(cmp_window)-np.array(segment))/len(cmp_window)
        else:
            sim = correlate(cmp_window,segment)[0]
        sim = round(sim,16)
        if len(seen_segments) != 0:
            if seen_segments[-1][-1] >= i-window_size:
                continue
        if num_segments < k:
            # we don't want to add duplicates
            if (sim not in sim_to_region):
                assert sim not in L, "sim should not be in L or sim_to_region"
                sim_to_region[sim] = list(range(interval_window[0],interval_window[1]))
                heapq.heappush(L, sim)
                num_segments += 1
        else:
            least_sim_largest_dist = heapq.nsmallest(1,L)[0]
            if sim > least_sim_largest_dist and sim not in L: # number of segments
                # need to remove the least similar region if similarity is greater than
                # the least similar one in the heap.
                del sim_to_region[least_sim_largest_dist]
                val = heapq.heappop(L)
                assert least_sim_largest_dist == val, "least similar not same as val"
                sim_to_region[sim] = list(range(interval_window[0],interval_window[1]))
                heapq.heappush(L,sim)
                # the two lines below are part of eliminating the least similar
                # here we pop the smallest element from L.
    return list(sim_to_region.values())


'''
:brief:
:param: indices -
:param: ts -
'''

def plot_similar_cyclic_regions(indices, ts):
    indices = sorted(indices, key=lambda x: x[0])
    lsts_idx_excluded = []
    curr_end = None
    plt.figure(figsize=(3, 3))
    # we assume non-overlapping indices
    for i,index_lst in enumerate(indices):
        plt.plot(index_lst,ts[index_lst],color="purple")
        if i == len(indices)-1:
            indices_to_end = list(range(index_lst[-1],len(ts)))
            plt.plot(indices_to_end,
                     ts[indices_to_end],
                    color="cyan")
        else:
            indices_in_between = list(range(index_lst[-1],indices[i+1][0]))
            plt.plot(indices_in_between,
                     ts[indices_in_between],
                    color="cyan")
        if i == 0 and index_lst[0] > 0:
            plt.plot(list(range(0,index_lst[0])),ts[:index_lst[0]],color="green")
        plt.figure(figsize=(3, 3))
    plt.show()

'''
:brief: finds the previous interval that's not overlapping with
the jth one.
:param: intervals
:param: j
:return: position of the non-overlapping
interval in the list of intervals, used to index into OPT.
'''
def are_overlapping(interval_1,interval_2):
    return interval_2[0] <= interval_1[1] and interval_2[1] >= interval_1[0]
def get_prec(intervals, j):
    prec = None
    curr_idx = j-1
    while curr_idx >= 0:
        curr_interval = intervals[curr_idx]
        if (not are_overlapping(curr_interval, intervals[j])):
            prec = curr_idx
            break
        curr_idx -= 1
    return prec
'''
OPT[j] = max{OPT[prec(j)]+1,OPT[j-1]}
:brief: max_num_intervals
:param: intervals
'''
def max_num_intervals(intervals):
    intervals.sort()
    OPT = [0]*len(intervals)
    for j in range(len(intervals)):
        prec_idx = get_prec(intervals, j)
        if prec_idx is None:
            OPT[j] = (1,[intervals[j]])
        else:
            OPT[j] = ((OPT[prec_idx][0]+1), OPT[prec_idx][1]+[intervals[j]]) \
                     if (OPT[prec_idx][0]+1) > OPT[j-1][0]\
                     else OPT[j-1]
    return OPT[-1]

'''
:brief: modify_seqs is a helper function called in get_consecutive_diffs
that finds the cumulative number of observations that are consecutively.
:param: prev -
:param: diffs -
:param: i -
:param: curr_seq -
'''
def modify_seqs(prev, diffs, i, curr_seq, cons_seq, is_positive=True):
    if prev is None:
        prev = diffs[i]

    if ((is_positive and prev > 0 and diffs[i] > 0) or\
        ((not is_positive) and prev < 0 and diffs[i] < 0)):
        curr_seq += 1
        if (curr_seq > 1):
            cons_seq += 1
            prev = diffs[i]
    else:
        curr_seq = 0
        prev = None
    return prev, curr_seq, cons_seq

'''
:param: threshold
:param: diffs
'''
def get_consecutive_diffs(threshold, diffs,look_at_positive_diffs=True):
    cons_seq_pos, curr_seq_pos = 0, 0
    cons_seq_neg, curr_seq_neg = 0, 0
    prev_pos, prev_neg = None, None
    for i in range(len(diffs)):
        if (look_at_positive_diffs):
            prev_pos, curr_seq_pos, cons_seq_pos = modify_seqs(prev_pos,
                                                  diffs,
                                                  i,
                                                  curr_seq_pos,
                                                  cons_seq_pos)
        else:
            prev_neg, curr_seq_neg, cons_seq_neg = modify_seqs(prev_neg,
                                                  diffs,
                                                  i,
                                                  curr_seq_neg,
                                                  cons_seq_neg,
                                                  is_positive=False)
    # we add 1 to account for initial consecutive sequence of 2
    # elements where we exclude the first element.
    if (look_at_positive_diffs):
        return ((cons_seq_pos+1)/len(diffs)) >= threshold
    return ((cons_seq_neg+1)/len(diffs)) >= threshold

'''
:brief: find_max_min_changes -
:param: vec -
:param: length -
:param: k -
:param: threshold - the minimum number of consecutive
:param: neighbors - the number of neighbors to average
when computing the smoothed points that are used in finding the regions with
highest and lowest rates of change (meant to denoise).
'''

# basic algorithm from thesis writeup.
def find_max_min_changes(vec, length, k, threshold=0.6, neighbors=15, is_increasing=True, is_smallest=False):
    optim_changes = []
    heapq.heapify(optim_changes)
    for i in range(length, len(vec)):
        diffs = deltas(np.array(vec[(i-length):i]))
        smoothed_points = linear_smoother(diffs, neighbors)[1]
        if get_consecutive_diffs(threshold, smoothed_points,is_increasing):
            if is_increasing:
                diffs = list(filter(lambda x: x >= 0, diffs))
            else:
                diffs = list(filter(lambda x: x < 0, diffs))
            euclidean_norm = (-1 if is_smallest else 1)*np.sqrt(np.sum(np.square(np.array(diffs))))
            if len(optim_changes) < k:
                heapq.heappush(optim_changes, [i-length, i, euclidean_norm])
            else:
                smallest_change = heapq.nsmallest(1,optim_changes)[0][2]
                if (euclidean_norm > smallest_change):
                    low_val = heapq.heappop(optim_changes)
                    heapq.heappush(optim_changes,[i-length, i, euclidean_norm])
    optim_intervals = list(map(lambda l: l[:2], optim_changes))
    optim_intervals = max_num_intervals(optim_intervals)
    return optim_intervals


'''
:param: vec -
:param: length -
:return: change_start -
'''

def optim_change(vec,length,greatest=True):
    change_start = 0
    if (greatest):
        change = 0
    else:
        change = float('inf')
    for i in range(length, len(vec)):
        diffs = deltas(np.array(vec[(i-length):i]))
        smoothed_points = linear_smoother(diffs, 15)[1]
        euclidean_norm = np.sqrt(np.sum(np.square(np.array(diffs))))
        if (get_consecutive_diffs(0.7, smoothed_points)):
            if (greatest):
                if (euclidean_norm > change):
                    change = euclidean_norm
                    change_start = i-length
            else:
                if (euclidean_norm < change):
                    change = euclidean_norm
                    change_start = i-length

    return change_start
