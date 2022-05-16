import argparse
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
from joblib import Parallel, delayed
import heapq
from scipy import correlate
from numba import vectorize, cuda

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
    prev_vec = data_vec[:first_portion[0],0] if vec_shape_len == 2\
                                             else data_vec[:first_portion[0]]
    post_vec = data_vec[(first_portion[-1]+1):, 0] if vec_shape_len == 2 else data_vec[(first_portion[-1]+1):]
    print(len(first_portion),len(second_portion),jth)
    new_first_indices = first_portion[:jth] + second_portion[jth:]
    first_portion = np.take(data_vec[:,0] if vec_shape_len == 2 else data_vec, new_first_indices, axis=0)
    output_vec = np.append(np.append(prev_vec,first_portion),post_vec)
    return output_vec

'''
:param: first_portion - 
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
:param: indep_vec 
:param: dep_vec - 
:param: jth - 
:param: observations_to_perturb - 
:return: output_indep, output_dep - 2-tuple of 

'''
def mc_sample_strumbelj(indep_vec, dep_vec, jth, observations_to_perturb):
    blocks = list(range(len(observations_to_perturb)))
    contribution_sum = 0
    output_vec = []
    first, second = np.random.choice(blocks, size = 2, replace=False)
    assert len(blocks) >= 3, "not enough blocks"
    assert (first < second) or (first > second), "Overlapping intervals."
    assert first != second, "indices same when not supposed to."
    first_portion = observations_to_perturb[first]
    second_portion = observations_to_perturb[second]
    '''
    plt.plot(list(range(0,first_portion[0])),dep_vec[:first_portion[0]], color="b")
    plt.plot(first_portion,list(np.take(dep_vec, first_portion, axis=0)),color="g")
    plt.plot(list(range(first_portion[-1]+1,second_portion[0])), dep_vec[(first_portion[-1]+1):second_portion[0]], color="b")
    plt.plot(second_portion, list(np.take(dep_vec, second_portion, axis=0)), color="g")
    plt.plot(list(range(second_portion[-1]+1,len(dep_vec))), dep_vec[(second_portion[-1]+1):], color="b")
    plt.savefig('./plot.png')
    plt.close()
    '''

    if (first < second):
        output_indep, output_dep = perturb_x_y(first_portion, 
                                        second_portion, indep_vec, dep_vec, jth)
    elif (first > second):
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
def sample_similar_segments(test_x_vec, test_y_vec, jth, ref_portion, sample_type):
    if (sample_type is not None and len(sample_type) != 0):
        return mc_sample_strumbelj(test_x_vec, test_y_vec, jth, most_similar_segments(ref_portion[0],ref_portion[1],
                                                                test_x_vec, 10))
    else:
        print("No specified sample type.")
        return []

'''
:param: input_tensor - 
:param: dep_tensor - .
:param: region_to_perturb - .
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
        result=[sample_similar_segments(input_tensor[:, i, 0:1, j], dep_tensor[:,i,j], jth,\
                                                region_to_perturb, sample_type) for i in range(input_tensor.shape[1]) \
                                                for j in range(input_tensor.shape[3])]
    perturbed_indep = list(map(lambda x: x[0], result))
    perturbed_dep = list(map(lambda x: x[1], result))
    perturbed_indep = np.reshape(np.array(perturbed_indep, dtype=np.float64), input_tensor.shape)
    perturbed_dep = np.reshape(np.array(perturbed_dep, dtype=np.float64), dep_tensor.shape)
    #np.save("perturbed_indep.npy", perturbed_indep, allow_pickle=True)
    #np.save("perturbed_dep.npy", perturbed_dep, allow_pickle=True)
    return perturbed_indep, perturbed_dep

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
def perturb_data(input_tensor, dep_tensor, region_to_perturb, data_type, dim, M_iterations, jth, sample_type="mc_sample_strumbelj"):
    perturbed_indep, perturbed_dep = perturb_func(input_tensor, 
                                                  dep_tensor,
                                                  region_to_perturb,
                                                  data_type,
                                                  dim,
                                                  jth,
                                                  sample_type)
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

# Noise Reduction Techniques
def linear_smoother(ts, num_neighbors, weight_method="gaussian"):
    averaged_points = []
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
            averaged_points.append(average(heap_items))
        else:
            # rather than a weighted average of nearest data points,
            # we take ordinary average.
            pass
    return averaged_points 

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

def most_similar_segments(start_idx,end_idx,ts,k,compute_diff=True,avg_diff=False):
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
    sim_to_region = dict()
    seen_segments = []
    for i in range(window_size, len(ts), window_size): # we jump by window_size
        #print("Entered")
        segment = ts[(i-window_size):i]
        if compute_diff:
            segment = deltas(segment)
        if avg_diff:
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
                sim_to_region[sim] = list(range(i-window_size,i))
                heapq.heappush(L,sim)
                num_segments += 1
                #print(L)
        else:
            least_sim_largest_dist = heapq.nsmallest(1,L)[0]
            if sim > least_sim_largest_dist and sim not in L: # number of segments 
                # exceeds k if sim is added.
#                 sim_to_region[sim] = list(range(i-window_size,i))
                # need to remove the least similar region if similarity is greater than
                # the least similar one in the heap.
                #print("\nEntered segments > k:", sim_to_region.keys(), L, "Smallest: ",heapq.nsmallest(1,L), "\n")
                del sim_to_region[least_sim_largest_dist]
                val = heapq.heappop(L)
                assert least_sim_largest_dist == val, "least similar not same as val" 
                sim_to_region[sim] = list(range(i-window_size,i))
                heapq.heappush(L,sim)
                #if (least_sim_largest_dist not in sim_to_region):
                    # print(f"sim_to_region no key {least_sim_largest_dist} {sim_to_region}")
                # the two lines below are part of eliminating the least similar
                # here we pop the smallest element from L.
        seen_segments.append([i-window_size,i])
    #if (len(sim_to_region.keys()) < 2):
    #    print(ts, start_idx,end_idx, sim_to_region, L)
    #print("Keys and length of the dictionary",sim_to_region.keys(), len(ts), window_size)
    return list(sim_to_region.values())

def plot_similar_cyclic_regions(indices, ts):
    indices = sorted(indices, key=lambda x: x[0])
    lsts_idx_excluded = []
    curr_end = None
    plt.plot(list(range(250)),pems8_data[250:500,0,0],color="yellow")
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

if __name__ == "__main__":
    detector = 0
    plt_title = "Average Speed of vehicles detected at Detector {}".format(detector)
    xlab = "Number of 5 minute intervals since first day of Jan. 2018"
    ylab = "Average Speed in mph"
