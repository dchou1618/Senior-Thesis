import argparse
import numpy as np

# predict_and_save_results_mstgcn - under lib/utils.py

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
:return:
'''

def ma(ts,window):
    ts_points = []
    for i in range(window,len(ts)):
        ts_points.append(np.sum(ts[(i-window):i])/window)
    return ts_points

'''
:param: vec
:return:
'''

def norm(vec):
    return np.sqrt(np.sum([x**2 for x in vec]))

'''

:param: vec1
:param: vec2
:return:
'''
def cosine_sim(vec1,vec2):
    return np.dot(vec1,vec2)/(norm(vec1)*norm(vec2))


'''
:param: vec
:return:
'''
def deltas(vec):
    diffs = []
    for i in range(1,len(vec)):
        diffs.append(vec[i]-vec[i-1])
    return diffs

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
    cmp_window = ts[start_idx:end_idx]
    if compute_diff:
        cmp_window = deltas(cmp_window)
    window_size = end_idx-start_idx
    L = []
    heapq.heapify(L)
    num_segments = 0
    sim_to_region = dict()
    seen_segments = []
    for i in range(window_size,len(ts)):
        segment = ts[(i-window_size):i]
        if compute_diff:
            segment = deltas(segment)
        if avg_diff:
            sim = np.sum(np.array(cmp_window)-np.array(segment))/len(cmp_window)
        else:
            sim = correlate(cmp_window,segment)[0]
            #sim = cosine_sim(cmp_window,segment)
        sim = round(sim,6)
        if len(seen_segments) != 0:
            if seen_segments[-1][-1] >= i-window_size:
                continue
        if num_segments < k:
            sim_to_region[sim] = list(range(i-window_size,i))
            heapq.heappush(L,sim)
            num_segments += 1
        else:
            least_sim_largest_dist = heapq.nsmallest(1,L)[0]
#             print("Keeping size:",len(L))
#             print("least_similar",least_sim_largest_dist)
            if sim >= least_sim_largest_dist: # number of segments exceeds k
#                 sim_to_region[sim] = list(range(i-window_size,i))
                # need to remove the least similar region if similarity is greater than
                # the least similar one in the heap.
                sim_to_region[sim] = list(range(i-window_size,i))
                heapq.heappush(L,sim)
                del sim_to_region[least_sim_largest_dist]
                # here we pop the smallest element from L.
                heapq.heappop(L)
        seen_segments.append([i-window_size,i])
    print(sim_to_region.keys())
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

# def probe_predictions(model_params, model, train_data, test_data):
#     initialized = model(model_params)
#     initialized.train(train_data)
#     predicted_data = initialized.predict(test_data)
#     Predicted average speeds: predicted_data[:,<detector_num>,2]
#     Plot differences between predicted and actual data

if __name__ == "__main__":
    detector = 0
    plt_title = "Average Speed of vehicles detected at Detector {}".format(detector)
    xlab = "Number of 5 minute intervals since first day of Jan. 2018"
    ylab = "Average Speed in mph"
