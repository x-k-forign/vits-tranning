import numpy as np

def maximum_path_each(path, value, t_y, t_x, max_neg_val=-1e9):
    """
    A Python version of the Cython function to compute the maximum path.
    """
    index = t_x - 1
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[y-1, x]
            if x == 0:
                if y == 0:
                    v_prev = 0.
                else:
                    v_prev = max_neg_val
            else:
                v_prev = value[y-1, x-1]
            value[y, x] += max(v_prev, v_cur)

    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y-1, index] < value[y-1, index-1]):
            index = index - 1


def maximum_path_c(paths, values, t_ys, t_xs):
    """
    A Python version of the Cython function to process multiple batches.
    """
    b = paths.shape[0]
    for i in range(b):
        maximum_path_each(paths[i], values[i], t_ys[i], t_xs[i])
