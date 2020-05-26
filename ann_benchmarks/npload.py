import numpy as np
import os
import re
import argparse

def load_input(path, max_n=None):
    pts_vec, labels_vec = [], []
    pts_files = [ f for f in sorted(os.listdir(path)) if f.startswith("pts_") ]
    for filename in pts_files:
        print(filename)
        curr_pts = np.load(path+'/'+filename)
        pts_vec.extend([curr_pts[x] for x in range(curr_pts.shape[0])])
            
        curr_names = np.load(path+'/'+re.sub("pts_","labels_",filename))
        labels_vec.extend([str(curr_names[x]) for x in range(curr_names.shape[0])])

        if (max_n is not None) and (len(pts_vec) > max_n):
            #we have read enougth 
            break
        
    if max_n is None:
        max_n = len(pts_vec)
    else:
        max_n = min(max_n, len(pts_vec))
        
    return np.asarray(pts_vec[0:max_n], dtype=np.float64), np.asarray(labels_vec[0:max_n])


if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--path", default="./descriptors_v2", type=str, required=True, help="path with descriptors and labels")
    parser.add_argument("-N", default=None, type=int, required=False, help="Number of points to load")

    args = parser.parse_args()

    dataset, labels = load_input(args.path, args.N)
