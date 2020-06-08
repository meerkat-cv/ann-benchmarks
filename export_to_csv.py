import os
import matplotlib as mpl
mpl.use('Agg')  # noqa
import matplotlib.pyplot as plt
import numpy as np
import argparse

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.algorithms.definitions import get_definitions
from ann_benchmarks.plotting.metrics import all_metrics as metrics
from ann_benchmarks.plotting.utils import compute_all_metrics

from ann_benchmarks import results

import csv

def export_to_csv(all_data, fn_out):
    #import json
    #with open("sample.json", "w"):
    #    json.dump(all_data, fp,  indent=2)
    
    csvfile = open(fn_out, 'w', newline='')
    writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    csv_header = None
    results = all_data['non-batch']
    for desc in sorted(results.keys(), key=lambda x: x.lower()):
        K = get_count_from_desc(desc)
        dataset_name = get_dataset_from_desc(desc)
        distance = get_distance_from_desc(desc)
        for algo_name in results[desc].keys():
            for run in results[desc][algo_name]:
                run_params = run[1]
                run_metrics = run[2].values()
                if csv_header is None :
                    csv_header = ["dataset", "K", "distance", "algorithm", "parameters", *metrics]
                    writer.writerow(csv_header)
                
                csv_row = [dataset_name,int(K),distance,algo_name,run_params,*run_metrics]
                writer.writerow(csv_row)
    csvfile.close()
    print("output written to : ", fn_out)
            
        

def get_run_desc(properties):
    return "%(dataset)s_%(count)d_%(distance)s" % properties

def get_count_from_desc(desc):
    return desc.split("_")[1]

def get_dataset_from_desc(desc):
    return desc.split("_")[0]

def get_distance_from_desc(desc):
    return desc.split("_")[2]

def get_dataset_label(desc):
    return "{} (k = {})".format(get_dataset_from_desc(desc),
                                get_count_from_desc(desc))

def load_all_results(dataset, recompute = False, cached_only=False):
    """Read all result files and compute all metrics"""
    all_runs_by_dataset = {'batch': {}, 'non-batch': {}}
    all_runs_by_algorithm = {'batch': {}, 'non-batch': {}}
    cached_true_dist = []
    old_sdn = None
    for properties, f in results.load_all_results(dataset):
        sdn = get_run_desc(properties)
        if sdn != old_sdn:
            dataset = get_dataset(properties["dataset"])
            #cached_true_dist = list(dataset["distances"])
            old_sdn = sdn
        algo = properties["algo"]
        ms = compute_all_metrics(
            dataset, f, properties, recompute, cached_only)
        algo_ds = get_dataset_label(sdn)
        idx = "non-batch"
        if properties["batch_mode"]:
            idx = "batch"
        all_runs_by_algorithm[idx].setdefault(
            algo, {}).setdefault(algo_ds, []).append(ms)
        all_runs_by_dataset[idx].setdefault(
            sdn, {}).setdefault(algo, []).append(ms)

    return (all_runs_by_dataset, all_runs_by_algorithm)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        metavar="DATASET",
        default='glove-100-angular')
    parser.add_argument(
        '--count',
        default=10)
    parser.add_argument(
        '--definitions',
        metavar='FILE',
        help='load algorithm definitions from FILE',
        default='algos.yaml')
    parser.add_argument(
        '--limit',
        default=-1)
    parser.add_argument(
        '-o', '--output')
    parser.add_argument(
        '--batch',
        help='Plot runs in batch mode',
        action='store_true')
    parser.add_argument(
        '--recompute',
        help='Clears the cache and recomputes the metrics',
        action='store_true')
    parser.add_argument(
        '--cached_only',
        help='Generates results only with cached values',
        action='store_true')
    args = parser.parse_args()

    if not args.output:
        args.output = 'results/%s.csv' % results.get_algorithm_name(
            args.dataset, args.batch)
        print('writing output to %s' % args.output)

    dataset = get_dataset(args.dataset)
    count = int(args.count)
    results = load_all_results(args.dataset, args.recompute, args.cached_only)
    if not results:
        raise Exception('Nothing to export')

    export_to_csv(results[0], args.output)
