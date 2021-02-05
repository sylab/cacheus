from algs.lib.cacheop import CacheOp
from algs.lib.traces import identify_trace, get_trace_reader
from algs.lib.progress_bar import ProgressBar
from algs.get_algorithm import get_algorithm
from timeit import timeit
from itertools import product
import numpy as np
import csv
import os

progress_bar_size = 30


class AlgorithmTest:
    def __init__(self, algorithm, cache_size, cache_size_label,
                 cache_size_label_type, window_size, trace_file, alg_args, **kwargs):
        self.algorithm = algorithm
        self.cache_size_label = cache_size_label
        self.cache_size_label_type = cache_size_label_type
        self.cache_size = cache_size
        self.trace_file = trace_file
        self.alg_args = alg_args
        self.window_size = window_size
        self.trace_type = identify_trace(trace_file)
        trace_reader = get_trace_reader(self.trace_type)
        self.reader = trace_reader(trace_file, **kwargs)

        self.misses = 0
        self.filters = 0
        self.writes = 0

    def _run(self):
        alg = get_algorithm(self.algorithm)(self.cache_size, self.window_size, **self.alg_args)
        progress_bar = ProgressBar(progress_bar_size,
                                   title="{} {}".format(
                                       self.algorithm, self.cache_size))
        for lba, write, _ in self.reader.read():
            op, evicted = alg.request(lba, _)
            if op == CacheOp.INSERT:
                self.misses += 1
                self.writes += 1
            elif op == CacheOp.HIT and write:
                self.writes += 1
            elif op == CacheOp.FILTER:
                self.misses += 1
                self.filters += 1
            progress_bar.progress = self.reader.progress
            progress_bar.print()
        progress_bar.print_complete()
        self.avg_pollution = np.mean(
            alg.pollution.Y
        ) if 'enable_pollution' in self.alg_args and self.alg_args[
            'enable_pollution'] else None

    def run(self, config):
        time = round(timeit(self._run, number=1), 2)
        ios = self.reader.num_requests()
        hits = ios - self.misses
        writes = self.writes
        filters = self.filters
        print(
            "Results: {:<10} size={:<8} hits={}, misses={}, hitrate={:4}% writes={} filters={} {:8}s {}"
            .format(self.algorithm, self.cache_size, hits, self.misses,
                    round(100 * hits / ios, 2), writes, filters, time,
                    self.trace_file),
            *(self.alg_args.items() if self.alg_args else {}))

        if "output_csv" in config:
            self.writeCSV(config["output_csv"], hits, ios, writes, filters,
                          time)

    def writeCSV(self, filename, hits, ios, writes, filters, time):
        with open(filename, 'a+') as csvfile:
            writer = csv.writer(csvfile,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                self.trace_file, self.trace_type, self.algorithm, hits,
                self.misses, writes, filters, self.cache_size,
                self.cache_size_label, self.cache_size_label_type,
                round(100 * hits / ios, 2),
                round(self.avg_pollution, 2) if self.avg_pollution else
                self.avg_pollution, time, *self.alg_args.items()
            ])


def runEntireTrace(trace_name, kwargs, title=None):
    trace_type = identify_trace(trace_name)
    trace_reader = get_trace_reader(trace_type)
    reader = trace_reader(trace_name, **kwargs)

    progress_bar = ProgressBar(progress_bar_size, title=title)

    for lba, write, _ in reader.read():
        progress_bar.progress = reader.progress
        progress_bar.print()
    progress_bar.print_complete()

    return reader


def getUniqueCount(trace_name, kwargs):
    reader = runEntireTrace(trace_name, kwargs, title="Counting Uniq")
    return reader.num_unique(), reader.num_requests()


def getReuseCount(trace_name, kwargs):
    reader = runEntireTrace(trace_name, kwargs, title="Counting Reuse")
    return reader.num_reuse(), reader.num_requests()


def generateTraceNames(trace):
    if trace.startswith('~'):
        trace = os.path.expanduser(trace)

    if os.path.isdir(trace):
        for trace_name in os.listdir(trace):
            yield os.path.join(trace, trace_name)
    elif os.path.isfile(trace):
        yield trace
    else:
        raise ValueError("{} is not a directory or a file".format(trace))


def generateAlgorithmTests(algorithm, cache_size, cache_size_label,
                           cache_size_label_type, window_size, trace_name, config):
    alg_config = {}
    if algorithm in config:
        keywords = list(config[algorithm])
        for values in product(*config[algorithm].values()):
            for key, value in zip(keywords, values):
                alg_config[key] = value
            yield AlgorithmTest(algorithm, cache_size, cache_size_label,
                                cache_size_label_type, window_size, trace_name, alg_config,
                                **config)
    else:
        yield AlgorithmTest(algorithm, cache_size, cache_size_label,
                            cache_size_label_type, window_size, trace_name, alg_config,
                            **config)


if __name__ == '__main__':
    import sys
    import json
    import math
    import os

    with open(sys.argv[1], 'r') as f:
        config = json.loads(f.read())

    # TODO revisit and cleanup
    if 'request_count_type' in config:
        if config['request_count_type'] == 'reuse':
            requestCounter = getReuseCount
        elif config['request_count_type'] == 'unique':
            requestCounter = getUniqueCount
        else:
            raise ValueError("Unknown request_count_type found in config")
    else:
        requestCounter = getUniqueCount

    for trace in config['traces']:
        for trace_name in generateTraceNames(trace):
            print(trace_name)
            if any(map(lambda x: isinstance(x, float), config['cache_sizes'])):
                count, total = requestCounter(trace_name, config)
                window_size = int(0.01*total)
            else:
                window_size = 100
            for cache_size in config['cache_sizes']:
                cache_size_label = cache_size
                cache_size_label_type = 'size'
                if isinstance(cache_size, float):
                    cache_size = math.floor(cache_size * count)
                    cache_size_label_type = config['request_count_type']
                if cache_size < 10:
                    print(
                        "Cache size {} too small for trace {}. Calculated size is {}. Skipping"
                        .format(cache_size_label, trace_name, cache_size),
                        file=sys.stderr)
                    continue

                for algorithm in config['algorithms']:
                    for test in generateAlgorithmTests(algorithm, cache_size,
                                                       cache_size_label,
                                                       cache_size_label_type,
                                                       window_size, trace_name, config):
                        test.run(config)
