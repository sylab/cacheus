## This repository contains the code used in the Cacheus Project

All caching algorithm implementations are located in the code/algs folder. These implementation were done using original code shared by the authors when possible or provided the correspoding publication. Examples of such policies are LIRS, ARC and DLIRS which cover state-of-the-art algorithms for caching.

### What is Cacheus?

Cacheus is a novel cache replacement algorithms designed for paging domain. This strategy is an evolution of the ML-based algorithm LeCaR which achieves good performance when cache sizes are small relative to the working set.

### How to run experiments?

To run experiments, the configuration file can be modified appropriately with the specific parameters such as input file location, cache size, algorithm and dataset name.
The next step is executing the following command in the console:

```python3 run.py example.config```

This framework also allows you to generate detailed graphs to visualize the internal state of the algorithms, as well as the hit rate over time and the access patterns of the workload using this command:

```python3 visual.py visual.config```

### Where to find the traces?

Summary of the workloads used in the paper:

1. [FIU SRC_Map](http://iotta.snia.org/traces/block-io/414) (All traces are a one-day duration).
2. [MSR Cambridge](http://iotta.snia.org/traces/block-io/388) (These traces are a one-week duration per file. For the paper, we extracted the first day only based on the timestamp).
3. [CloudVPS](http://visa.lab.asu.edu/web/resources/traces/traces-cloudvps/) (All traces are a one-day duration).
4. CloudCache is a collection of the [webserver](http://visa.lab.asu.edu/web/resources/traces/traces-webserver/) and [moodle](http://visa.lab.asu.edu/web/resources/traces/traces-moodle/) traces that were used in the [Cloudcache](https://www.usenix.org/conference/fast16/technical-sessions/presentation/arteaga) paper from FAST'16. (All one-day duration).
5. CloudPhysics are non-public traces used in the [SHARDS](https://www.usenix.org/conference/fast15/technical-sessions/presentation/waldspurger) paper from FAST'15 that were shared directly from the authors. 

### References

* The relevant paper to cite for follow-up or related work on Cacheus is:

``@inproceedings {cacheus-fast21,
author = {Liana V. Rodriguez and Farzana Yusuf and Steven Lyons and Eysler Paz and Raju Rangaswami and Jason Liu and Ming Zhao and Giri Narasimhan},
title = {Learning Cache Replacement with {CACHEUS}},
booktitle = {19th {USENIX} Conference on File and Storage Technologies ({FAST} 21)},
year = {2021},
url = {https://www.usenix.org/conference/fast21/presentation/valdes},
publisher = {{USENIX} Association},
month = February,
}``


* The relevant paper to cite for follow-up or related work on LeCaR is:

``@inproceedings {lecar-hotstorage19,
author = {Giuseppe Vietri and Liana V. Rodriguez and Wendy A. Martinez and Steven Lyons and Jason Liu and Raju Rangaswami and Ming Zhao and Giri Narasimhan},
title = {Driving Cache Replacement with ML-based LeCaR},
booktitle = {10th {USENIX} Workshop on Hot Topics in Storage and File Systems (HotStorage 18)},
year = {2018},
address = {Boston, MA},
url = {https://www.usenix.org/conference/hotstorage18/presentation/vietri},
publisher = {{USENIX} Association},
month = July,
}``

### Acknowledgments

Song Jiang shared the original code for LIRS that we adapted to Python and used for comparative evaluation.

