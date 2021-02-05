## This repository contains the code used in the Cacheus Project

All caching algorithm implementations are located in the code/algs folder. This folder contains multiple algorithm implementations using original code provided on several publications. Examples of such policies are LIRS, ARC and DLIRS which an cover important state-of-the-art algorithms for cache management.

In addition, this repo includes the parsing code to run cache simulation with several workloads available on the [SNIA Website](http://iotta.snia.org/tracetypes/3) such as FIU, MSR, Nexus 5 Smartphone, CloudCache and CloudVPS.

### Cacheus

Cacheus is a novel cache replacement algorithms designed for paging domain. This strategy is an evolution of the ML-based algorithm LeCaR which achieves good performance when cache sizes are small relative to the working set.

### Running

To run experiments, the configuration file can be modified appropriately with the specific parameters such as input file location, cache size, algorithm and dataset name.
The next step is executing the following command in the console:

```python run.py example.config```

This framework also allows generating detailed plots to visualize the internal state of the algorithms as well as hit-rate over time and workload's access patterns.

### Paper

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

