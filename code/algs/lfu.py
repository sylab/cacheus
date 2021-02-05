from .lib.heapdict import HeapDict
from .lib.pollutionator import Pollutionator
from .lib.visualizinator import Visualizinator
from .lib.cacheop import CacheOp


class LFU:
    class LFU_Entry:
        def __init__(self, oblock, freq=1, time=0):
            self.oblock = oblock
            self.freq = freq
            self.time = time

        def __lt__(self, other):
            if self.freq == other.freq:
                return self.time > other.time
            return self.freq < other.freq

        def __repr__(self):
            return "(o={}, f={}, t={})".format(self.oblock, self.freq,
                                               self.time)

    def __init__(self, cache_size, window_size, **kwargs):
        self.cache_size = cache_size
        self.lfu = HeapDict()
        self.time = 0
        self.visual = Visualizinator(labels=['hit-rate'],
                                     windowed_labels=['hit-rate'],
                                     window_size=window_size,
                                     **kwargs)

        self.pollution = Pollutionator(cache_size, **kwargs)

    def __contains__(self, oblock):
        return oblock in self.lfu

    def cacheFull(self):
        return len(self.lfu) == self.cache_size

    def addToCache(self, oblock):
        x = self.LFU_Entry(oblock, freq=1, time=self.time)
        self.lfu[oblock] = x

    def hit(self, oblock):
        x = self.lfu[oblock]
        x.freq += 1
        x.time = self.time
        self.lfu[oblock] = x

    def evict(self):
        lfu_min = self.lfu.popMin()
        self.pollution.remove(lfu_min.oblock)
        return lfu_min.oblock

    def miss(self, oblock):
        evicted = None

        if len(self.lfu) == self.cache_size:
            evicted = self.evict()
        self.addToCache(oblock)

        return evicted

    def request(self, oblock, ts):
        miss = True
        evicted = None
        op = CacheOp.INSERT

        self.time += 1

        if oblock in self:
            miss = False
            op = CacheOp.HIT
            self.hit(oblock)
        else:
            evicted = self.miss(oblock)

        self.visual.addWindow({'hit-rate': 0 if miss else 1}, self.time, ts)

        # Pollutionator
        if miss:
            self.pollution.incrementUniqueCount()
        self.pollution.setUnique(oblock)
        self.pollution.update(self.time)

        return op, evicted
