from .lib.dequedict import DequeDict
from .lib.pollutionator import Pollutionator
from .lib.visualizinator import Visualizinator
from .lib.cacheop import CacheOp


class MRU:
    class MRU_Entry:
        def __init__(self, oblock):
            self.oblock = oblock

        def __repr__(self):
            return "(o={})".format(self.oblock)

    def __init__(self, cache_size, window_size, **kwargs):
        self.cache_size = cache_size
        self.mru = DequeDict()

        self.time = 0

        self.visual = Visualizinator(labels=['hit-rate'],
                                     windowed_labels=['hit-rate'],
                                     window_size=window_size,
                                     **kwargs)

        self.pollution = Pollutionator(cache_size, **kwargs)

    def __contains__(self, oblock):
        return oblock in self.mru

    def cacheFull(self):
        return len(self.mru) == self.cache_size

    def addToCache(self, oblock):
        x = self.MRU_Entry(oblock)
        self.mru[oblock] = x

    def hit(self, oblock):
        x = self.mru[oblock]
        self.mru[oblock] = x

    def evict(self):
        mru = self.mru.popLast()
        self.pollution.remove(mru.oblock)
        return mru.oblock

    def miss(self, oblock):
        evicted = None

        if len(self.mru) == self.cache_size:
            evicted = self.evict()
        self.addToCache(oblock)

        return evicted

    def request(self, oblock, ts):
        miss = True
        evicted = None

        self.time += 1

        if oblock in self:
            miss = False
            self.hit(oblock)
        else:
            evicted = self.miss(oblock)

        self.visual.addWindow({'hit-rate': 0 if miss else 1}, self.time, ts)

        if miss:
            self.pollution.incrementUniqueCount()
        self.pollution.setUnique(oblock)
        self.pollution.update(self.time)

        op = CacheOp.INSERT if miss else CacheOp.HIT

        return op, evicted
