from .lib.dequedict import DequeDict
from .lib.visualizinator import Visualizinator
from .lib.pollutionator import Pollutionator
from .lib.cacheop import CacheOp


class ARC:
    class ARC_Entry:
        def __init__(self, oblock):
            self.oblock = oblock

        def __repr__(self):
            return "({})".format(self.oblock)

    def __init__(self, cache_size, window_size, **kwargs):
        self.cache_size = cache_size
        self.p = 0

        self.T1 = DequeDict()
        self.T2 = DequeDict()
        self.B1 = DequeDict()
        self.B2 = DequeDict()

        self.time = 0
        self.visual = Visualizinator(labels=['hit-rate', 'W_lru'],
                                     windowed_labels=['hit-rate'],
                                     window_size=window_size,
                                     **kwargs)

        self.pollution = Pollutionator(self.cache_size, **kwargs)

    def __contains__(self, oblock):
        return oblock in self.T1 or oblock in self.T2

    def cacheFull(self):
        return len(self.T1) + len(self.T2) == self.cache_size

    def addToCache(self, oblock):
        x = self.ARC_Entry(oblock)
        self.T1[oblock] = x

    def moveToList(self, entry, arc_list):
        arc_list[entry.oblock] = entry

    def hit(self, oblock, arc_list):
        x = arc_list[oblock]
        del arc_list[oblock]
        self.moveToList(x, self.T2)

    def evictFromList(self, arc_list):
        assert (len(arc_list) > 0)
        return arc_list.popFirst()

    def evict(self):
        len_L1 = len(self.T1) + len(self.B1)
        len_L2 = len(self.T2) + len(self.B2)

        if len_L1 >= self.cache_size:
            if len(self.T1) < self.cache_size:
                hist_evict = self.evictFromList(self.B1)
                evicted = self.replace()
            else:
                evicted = self.evictFromList(self.T1)
        elif len_L1 < self.cache_size and len_L1 + len_L2 >= self.cache_size:
            if len_L1 + len_L2 == 2 * self.cache_size:
                self.evictFromList(self.B2)
            evicted = self.replace()
        self.pollution.remove(evicted.oblock)
        return evicted.oblock

    def replace(self, x_in_B2=False):
        if len(self.T1) > 0 and ((x_in_B2 and len(self.T1) == self.p)
                                 or len(self.T1) > self.p):
            evicted = self.evictFromList(self.T1)
            self.moveToList(evicted, self.B1)
        else:
            evicted = self.evictFromList(self.T2)
            self.moveToList(evicted, self.B2)
        return evicted

    def missInHistory(self, oblock, history):
        x = history[oblock]
        x_in_B2 = oblock in self.B2
        del history[oblock]

        evicted = self.replace(x_in_B2)
        self.pollution.remove(evicted.oblock)

        self.moveToList(x, self.T2)

        return evicted.oblock

    def miss(self, oblock):
        evicted = None

        if oblock in self.B1:
            self.p = min(self.p + max(1,
                                      len(self.B2) // len(self.B1)),
                         self.cache_size)
            evicted = self.missInHistory(oblock, self.B1)
        elif oblock in self.B2:
            self.p = max(self.p - max(1, len(self.B1) // len(self.B2)), 0)
            evicted = self.missInHistory(oblock, self.B2)
        else:
            if self.cacheFull():
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
            if oblock in self.T1:
                self.hit(oblock, self.T1)
            else:
                self.hit(oblock, self.T2)
        else:
            evicted = self.miss(oblock)

        # Visualizinator
        self.visual.add({'W_lru': (self.time, float(self.p/self.cache_size), ts)})
        self.visual.addWindow({'hit-rate': 0 if miss else 1}, self.time, ts)

        # Pollutionator
        if miss:
            self.pollution.incrementUniqueCount()
        self.pollution.setUnique(oblock)
        self.pollution.update(self.time)

        return op, evicted
