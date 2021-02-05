from .lib.dequedict import DequeDict
from .lib.pollutionator import Pollutionator
from .lib.optional_args import process_kwargs
from .lib.visualizinator import Visualizinator
from .lib.cacheop import CacheOp


class LIRS:
    class LIRS_Entry:
        def __init__(self, oblock, is_LIR=False, in_cache=True):
            self.oblock = oblock
            self.is_LIR = is_LIR
            self.in_cache = in_cache

        def __repr__(self):
            return "(o={}, is_LIR={}, in_cache={})".format(
                self.oblock, self.is_LIR, self.in_cache)

    def __init__(self, cache_size, window_size, **kwargs):
        self.cache_size = cache_size

        self.hirs_ratio = 0.01

        process_kwargs(self, kwargs, acceptable_kws=['hirs_ratio'])

        self.hirs_limit = max(2, int((self.cache_size * self.hirs_ratio)))
        self.lirs_limit = self.cache_size - self.hirs_limit

        self.hirs_count = 0
        self.lirs_count = 0
        self.nonresident = 0

        # s stack, semi-split to find nonresident HIRs quickly
        self.s = DequeDict()
        self.nr_hirs = DequeDict()
        # q, the resident HIR stack
        self.q = DequeDict()

        self.time = 0
        self.last_oblock = None
        self.visual = Visualizinator(labels=['hit-rate', 'q_size'],
                                     windowed_labels=['hit-rate'],
                                     window_size=window_size,
                                     **kwargs)
        self.pollution = Pollutionator(cache_size, **kwargs)

    def __contains__(self, oblock):
        if oblock in self.s:
            return self.s[oblock].in_cache
        return oblock in self.q

    def cacheFull(self):
        return self.lirs_count + self.hirs_count == self.cache_size

    def hitLIR(self, oblock):
        lru_lir = self.s.first()
        x = self.s[oblock]
        self.s[oblock] = x
        if lru_lir is x:
            self.prune()

    def prune(self):
        while self.s:
            x = self.s.first()
            if x.is_LIR:
                break

            del self.s[x.oblock]
            if not x.in_cache:
                del self.nr_hirs[x.oblock]
                self.nonresident -= 1

    def hitHIRinLIRS(self, oblock):
        evicted = None

        x = self.s[oblock]

        if x.in_cache:
            del self.s[oblock]
            del self.q[oblock]
            self.hirs_count -= 1
        else:
            del self.s[oblock]
            del self.nr_hirs[oblock]
            self.nonresident -= 1

            if self.cacheFull():
                evicted = self.ejectHIR()

        if self.lirs_count >= self.lirs_limit:
            self.ejectLIR()

        self.s[oblock] = x
        x.in_cache = True
        x.is_LIR = True
        self.lirs_count += 1

        return evicted

    def ejectLIR(self):
        assert (self.s.first().is_LIR)

        lru = self.s.popFirst()
        self.lirs_count -= 1
        lru.is_LIR = False

        self.q[lru.oblock] = lru
        self.hirs_count += 1

        self.prune()

    def ejectHIR(self):
        lru = self.q.popFirst()
        self.hirs_count -= 1

        if lru.oblock in self.s:
            self.nr_hirs[lru.oblock] = lru
            lru.in_cache = False
            self.nonresident += 1

        self.pollution.remove(lru.oblock)

        return lru.oblock

    def hitHIRinQ(self, oblock):
        x = self.q[oblock]
        self.q[oblock] = x
        self.s[oblock] = x

    def limitStack(self):
        while len(self.s) > (2 * self.cache_size):
            lru = self.nr_hirs.popFirst()
            del self.s[lru.oblock]
            self.nonresident -= 1

    def miss(self, oblock):
        evicted = None

        if self.cacheFull():
            evicted = self.ejectHIR()

        if self.lirs_count < self.lirs_limit and self.hirs_count == 0:
            x = self.LIRS_Entry(oblock, is_LIR=True)
            self.s[oblock] = x
            self.lirs_count += 1
        else:
            x = self.LIRS_Entry(oblock, is_LIR=False)
            self.s[oblock] = x
            self.q[oblock] = x
            self.hirs_count += 1

        return evicted

    def request(self, oblock, ts):
        miss = oblock not in self
        evicted = None

        self.time += 1
        self.visual.add({
            'q_size': (self.time, self.hirs_limit, ts)
        })
        if oblock != self.last_oblock:
            self.last_oblock = oblock

            if oblock in self.s:
                x = self.s[oblock]
                if x.is_LIR:
                    self.hitLIR(oblock)
                else:
                    evicted = self.hitHIRinLIRS(oblock)
            elif oblock in self.q:
                self.hitHIRinQ(oblock)
            else:
                evicted = self.miss(oblock)

        self.limitStack()

        self.visual.addWindow({'hit-rate': 0 if miss else 1}, self.time, ts)

        if miss:
            self.pollution.incrementUniqueCount()
        self.pollution.setUnique(oblock)
        self.pollution.update(self.time)

        op = CacheOp.INSERT if miss else CacheOp.HIT

        return op, evicted
