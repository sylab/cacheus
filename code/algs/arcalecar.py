from .alecar6 import ALeCaR6
from .arc import ARC
from .lfu import LFU
from .lib.dequedict import DequeDict
from .lib.heapdict import HeapDict
from .lib.pollutionator import Pollutionator
from .lib.visualizinator import Visualizinator
from .lib.optional_args import process_kwargs
from .lib.cacheop import CacheOp
import numpy as np


class ARCALeCaR(ALeCaR6):
    class ARC(ARC):
        def get(self, oblock):
            if oblock in self.T1:
                return self.T1[oblock]
            if oblock in self.T2:
                return self.T2[oblock]
            if oblock in self.B1:
                return self.B1[oblock]
            if oblock in self.B2:
                return self.B2[oblock]
            return None

        def replaceSafe(self):
            # getting rid of x_in_B2 as that's never a concern.
            # removed the logic related to x_in_B2 as well
            if len(self.T1) > self.p:
                return self.T1.first()
            else:
                return self.T2.first()

        def nextVictim(self):
            len_L1 = len(self.T1) + len(self.B1)
            len_L2 = len(self.T2) + len(self.B2)

            if len_L1 >= self.cache_size:
                if len(self.T1) < self.cache_size:
                    return self.replaceSafe()
                else:
                    return self.T1.first()
            elif len_L1 < self.cache_size and len_L1 + len_L2 >= self.cache_size:
                return self.replaceSafe()

        def evictThis(self, oblock, put_in_history=True):
            if put_in_history:
                next_victim = self.nextVictim().oblock
                if oblock == next_victim:
                    evicted = self.evict()
                    assert (evicted == next_victim)
                else:
                    victim = self.get(oblock)
                    if oblock in self.T1:
                        if len(self.B1) + len(self.B2) == self.cache_size:
                            if len(self.B1) == 0:
                                self.B2.popFirst()
                            else:
                                self.B1.popFirst()
                        del self.T1[oblock]
                        self.B1[oblock] = victim
                    else:
                        if len(self.B1) + len(self.B2) == self.cache_size:
                            if len(self.B2) == 0:
                                self.B1.popFirst()
                            else:
                                self.B2.popFirst()
                        del self.T2[oblock]
                        self.B2[oblock] = victim
            else:
                if oblock in self.T1:
                    del self.T1[oblock]
                else:
                    del self.T2[oblock]

        def missInHistory(self, oblock, history):
            x = history[oblock]
            x_in_B2 = oblock in self.B2
            del history[oblock]

            if len(self.T1) + len(self.T2) == self.cache_size:
                evicted = self.replace(x_in_B2)
                self.pollution.remove(evicted.oblock)

            self.moveToList(x, self.T2)

    class LFU(LFU):
        def get(self, oblock):
            return self.lfu[oblock]

        def nextVictim(self):
            return self.lfu.min()

        def evictThis(self, oblock):
            del self.lfu[oblock]

        def request(self, oblock, ts, freq=None):
            miss = super().request(oblock, ts)

            if freq != None:
                assert (isinstance(freq, int))
                x = self.lfu[oblock]
                del self.lfu[oblock]
                x.freq = freq
                self.lfu[oblock] = x

            return miss

    class ARCALeCaR_Entry(ALeCaR6.ALeCaR6_Entry):
        pass

    class ARCALeCaR_Learning_Rate(ALeCaR6.ALeCaR6_Learning_Rate):
        pass

    def __init__(self, cache_size, window_size, **kwargs):
        np.random.seed(123)
        self.time = 0

        self.cache_size = cache_size

        kwargs_arc = {}
        kwargs_lfu = {}
        if 'arc' in kwargs:
            kwargs_arc = kwargs['arc']
        if 'lfu' in kwargs:
            kwargs_lfu = kwargs['lfu']

        self.arc = self.ARC(cache_size, window_size, **kwargs_arc)
        self.lfu = self.LFU(cache_size, window_size, **kwargs_lfu)

        self.history_size = cache_size
        self.history = DequeDict()

        self.initial_weight = 0.5

        self.learning_rate = self.ARCALeCaR_Learning_Rate(cache_size, **kwargs)

        process_kwargs(self,
                       kwargs,
                       acceptable_kws=['initial_weight', 'history_size'])

        self.W = np.array([self.initial_weight, 1 - self.initial_weight],
                          dtype=np.float32)

        self.visual = Visualizinator(labels=['W_arc', 'W_lfu', 'hit-rate'],
                                     windowed_labels=['hit-rate'],
                                     window_size=window_size,
                                     **kwargs)

        self.pollution = Pollutionator(cache_size, **kwargs)

    def __contains__(self, oblock):
        return oblock in self.lfu

    def cacheFull(self):
        return self.lfu.cacheFull()

    def addToCache(self, oblock, freq):
        self.arc.request(oblock, None)
        self.lfu.request(oblock, None, freq=freq)

    def addToHistory(self, x, policy):
        policy_history = None
        if policy == 0:
            policy_history = "ARC"
        elif policy == 1:
            policy_history = "LFU"
        elif policy == -1:
            return False

        # prune history for lazy removal to match ARC's history
        if len(self.history) == 2 * self.history_size:
            history_oblocks = [meta.oblock for meta in self.history]
            for oblock in history_oblocks:
                if not (oblock in self.arc.B1 or oblock in self.arc.B2):
                    del self.history[oblock]

        x.evicted_me = policy_history

        self.history[x.oblock] = x

        return True

    def evict(self):
        arc = self.arc.nextVictim()
        lfu = self.lfu.nextVictim()

        assert (arc != None)
        assert (lfu != None)

        evicted = arc
        policy = self.getChoice()

        if arc.oblock == lfu.oblock:
            evicted, policy = arc, -1
        elif policy == 0:
            evicted = arc
        else:
            evicted = lfu

        # save info to meta ARCLeCaREntry for history
        meta = self.ARCALeCaR_Entry(evicted.oblock, time=self.time)

        # arc data for meta
        # Not really any that matters. All of that is in ARC

        # lfu data for meta
        meta.freq = self.lfu.get(evicted.oblock).freq

        put_in_history = self.addToHistory(meta, policy)

        # evict from both
        self.arc.evictThis(evicted.oblock, put_in_history=put_in_history)
        self.lfu.evictThis(evicted.oblock)

        self.pollution.remove(evicted.oblock)

        return meta.oblock, policy

    def hit(self, oblock):
        self.arc.request(oblock, None)
        self.lfu.request(oblock, None)

    # NOTE: adjustWeights has parameters rewardLRU and rewardLFU but technically
    # the naming didn't matter and we can use as it is

    def miss(self, oblock):
        freq = None
        evicted = None

        if oblock in self.history:
            # history based on N in ARC's history
            # we can do get since we missed so this should be checking
            # B1 or B2 only now
            if self.arc.get(oblock) != None:
                meta = self.history[oblock]
                freq = meta.freq + 1

                if meta.evicted_me == "ARC":
                    self.adjustWeights(-1, 0)
                else:
                    self.adjustWeights(0, -1)

            del self.history[oblock]

        if len(self.lfu.lfu) == self.cache_size:
            evicted, policy = self.evict()

        self.addToCache(oblock, freq)

        return evicted

    def request(self, oblock, ts):
        miss = True
        evicted = None
        op = CacheOp.INSERT

        self.time += 1

        self.visual.add({
            'W_arc': (self.time, self.W[0]),
            'W_lfu': (self.time, self.W[1])
        })

        self.learning_rate.update(self.time)

        if oblock in self:
            miss = False
            op = CacheOp.HIT
            self.hit(oblock)
        else:
            evicted = self.miss(oblock)

        self.visual.addWindow({'hit-rate': 0 if miss else 1}, self.time, ts)

        if not miss:
            self.learning_rate.hitrate += 1

        if miss:
            self.pollution.incrementUniqueCount()
        self.pollution.setUnique(oblock)
        self.pollution.update(self.time)

        return op, evicted
