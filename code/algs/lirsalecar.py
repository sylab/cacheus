from .lib.dequedict import DequeDict
from .lib.heapdict import HeapDict
from .lib.pollutionator import Pollutionator
from .lib.visualizinator import Visualizinator
from .lib.optional_args import process_kwargs
from .lib.cacheop import CacheOp
from collections import OrderedDict

import numpy as np


class LIRSALeCaR:
    ######################
    ## INTERNAL CLASSES ##
    ######################

    # Entry to track the page information
    class ALeCaR6_Entry:
        def __init__(self, oblock, freq=1, time=0):
            self.oblock = oblock
            self.freq = freq
            self.time = time
            self.evicted_time = None

        # Minimal comparitors needed for HeapDict
        def __lt__(self, other):
            if self.freq == other.freq:
                return np.random.randint(1, 101) % 2 == 0
            return self.freq < other.freq

        # Useful for debugging
        def __repr__(self):
            return "(o={}, f={}, t={})".format(self.oblock, self.freq,
                                               self.time)

    # Adaptive learning rate of ALeCaR6
    # TODO consider an internal time instead of taking time as a parameter
    class ALeCaR6_Learning_Rate:
        # kwargs: We're using keyword arguments so that they can be passed down as
        #         needed. We can filter through the keywords for ones we want,
        #         ignoring those we don't use. We then update our instance with
        #         the passed values for the given keys after the default
        #         initializations and before the possibly passed keys are used in
        #         a way that cannot be taken back, such as setting the learning rate
        #         reset point, which is reliant on the starting learning_rate
        def __init__(self, period_length, **kwargs):
            self.learning_rate = np.sqrt((2.0 * np.log(2)) / period_length)

            process_kwargs(self, kwargs, acceptable_kws=['learning_rate'])

            self.learning_rate_reset = min(max(self.learning_rate, 0.001), 1)
            self.learning_rate_curr = self.learning_rate
            self.learning_rate_prev = 0.0
            self.learning_rates = []

            self.period_len = period_length

            self.hitrate = 0
            self.hitrate_prev = 0.0
            self.hitrate_diff_prev = 0.0

            self.hitrate_nega_count = 0
            self.hitrate_zero_count = 0

        # Used to use the learning_rate value to multiply without
        # having to use self.learning_rate.learning_rate, which can
        # impact readability
        def __mul__(self, other):
            return self.learning_rate * other

        # Update the adaptive learning rate when we've reached the end of a period
        def update(self, time):
            if time % self.period_len == 0:
                # TODO: remove float() when using Python3
                hitrate_curr = round(self.hitrate / float(self.period_len), 3)
                hitrate_diff = round(hitrate_curr - self.hitrate_prev, 3)

                delta_LR = round(self.learning_rate_curr, 3) - round(
                    self.learning_rate_prev, 3)
                delta, delta_HR = self.updateInDeltaDirection(
                    delta_LR, hitrate_diff)

                if delta > 0:
                    self.learning_rate = min(
                        self.learning_rate +
                        abs(self.learning_rate * delta_LR), 1)
                    self.hitrate_nega_count = 0
                    self.hitrate_zero_count = 0
                elif delta < 0:
                    self.learning_rate = max(
                        self.learning_rate -
                        abs(self.learning_rate * delta_LR), 0.001)
                    self.hitrate_nega_count = 0
                    self.hitrate_zero_count = 0
                elif delta == 0 and hitrate_diff <= 0:
                    if (hitrate_curr <= 0 and hitrate_diff == 0):
                        self.hitrate_zero_count += 1
                    if hitrate_diff < 0:
                        self.hitrate_nega_count += 1
                        self.hitrate_zero_count += 1
                    if self.hitrate_zero_count >= 10:
                        self.learning_rate = self.learning_rate_reset
                        self.hitrate_zero_count = 0
                    elif hitrate_diff < 0:
                        if self.hitrate_nega_count >= 10:
                            self.learning_rate = self.learning_rate_reset
                            self.hitrate_nega_count = 0
                        else:
                            self.updateInRandomDirection()
                self.learning_rate_prev = self.learning_rate_curr
                self.learning_rate_curr = self.learning_rate
                self.hitrate_prev = hitrate_curr
                self.hitrate_diff_prev = hitrate_diff
                self.hitrate = 0

            # TODO check that this is necessary and shouldn't be moved to
            #      the Visualizinator
            self.learning_rates.append(self.learning_rate)

        # Update the learning rate according to the change in learning_rate and hitrate
        def updateInDeltaDirection(self, learning_rate_diff, hitrate_diff):
            delta = learning_rate_diff * hitrate_diff
            # Get delta = 1 if learning_rate_diff and hitrate_diff are both positive or negative
            # Get delta =-1 if learning_rate_diff and hitrate_diff have different signs
            # Get delta = 0 if either learning_rate_diff or hitrate_diff == 0
            delta = int(delta / abs(delta)) if delta != 0 else 0
            delta_HR = 0 if delta == 0 and learning_rate_diff != 0 else 1
            return delta, delta_HR

        # Update the learning rate in a random direction or correct it from extremes
        def updateInRandomDirection(self):
            if self.learning_rate >= 1:
                self.learning_rate = 0.9
            elif self.learning_rate <= 0.001:
                self.learning_rate = 0.005
            elif np.random.choice(['Increase', 'Decrease']) == 'Increase':
                self.learning_rate = min(self.learning_rate * 1.25, 1)
            else:
                self.learning_rate = max(self.learning_rate * 0.75, 0.001)

    # kwargs: We're using keyword arguments so that they can be passed down as
    #         needed. We can filter through the keywords for ones we want,
    #         ignoring those we don't use. We then update our instance with
    #         the passed values for the given keys after the default
    #         initializations and before the possibly passed keys are used in
    #         a way that cannot be taken back, such as setting the weights(W)
    #         Please note that cache_size is a required argument and not
    #         optional like all the kwargs are
    def __init__(self, cache_size, window_size, **kwargs):
        # Randomness and Time
        np.random.seed(123)
        self.time = 0

        # Cache
        self.cache_size = cache_size
        self.lirs = self.LIRS(cache_size, 0.01)
        self.lfu = HeapDict()

        # Histories
        self.history_size = int(cache_size * 0.25)
        self.lfu_hist = DequeDict()

        # Decision Weights Initilized
        self.initial_weight = 0.5

        # Learning Rate
        self.learning_rate = self.ALeCaR6_Learning_Rate(cache_size, **kwargs)

        # Apply values in kwargs, before any accepted_kws members
        # are prerequisites
        process_kwargs(self,
                       kwargs,
                       acceptable_kws=['initial_weight', 'history_size'])

        # Decision Weights
        self.W = np.array([self.initial_weight, 1 - self.initial_weight],
                          dtype=np.float32)
        # Visualize
        self.visual = Visualizinator(labels=['W_lru', 'W_lfu', 'hit-rate'],
                                     windowed_labels=['hit-rate'],
                                     window_size=cache_size,
                                     **kwargs)

        # Pollution
        self.pollution = Pollutionator(cache_size, **kwargs)

    # Process and access request for the given oblock
    def request(self, oblock, ts):
        miss = True
        evicted = None

        self.time += 1

        self.visual.add({
            'W_lru': (self.time, self.W[0], ts),
            'W_lfu': (self.time, self.W[1], ts)
        })

        self.learning_rate.update(self.time)

        if oblock in self:
            miss = False
            self.hit(oblock)
        else:
            evicted = self.miss(oblock)

        # Windowed
        self.visual.addWindow({'hit-rate': 0 if miss else 1}, self.time, ts)

        # Learning Rate
        if not miss:
            self.learning_rate.hitrate += 1

        # Pollution
        if miss:
            self.pollution.incrementUniqueCount()
        self.pollution.setUnique(oblock)
        self.pollution.update(self.time)

        op = CacheOp.INSERT if miss else CacheOp.HIT

        return op, evicted

    # Cache Hit
    def hit(self, oblock):
        x = self.lfu[oblock]
        x.time = self.time

        self.lirs.pageHIT(oblock)

        x.freq += 1
        self.lfu[oblock] = x

    # Cache Miss
    def miss(self, oblock):
        evicted = None

        freq = 0
        from_lfu_hist = False
        if oblock in self.lirs.nonresidentHIRsInS:
            entry = self.lirs.nonresidentHIRsInS[oblock]
            if entry.isPrunned:
                del self.lirs.nonresidentHIRsInS[oblock]
            freq = entry.freq
            self.adjustWeights(-1, 0)
        elif oblock in self.lfu_hist:
            entry = self.lfu_hist[oblock]
            freq = entry.freq
            del self.lfu_hist[oblock]
            from_lfu_hist = True
            self.adjustWeights(0, -1)

        # If the cache is full, evict
        if self.lirs.numberHIRpages + self.lirs.numberLIRpages == self.cache_size:
            assert len(self.lfu.heap) == self.cache_size
            evicted, _ = self.evict(oblock)

        self.addToCache(oblock, freq, from_lfu_hist)

        return evicted

    # Evict an entry
    def evict(self, avoid_removal):
        lirs = next(iter(self.lirs.QStack))
        lfu = self.getHeapMin().oblock

        evicted = lirs
        policy = self.getChoice()

        # Since we're using Entry references, we use is to check
        # that the LRU and LFU Entries are the same Entry
        if lirs == lfu:
            evicted, policy = lirs, -1
        elif policy == 0:
            evicted = lirs
        else:
            evicted = lfu

        self.lirs.delete(evicted, policy == 1, avoid_removal)
        x = self.lfu[evicted]
        del self.lfu[evicted]

        self.pollution.remove(evicted)

        self.addToHistory(x, policy)

        return evicted, policy

    # Add Entry to history dictated by policy
    # policy: 0, Add Entry to LRU History
    #         1, Add Entry to LFU History
    #        -1, Do not add Entry to any History
    def addToHistory(self, x, policy):
        if policy != 1:
            return
        if len(self.lfu_hist) == self.history_size:
            evicted = self.getLRU(self.lfu_hist)
            del self.lfu_hist[evicted.oblock]
        self.lfu_hist[x.oblock] = x

    # Add Entry to cache with given frequency
    def addToCache(self, oblock, freq, from_lfu_history):
        x = self.ALeCaR6_Entry(oblock, freq + 1, self.time)
        self.lirs.addToCache(oblock, from_lfu_history, freq)

        self.lfu[oblock] = x

    # True if oblock is in cache (which LRU can represent)
    def __contains__(self, oblock):
        return oblock in self.lirs

    def cacheFull(self):
        return self.lirs.cacheFull()

    # Get the LRU item in the given DequeDict
    # NOTE: DequeDict can be: lru, lru_hist, or lfu_hist
    # NOTE: does *NOT* remove the LRU Entry from given DequeDict
    def getLRU(self, dequeDict):
        return dequeDict.first()

    # Get the LFU min item in the LFU (HeapDict)
    # NOTE: does *NOT* remove the LFU Entry from LFU
    def getHeapMin(self):
        return self.lfu.min()

    # Get the random eviction choice based on current weights
    def getChoice(self):
        return 0 if np.random.rand() < self.W[0] else 1

    # Adjust the weights based on the given rewards for LRU and LFU
    def adjustWeights(self, rewardLRU, rewardLFU):
        reward = np.array([rewardLRU, rewardLFU], dtype=np.float32)
        self.W = self.W * np.exp(self.learning_rate * reward)
        self.W = self.W / np.sum(self.W)

        if self.W[0] >= 0.99:
            self.W = np.array([0.99, 0.01], dtype=np.float32)
        elif self.W[1] >= 0.99:
            self.W = np.array([0.01, 0.99], dtype=np.float32)

    class LIRS():
        class CacheMetaData(object):
            def __init__(self):
                self.isLir = False
                self.isResident = False
                self.freq = 0
                self.isPrunned = False

        def __init__(self, size, hir_percent):
            self.size = size
            if self.size < 4:
                self.size = 4

            if hir_percent is not None:
                self.hirsRatio = hir_percent
            else:
                self.hirsRatio = 0.01

            self.hirsSize = int((float(self.size) * self.hirsRatio))
            if self.hirsSize < 2:
                self.hirsSize = 2
            self.maxLirSize = self.size - self.hirsSize
            self.numberHIRpages = 0
            self.numberLIRpages = 0
            self.SStack = OrderedDict()
            self.QStack = OrderedDict()
            self.nonresidentHIRsInS = OrderedDict()
            self.pageFault = True
            self.M = 0.75  # or 1

        def __contains__(self, page):
            if page in self.SStack:
                return self.SStack[page].isResident
            elif page in self.QStack:
                return True
            else:
                return False

        def cacheFull(self):
            return self.numberHIRpages + self.numberLIRpages == self.size

        def __getitem__(self, page):
            if page in self.SStack:
                return self.SStack[page]
            elif page in self.QStack:
                return self.QStack[page]
            else:
                raise "Key Error - Not Found %d" % page

        def pageHIT(self, page):
            assert self.numberHIRpages + self.numberLIRpages <= self.size
            assert len(self.QStack) <= self.hirsSize
            assert len(self.SStack) <= (1 + self.M) * self.size

            if page in self.SStack:
                if self.SStack[page].isLir:
                    self.hitInLir(page)
                else:
                    self.hitinHIR(page)
            elif page in self.QStack:
                self.hitInQ(page)
            else:
                raise KeyError("Page not in cache")

        def hitInLir(self, page):
            assert page not in self.nonresidentHIRsInS
            assert page not in self.QStack

            data = self.SStack[page]
            data.freq += 1

            firstKey = next(iter(self.SStack))

            del self.SStack[page]
            self.SStack[page] = data

            if firstKey == page:
                self.pruneS(None)

        def hitinHIR(self, page):
            assert page in self.SStack
            assert page in self.QStack
            assert page not in self.nonresidentHIRsInS
            assert self.SStack[page] is self.QStack[page]

            firstLir = next(iter(self.SStack))
            firstLirData = self.SStack[firstLir]
            assert firstLirData.isLir == True
            assert firstLirData.isResident == True
            assert firstLir not in self.QStack
            assert firstLir not in self.nonresidentHIRsInS

            del self.QStack[page]
            self.numberHIRpages -= 1

            data = self.SStack[page]
            del self.SStack[page]

            assert self.numberLIRpages <= self.maxLirSize

            data.freq += 1
            data.isLir = True

            assert data.isResident == True

            self.numberLIRpages += 1
            self.SStack[page] = data

            if self.numberLIRpages > self.maxLirSize:
                self.numberLIRpages -= 1
                self.LIRtoHIR()
                self.numberHIRpages += 1
                self.pruneS(None)
                assert self.numberLIRpages == self.maxLirSize

        def LIRtoHIR(self):
            firstLirkey = next(iter(self.SStack))
            firsLirData = self.SStack[firstLirkey]

            assert firsLirData.isLir == True
            assert firsLirData.isResident == True

            del self.SStack[firstLirkey]

            assert len(self.QStack) < self.hirsSize
            assert self.numberHIRpages < self.hirsSize
            firsLirData.isLir = False
            self.QStack[firstLirkey] = firsLirData

        def hitInQ(self, page):
            assert page not in self.nonresidentHIRsInS
            assert self.QStack[page].isLir == False
            assert self.QStack[page].isResident == True

            data = self.QStack[page]
            data.freq += 1

            del self.QStack[page]

            self.QStack[page] = data
            self.SStack[page] = data

        def addToCache(self, page, from_lfu, saved_freq):
            assert self.numberHIRpages + self.numberLIRpages < self.size
            assert len(self.QStack) == self.numberHIRpages
            assert len(self.SStack) <= (1 + self.M) * self.size
            assert (self.numberLIRpages <
                    self.maxLirSize) or (self.numberHIRpages < self.hirsSize)
            assert page not in self

            if (self.numberLIRpages < self.maxLirSize) and (self.numberHIRpages
                                                            < self.hirsSize):
                assert self.numberHIRpages == 0

            if from_lfu:
                self.promotionFromLFU(page, saved_freq)
            else:
                if page in self.nonresidentHIRsInS:
                    assert page in self.SStack

                if page in self.SStack:
                    data = self.SStack[page]
                    assert data.isLir == False
                    assert data.isResident == False
                    assert data.isPrunned == False
                    assert page in self.nonresidentHIRsInS
                    assert self.nonresidentHIRsInS[page] is self.SStack[page]

                    self.hitInHistory(page)
                else:
                    assert page not in self.QStack
                    self.fullMiss(page, saved_freq)

        def hitInHistory(self, page):
            if self.numberLIRpages == self.maxLirSize:
                assert self.numberHIRpages < self.hirsSize
                assert len(self.QStack) < self.hirsSize
                assert page not in self.QStack

                data = self.SStack[page]

                del self.SStack[page]
                del self.nonresidentHIRsInS[page]

                data.freq += 1
                data.isLir = True

                assert data.isResident == False
                data.isResident = True

                self.SStack[page] = data

                self.LIRtoHIR()
                self.numberHIRpages += 1
                self.pruneS(None)
            else:
                assert page not in self.QStack
                data = self.SStack[page]
                del self.SStack[page]
                del self.nonresidentHIRsInS[page]

                self.numberLIRpages += 1
                data.freq += 1
                data.isLir = True

                assert data.isResident == False
                data.isResident = True

                self.SStack[page] = data

        def fullMiss(self, page, saved_freq):
            assert self.numberHIRpages == len(self.QStack)
            data = self.CacheMetaData()
            data.isResident = True
            data.freq = saved_freq + 1

            if self.numberLIRpages == self.maxLirSize:
                data.isLir = False
                self.SStack[page] = data
                self.QStack[page] = data
                self.numberHIRpages += 1
            else:
                if self.numberHIRpages == 0:
                    data.isLir = True
                    self.SStack[page] = data
                    self.numberLIRpages += 1
                else:
                    # Arbitrary Case
                    if self.numberHIRpages == self.hirsSize:
                        lastHirKey = next(iter(self.QStack))
                        lastHirData = self.QStack[lastHirKey]

                        assert lastHirKey not in self.nonresidentHIRsInS
                        assert lastHirData.isLir == False
                        assert lastHirData.isResident == True

                        if lastHirKey in self.SStack:
                            self.forceHIRtoLIR()
                            data.isLir = False
                            self.SStack[page] = data
                            self.QStack[page] = data
                            self.numberHIRpages += 1
                        else:
                            self.putAtTheBottonOfS(page, saved_freq)
                    else:
                        raise Exception("Should not happen")

        def forceHIRtoLIR(self):
            lastHirKey = next(iter(self.QStack))
            lastHirData = self.QStack[lastHirKey]

            assert lastHirData.isLir == False
            assert lastHirData.isResident == True
            assert lastHirKey not in self.nonresidentHIRsInS
            del self.QStack[lastHirKey]

            lastHirData.isLir = True
            lastHirData.isResident = True

            assert self.SStack[lastHirKey] is lastHirData

            if lastHirKey in self.SStack:
                del self.SStack[lastHirKey]

            self.SStack[lastHirKey] = lastHirData
            self.SStack.move_to_end(lastHirKey, last=False)

            self.numberLIRpages += 1
            self.numberHIRpages -= 1

        def putAtTheBottonOfS(self, page, saved_freq):
            data = self.CacheMetaData()
            data.isResident = True
            data.freq = saved_freq + 1
            data.isLir = True

            self.SStack[page] = data
            self.SStack.move_to_end(page, last=False)
            self.numberLIRpages += 1

        def promotionFromLFU(self, page, saved_freq):
            assert page not in self.nonresidentHIRsInS

            data = self.CacheMetaData()
            data.freq = saved_freq + 1
            data.isLir = True
            data.isResident = True

            if self.numberLIRpages == self.maxLirSize:
                assert self.numberHIRpages == len(self.QStack)
                assert len(self.QStack) < self.hirsSize
                assert page not in self.QStack
                assert page not in self.SStack

                self.SStack[page] = data

                self.LIRtoHIR()
                self.numberHIRpages += 1
                self.pruneS(None)
            else:
                assert self.numberHIRpages == 0 or self.numberHIRpages == self.hirsSize

                self.SStack[page] = data
                self.numberLIRpages += 1
                # bruh
                pass

        def delete(self, page, lfu_deletion, avoid_removal):
            assert self.numberLIRpages == self.maxLirSize and self.numberHIRpages == self.hirsSize
            assert self.numberHIRpages + self.numberLIRpages == self.size
            assert len(self.SStack) <= (1 + self.M) * self.size
            assert page in self
            assert page not in self.nonresidentHIRsInS

            if lfu_deletion:
                if page in self.SStack:
                    if self.SStack[page].isLir:
                        return self.deleteLirPageByLFU(page, avoid_removal)
                    else:
                        return self.deleteHirPageByLFU(page)
                elif page in self.QStack:
                    return self.deleleteHirQByLFU(page)
                else:
                    raise Exception("Page not in self - LFU")
            else:
                firstQkey = next(iter(self.QStack))
                assert firstQkey == page

                if page in self.SStack:
                    assert self.SStack[page].isLir == False
                    return self.deleteHirInS(page, avoid_removal)
                elif page in self.QStack:
                    return self.deleteHirInQ(page)
                else:
                    raise Exception("Page not in self - LIRS")

        def deleteLirPageByLFU(self, page, avoid_removal):
            data = self.SStack[page]
            freq = data.freq

            self.numberLIRpages -= 1

            firstKey = next(iter(self.SStack))
            firstData = self.SStack[firstKey]

            assert firstData.isLir == True
            assert firstData.isResident == True
            assert firstKey not in self.QStack
            assert firstKey not in self.nonresidentHIRsInS

            del self.SStack[page]

            if firstKey == page:
                # Bro be careful with the pruns
                self.pruneS(avoid_removal)

            assert freq > 0
            return freq

        def deleteHirPageByLFU(self, page):
            assert page in self.QStack

            data = self.SStack[page]

            assert data.isLir == False
            assert data.isResident == True

            freq = data.freq

            assert self.QStack[page].freq == freq
            assert freq > 0

            del self.SStack[page]
            del self.QStack[page]

            self.numberHIRpages -= 1

            return freq

        def deleleteHirQByLFU(self, page):
            data = self.QStack[page]

            assert data.isLir == False
            assert data.isResident == True

            freq = data.freq
            assert freq > 0

            del self.QStack[page]
            self.numberHIRpages -= 1

            return freq

        def deleteHirInS(self, page, avoid_removal):
            data = self.SStack[page]

            assert data.isLir == False
            assert data.isResident == True
            assert page in self.QStack

            del self.QStack[page]
            self.numberHIRpages -= 1

            data.isResident = False

            assert self.SStack[page].isResident == False

            self.nonresidentHIRsInS[page] = data

            if len(self.nonresidentHIRsInS) > int(float(self.size) * self.M):
                iterObj = iter(self.nonresidentHIRsInS)
                firstNonRed = next(iterObj)

                if firstNonRed == avoid_removal:
                    firstNonRed = next(iterObj)
                dataNonRed = self.nonresidentHIRsInS[firstNonRed]
                del self.nonresidentHIRsInS[firstNonRed]

                if dataNonRed.isPrunned:
                    assert firstNonRed not in self.SStack
                else:
                    del self.SStack[firstNonRed]

                assert len(self.nonresidentHIRsInS) == int(
                    float(self.size) * self.M)

            freq = data.freq

            assert freq > 0

            return freq

        def deleteHirInQ(self, page):
            data = self.QStack[page]

            assert data.isLir == False
            assert data.isResident == True
            assert page not in self.nonresidentHIRsInS

            del self.QStack[page]
            self.numberHIRpages -= 1

            freq = data.freq

            assert freq > 0

            return freq

        def pruneS(self, avoid_removal):
            keys_to_delete = []
            avoid_removal_data = None
            for key in self.SStack:
                data = self.SStack[key]

                if data.isLir:
                    break

                if key == avoid_removal:
                    avoid_removal_data = self.SStack[avoid_removal]
                    keys_to_delete.append(avoid_removal)
                    continue
                keys_to_delete.append(key)

                if key in self.nonresidentHIRsInS:
                    self.nonresidentHIRsInS[key].isPrunned = True
                else:
                    assert key in self.QStack

            for key in keys_to_delete:
                del self.SStack[key]
            if avoid_removal_data:
                self.SStack[avoid_removal] = avoid_removal_data
