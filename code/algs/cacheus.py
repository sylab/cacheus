from .lib.dequedict import DequeDict
from .lib.heapdict import HeapDict
from .lib.pollutionator import Pollutionator
from .lib.visualizinator import Visualizinator
from .lib.optional_args import process_kwargs
from .lib.cacheop import CacheOp
import numpy as np


class Cacheus:
    ######################
    ## INTERNAL CLASSES ##
    ######################

    # Entry to track the page information
    class Cacheus_Entry:
        # TODO is_new logic to is not demoted
        def __init__(self, oblock, freq=1, time=0, is_new=True):
            self.oblock = oblock
            self.freq = freq
            self.time = time
            self.evicted_time = None
            self.is_demoted = False
            self.is_new = is_new

        # Minimal comparitors needed for HeapDict
        def __lt__(self, other):
            if self.freq == other.freq:
                return self.time > other.time
            return self.freq < other.freq

        # Useful for debugging
        def __repr__(self):
            return "(o={}, f={}, t={})".format(self.oblock, self.freq,
                                               self.time)

    # Adaptive learning rate of Cacheus
    # TODO consider an internal time instead of taking time as a parameter
    class Cacheus_Learning_Rate:
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
            self.hitrate_zero_count = 0
            self.hitrate_nega_count = 0

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

        #two stacks for the cache(s and q) to mark demoted items faster
        self.s = DequeDict()
        self.q = DequeDict()

        #lfu heap
        self.lfu = HeapDict()

        # Histories
        self.history_size = cache_size // 2
        self.lru_hist = DequeDict()
        self.lfu_hist = DequeDict()

        # Decision Weights Initilized
        self.initial_weight = 0.5

        # Learning Rate
        self.learning_rate = self.Cacheus_Learning_Rate(
            cache_size, **kwargs)

        process_kwargs(self,
                       kwargs,
                       acceptable_kws=['initial_weight', 'history_size'])

        # Decision Weights
        self.W = np.array([self.initial_weight, 1 - self.initial_weight],
                          dtype=np.float32)

        # Variables
        hirsRatio = 0.01
        self.q_limit = max(1, int((hirsRatio * self.cache_size) + 0.5))
        self.s_limit = self.cache_size - self.q_limit
        self.q_size = 0
        self.s_size = 0
        self.dem_count = 0
        self.nor_count = 0
        self.q_sizes = []

        # Visualize
        self.visual = Visualizinator(
            labels=['W_lru', 'W_lfu', 'hit-rate', 'q_size'],
            windowed_labels=['hit-rate'],
            window_size = window_size,
            **kwargs)
        # Pollution
        self.pollution = Pollutionator(cache_size, **kwargs)

    # True if oblock is in cache (which LRU can represent)
    def __contains__(self, oblock):
        return (oblock in self.s or oblock in self.q)

    def cacheFull(self):
        return len(self.s) + len(self.q) == self.cache_size

    #Hit in MRU portion of the cache
    def hitinS(self, oblock):
        x = self.s[oblock]
        x.time = self.time
        self.s[oblock] = x

        x.freq += 1
        self.lfu[oblock] = x

    #Hit in LRU portion of the cache
    def hitinQ(self, oblock):
        x = self.q[oblock]
        x.time = self.time

        x.freq += 1
        self.lfu[oblock] = x

        if x.is_demoted:
            self.adjustSize(True)
            x.is_demoted = False
            self.dem_count -= 1
        del self.q[x.oblock]
        self.q_size -= 1

        if self.s_size >= self.s_limit:
            y = self.s.popFirst()
            y.is_demoted = True
            self.dem_count += 1
            self.s_size -= 1
            self.q[y.oblock] = y
            self.q_size += 1

        self.s[x.oblock] = x
        self.s_size += 1

    # Add Entry to S with given frequency
    def addToS(self, oblock, freq, isNew=True):
        x = self.Cacheus_Entry(oblock, freq, self.time, isNew)
        self.s[oblock] = x
        self.lfu[oblock] = x
        self.s_size += 1

    def addToQ(self, oblock, freq, isNew=True):
        x = self.Cacheus_Entry(oblock, freq, self.time, isNew)
        self.q[oblock] = x
        self.lfu[oblock] = x
        self.q_size += 1

    # Add Entry to history dictated by policy
    # policy: 0, Add Entry to LRU History
    #         1, Add Entry to LFU History
    #        -1, Do not add Entry to any History
    def addToHistory(self, x, policy):
        # Use reference to policy_history to reduce redundant code
        policy_history = None
        if policy == 0:
            policy_history = self.lru_hist
            if x.is_new:
                self.nor_count += 1
        elif policy == 1:
            policy_history = self.lfu_hist
        elif policy == -1:
            return

        # Evict from history is it is full
        if len(policy_history) == self.history_size:
            evicted = self.getLRU(policy_history)
            del policy_history[evicted.oblock]
	    #adding this part here ######
            if policy_history == self.lru_hist and evicted.is_new:
            	self.nor_count -= 1
        policy_history[x.oblock] = x

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

    # Evict an entry
    def evict(self):
        lru = self.getLRU(self.q)
        lfu = self.getHeapMin()

        evicted = lru
        policy = self.getChoice()

        # Since we're using Entry references, we use is to check
        # that the LRU and LFU Entries are the same Entry
        if lru is lfu:
            evicted, policy = lru, -1
        elif policy == 0:
            evicted = lru
            del self.q[evicted.oblock]
            self.q_size -= 1
        elif policy == 1:
            evicted = lfu
            if evicted.oblock in self.s:
                del self.s[evicted.oblock]
                self.s_size -= 1
            elif evicted.oblock in self.q:
                del self.q[evicted.oblock]
                self.q_size -= 1

        if evicted.is_demoted:
            self.dem_count -= 1
            evicted.is_demoted = False
		
        if policy == -1:
            del self.q[evicted.oblock]
            self.q_size -= 1

        del self.lfu[evicted.oblock]
        evicted.evicted_time = self.time
        self.pollution.remove(evicted.oblock)

        self.addToHistory(evicted, policy)

        return evicted.oblock, policy

    # Adjust the weights based on the given rewards for LRU and LFU
    def adjustWeights(self, rewardLRU, rewardLFU):
        reward = np.array([rewardLRU, rewardLFU], dtype=np.float32)
        self.W = self.W * np.exp(self.learning_rate * reward)
        self.W = self.W / np.sum(self.W)

        if self.W[0] >= 0.99:
            self.W = np.array([0.99, 0.01], dtype=np.float32)
        elif self.W[1] >= 0.99:
            self.W = np.array([0.01, 0.99], dtype=np.float32)

    def adjustSize(self, hit_in_Q):
        if hit_in_Q:
            dem_count = max(1, self.dem_count)
            self.s_limit = min(
                self.cache_size - 1, self.s_limit +
                max(1, int((self.nor_count / dem_count) + 0.5)))
            self.q_limit = self.cache_size - self.s_limit
        else:
            nor_count = max(1, self.nor_count)
            self.q_limit = min(
                self.cache_size - 1, self.q_limit +
                max(1, int((self.dem_count / nor_count) + 0.5)))
            self.s_limit = self.cache_size - self.q_limit

    def hitinLRUHist(self, oblock):
        evicted = None

        entry = self.lru_hist[oblock]
        freq = entry.freq + 1
        del self.lru_hist[oblock]
        if entry.is_new:
            self.nor_count -= 1
            entry.is_new = False
            self.adjustSize(False)
        self.adjustWeights(-1, 0)

        if (self.s_size + self.q_size) >= self.cache_size:
            evicted, policy = self.evict()

        self.addToS(entry.oblock, entry.freq, isNew=False)
        self.limitStack()

        return evicted

    def hitinLFUHist(self, oblock):
        evicted = None

        entry = self.lfu_hist[oblock]
        freq = entry.freq + 1
        del self.lfu_hist[oblock]
        self.adjustWeights(0, -1)

        if (self.s_size + self.q_size) >= self.cache_size:
            evicted, policy = self.evict()

        self.addToS(entry.oblock, entry.freq, isNew=False)
        self.limitStack()

        return evicted

    def limitStack(self):
        while self.s_size >= self.s_limit:
            #mark demoted
            demoted = self.s.popFirst()
            self.s_size -= 1

            demoted.is_demoted = True
            self.dem_count += 1

            self.q[demoted.oblock] = demoted
            self.q_size += 1

    # Cache Miss
    def miss(self, oblock):
        evicted = None

        freq = 1

        if self.s_size < self.s_limit and self.q_size == 0:
            self.addToS(oblock, freq, isNew=False)
        elif self.s_size + self.q_size < self.cache_size and self.q_size < self.q_limit:
            self.addToQ(oblock, freq, isNew=False)
        else:
            # NOTE: does this belong before the previous if-else block?
            # TODO Eviction of multiple blocks in Q ?
            # If the cache is full, evict
            if (self.s_size + self.q_size) >= self.cache_size:
                evicted, policy = self.evict()

            # NOTE: it's possible that these should be in an else instead
            #Filling up the cache
            self.addToQ(oblock, freq, isNew=True)
            self.limitStack()

        return evicted

    # Process and access request for the given oblock

    def request(self, oblock, ts):
        miss = False
        evicted = None

        self.time += 1

        self.visual.add({
            'W_lru': (self.time, self.W[0], ts),
            'W_lfu': (self.time, self.W[1], ts),
            'q_size': (self.time, self.q_size, ts)
        })

        self.learning_rate.update(self.time)

        if oblock in self.s:
            self.hitinS(oblock)
        elif oblock in self.q:
            self.hitinQ(oblock)
        elif oblock in self.lru_hist:
            miss = True
            evicted = self.hitinLRUHist(oblock)
        elif oblock in self.lfu_hist:
            miss = True
            evicted = self.hitinLFUHist(oblock)
        else:
            miss = True
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
        if self.time % self.cache_size == 0:
            self.pollution.update(self.time)

        op = CacheOp.INSERT if miss else CacheOp.HIT

        return op, evicted

    def getQsize(self):
        x, y = zip(*self.visual.get('q_size'))
        return y
