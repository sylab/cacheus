from .lib.pollutionator import Pollutionator
from .lib.visualizinator import Visualizinator
from .lib.cacheop import CacheOp

class MIN:
    class MIN_Entry:
        def __init__(self, oblock, index):
            self.oblock = oblock
            self.index = index

        def __repr__(self):
            return "(o={})".format(self.oblock)

    class MIN_index:
        def __init__(self, index, count):
            self.index_ = index
            self.count = count

    def __init__(self, cache_size, window_size, **kwargs):
        self.cache_size = cache_size
        
        #set of blocks 
        self.request_lbn = {}
        #set of counters per index block
        self.request_index = {}
        
        #counters per index
        self.count_compl = 1
        self.curr = 0

        self.time = 0

        self.visual = Visualizinator(labels=['hit-rate'],
                                     windowed_labels=['hit-rate'],
                                     window_size=window_size,
                                     **kwargs)
        self.pollution = Pollutionator(cache_size, **kwargs)

    def CounterInc(self, index):
        if index in self.request_index:
            c = self.request_index[index]
            c.count += 1
        else:
            c = self.MIN_index(index, count=0)
            c.count += 1
            self.request_index[index] = c

    def CounterDec(self, index):
        if index in self.request_index:
            c = self.request_index[index]
        else:
            return
        c.count -= 1
        if c.count == 0:
            del self.request_index[index]
    
    def IndexCount(self, index):
        if index in self.request_index:
            c = self.request_index[index]
            return c.count
        return 0

    def request(self, oblock, ts):
        miss = True
        evicted = None
        self.time += 1

        counter = 0
        temp = 0
        
        if oblock in self.request_lbn:
            x = self.request_lbn[oblock]
        else:
            x = self.MIN_Entry(oblock, index=0)
            self.request_lbn[oblock] = x

        if (x.index < self.count_compl):
            self.curr += 1
            self.CounterDec(x.index)
            x.index = self.curr
            self.CounterInc(x.index)
            
            op = CacheOp.INSERT if miss else CacheOp.HIT
            self.visual.addWindow({'hit-rate': 0 if miss else 1}, self.time, ts)
            return op, evicted

        if (x.index == self.curr):
            miss = False
            op = CacheOp.INSERT if miss else CacheOp.HIT
            self.visual.addWindow({'hit-rate': 0 if miss else 1}, self.time, ts)
            return op, evicted
        
        if (x.index < self.curr and x.index >= self.count_compl):
            self.CounterDec(x.index)
            x.index = self.curr
            self.CounterInc(x.index)
            temp = self.curr
            counter = 0
            
            while (1):
                counter += self.IndexCount(temp)
                if (counter == self.cache_size or temp == self.count_compl):
                    self.count_compl = temp

                    miss = False
                    op = CacheOp.INSERT if miss else CacheOp.HIT
                    self.visual.addWindow({'hit-rate': 0 if miss else 1}, self.time, ts)
                    return op, evicted
            
                if counter < self.cache_size:
                    counter -= 1
                    temp -= 1
                    continue
 
            assert(False)
        
        miss = False
        op = CacheOp.INSERT if miss else CacheOp.HIT
        self.visual.addWindow({'hit-rate': 0 if miss else 1}, self.time, ts)
        return op, evicted
