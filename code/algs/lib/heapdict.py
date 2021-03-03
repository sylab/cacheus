# Keep a dictionary that has items ordered in a heap structure
class HeapDict:
    # Entry that holds the key, value and current index in the heap
    # Index tracking helps find where the Entry is in the heap quickly
    class HeapEntry:
        def __init__(self, key, value):
            self.key = key
            self.value = value
            self.index = -1

        # Minimal comparisons needed for heap
        def __lt__(self, other):
            return self.value < other.value

        # Helpful for debugging, print key, value and index
        def __repr__(self):
            return "(k={}, v={}, i={})".format(self.key, self.value,
                                               self.index)

    def __init__(self):
        self.htbl = {}
        self.heap = []

    # Helpful for debugging, print out entire heap
    def __repr__(self):
        return "<HeapDict({})>".format(self.heap)

    def __contains__(self, key):
        return key in self.htbl

    def __len__(self):
        return len(self.heap)

    # value = heapDict[key]
    def __getitem__(self, key):
        return self.htbl[key].value

    # heapDict[key] = value
    # NOTE: pushes a new item, updates an old item in the heap
    def __setitem__(self, key, value):
        if key in self.htbl:
            self.__update(key, value)
        else:
            self.__push(key, value)

    # del heapDict[key]
    # Just removes the item completely as expected
    def __delitem__(self, key):
        self.__remove(key)

    # Gets the min of the heap
    # NOTE: can be None if there is nothing in the heap
    # NOTE: does *NOT* remove min from the heap
    def min(self):
        if self.heap:
            return self.heap[0].value
        return None

    # Removes and returns the min of the heap
    # NOTE: heap cannot be empty
    def popMin(self):
        assert (self.heap)

        heapMin = self.heap[0]
        self.__remove(heapMin.key)
        return heapMin.value

    # swap where two entries are in the heap
    def __swap(self, entryA, entryB):
        entryA.index, entryB.index = entryB.index, entryA.index
        self.heap[entryA.index] = entryA
        self.heap[entryB.index] = entryB

    # get the parent of the entry at the given index
    # NOTE: reminder that heap is an abstraction of a balanced tree
    def __parent(self, index):
        parent_index = (index - 1) // 2
        return self.heap[parent_index]

    # get the child to the left of the entry at the given index
    # NOTE: reminder that heap is an abstraction of a balanced tree
    def __childLeft(self, index):
        left_index = (2 * index) + 1
        if left_index < len(self.heap):
            return self.heap[left_index]
        return None

    # get the child to the right of the entry at the given index
    # NOTE: reminder that heap is an abstraction of a balanced tree
    def __childRight(self, index):
        right_index = (2 * index) + 2
        if right_index < len(self.heap):
            return self.heap[right_index]
        return None

    # While heapupify is not common, it just means to move the entry
    # "upward" should its value have decreased. Often called decrease_key.
    def __heapupify(self, index):
        parent = self.__parent(index)
        entry = self.heap[index]

        while entry.index > 0 and entry.value < parent.value:
            self.__swap(entry, parent)
            parent = self.__parent(entry.index)

    # Move the entry at the given index down the heap if needed
    def __heapify(self, index):
        if index >= len(self.heap):
            return

        entry = self.heap[index]
        while entry.index < len(self.heap):
            # NOTE: due to how min() does not work well with None, we must
            #       at the very least check and replace Nones with the current
            #       entry to avoid problems
            childLeft = self.__childLeft(entry.index) if self.__childLeft(
                entry.index) else entry
            childRight = self.__childRight(entry.index) if self.__childRight(
                entry.index) else entry
            minEntry = min(entry, childLeft, childRight)

            if minEntry is not entry:
                self.__swap(entry, minEntry)
            else:
                break

    # Remove the entry with the given key completely from HeapDict
    def __remove(self, key):
        # O(logn)
        assert (key in self.htbl)

        entry = self.htbl[key]
        last = self.heap[-1]

        self.__swap(entry, last)
        del self.heap[-1]

        if entry is not last:
            self.__heapupify(last.index)
            self.__heapify(last.index)

        del self.htbl[key]

    # Insert a new item into the HeapDict
    def __push(self, key, value):
        # O(logn)
        assert (key not in self.htbl)

        entry = self.HeapEntry(key, value)
        self.htbl[key] = entry

        entry.index = len(self.heap)
        self.heap.append(entry)
        self.__heapupify(entry.index)

    # Update an item that currently exists in the HeapDict
    def __update(self, key, value):
        # normal update is O(logn)
        # remove + push is 2*O(logn), just do this for simplicity
        entry = self.htbl[key]
        entry.value = value
        self.__heapupify(entry.index)
        self.__heapify(entry.index)


# Some quick test code to make sure this works
if __name__ == '__main__':
    import heapq
    import random

    for _ in range(1000):
        hd = HeapDict()
        hq = []

        ordered_l = [i + 1 for i in range(1000)]
        shuffled_l = ordered_l[:]
        random.shuffle(shuffled_l)

        # with how heapdict works, let's keep a lookup for the keys used
        key_map = {}

        # Test push and min order
        for key, e in enumerate(shuffled_l):
            hd[key] = e
            heapq.heappush(hq, e)
            key_map[e] = key

        for e in ordered_l:
            f = hd.popMin()
            g = heapq.heappop(hq)
            assert (e == f == g)

        # Test how random removes/updates affects min order
        for e in shuffled_l:
            hd[key_map[e]] = e
            heapq.heappush(hq, e)

        modified_l = ordered_l[:]
        random_remove = random.sample(ordered_l, len(ordered_l) // 4)

        for e in random_remove:
            modified_l.remove(e)
            del hd[key_map[e]]
            hq.remove(e)

        # random update
        random_modify = random.sample(modified_l, len(modified_l) // 4)
        random_new_value = random_modify[:]
        random.shuffle(random_new_value)

        for e, v in zip(random_modify, random_new_value):
            hd[key_map[e]] = v
            hq.remove(e)
            heapq.heapify(hq)
            heapq.heappush(hq, v)

        for e in modified_l:
            f = hd.popMin()
            g = heapq.heappop(hq)
            assert (e == f == g)
