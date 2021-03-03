# Keep a dictionary that has items ordered in a deque structure
class DequeDict:
    # Entry that holds the key, value
    class DequeEntry:
        def __init__(self, key, value):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None

        # Helpful for debugging, print key and value
        def __repr__(self):
            return "(k={}, v={})".format(self.key, self.value)

    def __init__(self):
        self.htbl = {}
        self.head = None
        self.tail = None

    # Helpful for debugging, print out entire Deque
    def __repr__(self):
        entries = []
        entry = self.head
        while entry:
            entries.append(entry)
            entry = entry.next
        return "<DequeDict({})>".format(entries)

    # Iterator
    def __iter__(self):
        self.current = self.head
        return self

    # Next for iterator
    def __next__(self):
        if self.current == None:
            raise StopIteration
        value = self.current.value
        self.current = self.current.next
        return value

    # For Python2
    next = __next__

    def __contains__(self, key):
        return key in self.htbl

    def __len__(self):
        return len(self.htbl)

    # value = dequeDict[key]
    def __getitem__(self, key):
        return self.htbl[key].value

    # dequeDict[key] = value
    # NOTE: pushes a new item, updates an old item in the Deque
    def __setitem__(self, key, value):
        if key in self.htbl:
            self.__update(key, value)
        else:
            self.__push(key, value)

    # del dequeDict[key]
    # Just removes the item completely as expected
    def __delitem__(self, key):
        self.__remove(key)

    # Get first item
    # NOTE: LRU since queue is FIFO
    def first(self):
        return self.head.value

    # Push (key, value) as first in deque
    def pushFirst(self, key, value):
        assert (key not in self.htbl)

        entry = self.DequeEntry(key, value)
        self.htbl[key] = entry

        headEntry = self.head
        if headEntry:
            headEntry.prev = entry
            entry.next = headEntry
        else:
            self.tail = entry
        self.head = entry

    # Remove and return first item
    # NOTE: LRU since queue is FIFO
    def popFirst(self):
        first = self.head
        self.__remove(first.key)
        return first.value

    # Get last item
    # NOTE: MRU since queue is FIFO
    def last(self):
        return self.tail.value

    # Remove and return last item
    # NOTE: MRU since queue is FIFO
    def popLast(self):
        last = self.tail
        self.__remove(last.key)
        return last.value

    # Remove the entry with the given key completely from DequeDict
    def __remove(self, key):
        assert (key in self.htbl)

        entry = self.htbl[key]
        prevEntry = entry.prev
        nextEntry = entry.next

        if prevEntry:
            prevEntry.next = nextEntry
        else:
            self.head = nextEntry

        if nextEntry:
            nextEntry.prev = prevEntry
        else:
            self.tail = prevEntry

        del self.htbl[key]

    # Insert a new item into the DequeDict
    def __push(self, key, value):
        assert (key not in self.htbl)

        entry = self.DequeEntry(key, value)
        self.htbl[key] = entry

        tailEntry = self.tail
        if tailEntry:
            tailEntry.next = entry
            entry.prev = tailEntry
        else:
            self.head = entry
        self.tail = entry

    # Update an item that currently exists in the DequeDict
    def __update(self, key, value):
        # Just remove and push since it should be very fast anyways
        self.__remove(key)
        self.__push(key, value)


# Some quick test code to make sure this works
if __name__ == '__main__':
    dd = DequeDict()

    l = [1, 2, 3, 4, 5, 6]
    for e in l:
        dd[len(l) - e] = e

    for e, f in zip(l, dd):
        assert (e == f)

    for e in l:
        f = dd.popFirst()
        assert (e == f)

    for e in l:
        dd[len(l) - e] = e

    for e in l[::-1]:
        f = dd.popLast()
        assert (e == f)

    for e in l:
        dd[len(l) - e] = e

    dd[3] = -1
    dd[5] = 7

    del dd[1]

    assert (dd.popFirst() == 2)
    assert (dd.popFirst() == 4)
    assert (dd.popFirst() == 6)
    assert (dd.popFirst() == -1)
    assert (dd.popFirst() == 7)
