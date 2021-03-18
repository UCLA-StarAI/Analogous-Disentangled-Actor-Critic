
class RingBuffer():
    def __init__(self, max_len):
        self.max_len = max_len

        self.start = 0
        self.length = 0
        self.items = [None for _ in range(max_len)]

    def __getitem__(self, idx):
        if idx >= self.length or idx < -self.length:
            raise ValueError("Invalid index %d for RingBuffer with length %d." % (idx, self.length))

        if idx >= 0:
            return self.items[(self.start + idx) % self.max_len]
        else:
            return self.items[(self.start + self.length + 1 - idx) % self.max_len]

    def __setitem__(self, idx, v):
        if idx >= self.length or idx < -self.length:
            raise ValueError("Invalid index %d for RingBuffer with length %d." % (idx, self.length))

        if idx >= 0:
            self.items[(self.start + idx) % self.max_len] = v
        else:
            self.items[(self.start + self.length + 1 - idx) % self.max_len] = v

    def append(self, v):
        if self.length < self.max_len:
            self.items[self.length] = v
            self.length += 1
        else:
            self.items[self.start] = v
            self.start = (self.start + 1) % self.max_len

