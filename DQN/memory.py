import numpy as np

# class to store transitions


class Memory():
    def __init__(self, max_size=100000):
        self.transitions = np.asarray([])
        self.eps = 0.0001
        self.max_priority = 1
        self.size = 0
        self.current_idx = 0
        self.max_size=max_size
        self.normalization_constant=self.max_size

    def add_transition(self, new_transition):
        new_transition += [self.max_priority]
        if self.size == 0:
            # fill buffer with new transition till its full
            blank_buffer = [np.asarray(new_transition, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)
        # overwrites oldest transitions
        self.transitions[self.current_idx,:] = np.asarray(new_transition, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size
        
    def update_priorities (self, indices, td_errors):
        old_priorities = np.sum(self.transitions[indices, 5])
        new_priorities = np.sum(td_errors) + td_errors.size * self.eps
        
        self.normalization_constant += new_priorities - old_priorities 
        self.transitions[indices, 5] = td_errors + self.eps
        self.max_priority = np.max(np.max(td_errors) + self.eps, self.max_priority)

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        indices =np.random.choice(range(self.size), size=batch, replace=False, p=self.transitions[:,5]/self.normalization_constant)
        return self.transitions[indices,:], indices


