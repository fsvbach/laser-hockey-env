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
        self.normalization_constant=0

    def add_transition(self, new_transition):
        
        if self.size == 0:
            # fill buffer with new transition till its full
            blank_buffer = [np.asarray(new_transition+[0], dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)
        # overwrites oldest transitions
        new_transition += [self.max_priority]
        self.normalization_constant += self.max_priority - self.transitions[self.current_idx, 5]
        self.transitions[self.current_idx,:] = np.asarray(new_transition, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size
        #print("normalization constant, priorities: ", self.normalization_constant, ", ", self.transitions[:self.size, 5])
        
    def update_priorities (self, indices, td_errors):
        old_priorities = np.sum(self.transitions[indices, 5])
        new_priorities = np.sum(td_errors) + td_errors.size * self.eps
        
        self.normalization_constant += new_priorities - old_priorities 
        self.transitions[indices, 5] = td_errors + self.eps
        max_td_error = np.max(td_errors)
        #print("td errors, max td error: ", td_errors, max_td_error)
        self.max_priority = np.max([max_td_error + self.eps, self.max_priority])

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size

        priorities = np.array(self.transitions[:self.size,5], dtype=float)/self.normalization_constant
        #print("priorities size: ", len(priorities))
        #print("transitions size: ", self.size)
        indices =np.random.choice(self.size, size=batch, replace=False, p=priorities)

        return self.transitions[indices,:], indices


