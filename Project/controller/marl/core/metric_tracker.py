from collections import defaultdict


class MetricTracker:
    def __init__(self):
        self.reset_tracker()
    
    def update(self, key, value):
        self.metrics[key] += value
        self.counts[key] += 1
    
    def get_average(self, key, default=""):
        if key not in self.metrics:
            return default
        
        return self.metrics[key] / self.counts[key]

    def reset_tracker(self):
        self.metrics = defaultdict(int)
        self.counts = defaultdict(int)