import json

class QuestionIteratorMath:
    def __init__(self, jsonl_file_path: str, sample_counts: int, max_epochs: int = 1):
        # the sample file can be found at: GRDPO2/datasets/math/samples.jsonl
        self.jsonl_file_path = jsonl_file_path
        self.sample_counts = sample_counts
        self.max_epochs = max_epochs
        self.current_epoch = 0
        self.current_position = 0
        
        # Load jsonl file without shuffle
        self.data = []
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        
        self.total_samples = len(self.data)

    def __iter__(self):
        return self
    
    def __next__(self):
        # if max_epochs not reached, loop from begin
        if self.current_epoch >= self.max_epochs:
            raise StopIteration
        
        # Check if we need to start a new epoch
        if self.current_position + self.sample_counts > self.total_samples:
            self.current_epoch += 1
            if self.current_epoch >= self.max_epochs:
                raise StopIteration
            self.current_position = 0
        
        # Get samples for this iteration
        start_pos = self.current_position
        end_pos = min(start_pos + self.sample_counts, self.total_samples)
        
        # Handle wrap-around if needed
        if end_pos - start_pos < self.sample_counts:
            # Need to wrap around to the beginning for remaining samples
            remaining = self.sample_counts - (end_pos - start_pos)
            samples = self.data[start_pos:end_pos] + self.data[0:remaining]
            self.current_position = remaining
            self.current_epoch += 1
            if self.current_epoch >= self.max_epochs:
                # This was the last batch, next call will stop
                pass
        else:
            samples = self.data[start_pos:end_pos]
            self.current_position = end_pos
            
            # Check if we've completed this epoch
            if self.current_position >= self.total_samples:
                self.current_epoch += 1
                self.current_position = 0

        return samples