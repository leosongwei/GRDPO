import json

class QuestionIteratorMath:
    def __init__(self, jsonl_file_path: str, sample_counts: int, max_epochs: int = 1):
        # the sample file can be found at: GRDPO2/datasets/math/samples.jsonl
        pass
        # load jsonl file without shuffle

    def __iter__(self):
        return self
    
    def __next__(self):
        # if max_epochs not reached, loop from begin
        pass
        # 返回：List[Dict]，个数为sample_counts，挨个取就行