import random

class QuestionIterator:
    def __init__(self, data_dict, max_epochs, sample_counts, base_category='medium'):
        self.original = data_dict
        self.max_epochs = max_epochs
        self.sample_counts = sample_counts
        self.base_category = base_category
        self.base_length = len(data_dict[self.base_category])
        self.current_epoch = 0
        self.total_steps = 0
        
        # Initialize pointers and current lists for each category
        self.category_pointers = {}  # {category: pointer}
        self.category_lists = {}     # {category: list}
        
        # Initialize all categories in sample_counts
        for category in self.sample_counts:
            assert category in self.original, f"Category '{category}' not found in data_dict."
            self.category_pointers[category] = 0
            self.category_lists[category] = list(self.original[category])
        
        # Initialize the first shuffle
        self._shuffle_all()

    def _shuffle_all(self):
        """Shuffles all categories using current_epoch as the seed."""
        seed = self.current_epoch
        random.seed(seed)
        for category in self.category_lists:
            original_list = list(self.original[category])
            random.shuffle(original_list)
            self.category_lists[category] = original_list
        # Reset pointers to 0 for all categories
        for category in self.category_pointers:
            self.category_pointers[category] = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_epoch >= self.max_epochs:
            raise StopIteration

        samples = []
        for category in self.sample_counts:
            count = self.sample_counts[category]
            ptr = self.category_pointers[category]
            lst = self.category_lists[category]
            length = len(lst)
            
            end = ptr + count
            if end <= length:
                selected = lst[ptr:end]
                new_ptr = end
            else:
                first_part = lst[ptr:]
                second_part = lst[0 : (end - length)]
                selected = first_part + second_part
                new_ptr = end % length
            
            samples.extend(selected)
            self.category_pointers[category] = new_ptr

        # Update total_steps based on base_category's count
        base_count = self.sample_counts[self.base_category]
        self.total_steps += base_count

        # Check if an epoch has finished
        if self.total_steps >= (self.current_epoch + 1) * self.base_length:
            self.current_epoch += 1
            if self.current_epoch < self.max_epochs:
                self._shuffle_all()
            else:
                # No more epochs, next call will stop
                pass

        # Check if we have exceeded max_epochs
        if self.current_epoch > self.max_epochs:
            raise StopIteration

        return samples
