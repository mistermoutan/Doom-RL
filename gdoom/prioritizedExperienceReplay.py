# Using https://gist.github.com/simoninithomas/d6adc6edb0a7f37d6323a5e3d2ab72ec#file-dueling-deep-q-learning-with-doom-double-dqns-and-prioritized-experience-replay-ipynb
from utils import *

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class SumTree(object):

    def __init__(self, capacity):
        self.data_pointer = 0
        self.capacity = capacity
        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        nbrLeaf = Sum_{i=0}ˆ{k} 2**i with 2**k=capacity
        """
        self.tree = np.zeros(2 * capacity - 1)
        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        '''
        Here we add our priority score in the sumtree leaf and add the experience in data
        '''
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data # NOTE: Transition?
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        '''
        Update the leaf priority score and propagate the change through tree
        '''
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        while tree_index != 0:
            tree_index = tree_index - 1 // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        '''
        Here we get the leaf_index, priority value of that leaf and experience associated with that index
        '''
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0
        while True:
            left_child_index = 2*parent_index + 1
            right_child_index = left_child_index + 1

            # Reach bottom.
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            # Downward search for higher priority node.
            else:

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[leaf_index]

    @property
    def total_priority(self):
        return self.tree[0]
    

class Memory(object):
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001
    
    absolute_error_upper = 1.  # clipped abs error


    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = SumTree(capacity)

    def push(self, *args):
        '''
        Store a new experience in our tree
        Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
        '''
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        self.tree.add(max_priority, Transition(*args))


    def sample(self, minibatch_size):
        # Create a sample array that will contains the minibatch
        memory_b = []

        b_idx, b_ISWeights = np.empty((minibatch_size,), dtype=np.int32), np.empty((minibatch_size, 1), dtype=np.float32)
        priority_segment = self.tree.total_priority / minibatch_size

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling]) 

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * minibatch_size) ** (-self.PER_b)

        for i in range(minibatch_size):

            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            
            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)


            #P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(minibatch_size * sampling_probabilities, -self.PER_b)/ max_weight

            b_idx[i]= index

            memory_b.append(data)

        return b_idx, memory_b, b_ISWeights


    def batch_update(self, tree_idx, abs_errors):
        '''
        Update the priorities on the tree
        '''
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

















