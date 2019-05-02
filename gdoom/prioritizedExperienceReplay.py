# Using https://gist.github.com/simoninithomas/d6adc6edb0a7f37d6323a5e3d2ab72ec#file-dueling-deep-q-learning-with-doom-double-dqns-and-prioritized-experience-replay-ipynb
from utils import *

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
    

