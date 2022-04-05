# Priority Queue Implementation by Noah Walker
# Directly adapted from the source code of Project 3.
# Major changes in this version:
#   - Removed UAPQ/BHPQ (simpler queues)
#   - Special comparison functions are removed.
#     (We now use Python's infinity instead of -1).
#   - Implemented Magic PQ: Partial BHPQ with three changes:
#     - a unified key function.
#     - no lookup table (saves space)
#     - no decreasekey as it is not necessary to the algorithm
import math

class MagicPriorityQueue:
    def __init__(self, costFn):
        # Time: O(1), Space: O(1).
        # Input: a lambda function for generating the key on
        #        provided elements.
        # TODO: Consider caching the cost result, as it is probably deterministic.
        self.costFn = costFn

    # parent, left_child, right_child:
    #   Time: O(1), basic mathematical operations.
    #   Space: O(1), single integer input with constant number of operations.
    def parent(index):
        return max(0, ((index+1) // 2) - 1)

    def left_child(index):
        return 2*(index + 1) - 1

    def right_child(index):
        return 2*(index + 1)

    def makequeue(self, elements):
        # Time: O(n log n), running insert (O(log n)) exactly n times.
        # Space: O(n) due to the creation and population of self.back.
        self.back = []
        for element in elements:
            self.insert(element)

    def insert(self, element):
        # Time: O(log n) by virtue of bubble (which is O(log n)).
        # Space: O(1), as we process only one element.
        #        O(n) if we consider self.back as part of insert's space complexity.
        self.back.append(element)
        self.bubble(self.length() - 1)

    def deletemin(self):
        # Time: O(log n). We traverse pairs of children in the binary heap.
        #       This is up to 2log n operations, as there are log n levels.
        # Space: O(n) considering self.back. O(1) otherwise.
        if self.length() == 1:
            # Empty the queue if it has only one element.
            return self.back.pop(0)
        # Shuffle the last node in the tree to the top.
        to_return = self.back[0]
        self.back[0] = self.back.pop(-1)
        heap_size = len(self.back)
        # These three variables are indices to track which elements we may compare to.
        current_element_location = 0
        left_child = MagicPriorityQueue.left_child(current_element_location)
        right_child = MagicPriorityQueue.right_child(current_element_location)
        # Time: O(log n) for this loop, as the loop will break once the indices for the
        #       children exceed the size of the heap. This is log n, especially as we
        #       note that left_child and right_child are always at least twice their parent index.
        while True:
            # Using 2-element array as tuples are immutable in Python. Left, right distance.
            # Selecting infinity here as the parent cannot bubble down beneath infinity.
            children = [math.inf, math.inf]
            # For both children, if they exist in the heap, consider their distance.
            if left_child < heap_size:
                children[0] = self.costFn(self.back[left_child])
            if right_child < heap_size:
                children[1] = self.costFn(self.back[right_child])
            # If the left child is the lowest and the parent is greater, swap.
            if (children[0] <= children[1]) and self.costFn(self.back[current_element_location]) > children[0]:
                self.magicswap(current_element_location, left_child)
                # Update all indices for next loop.
                current_element_location = left_child
                left_child = MagicPriorityQueue.left_child(current_element_location)
                right_child = MagicPriorityQueue.right_child(current_element_location)
                continue
            # Otherwise, if the right child is the lowest and the parent is greater, swap.
            elif (children[1] <= children[0]) and self.costFn(self.back[current_element_location]) > children[1]:
                self.magicswap(current_element_location, right_child)
                # Update all indices for next loop.
                current_element_location = right_child
                left_child = MagicPriorityQueue.left_child(current_element_location)
                right_child = MagicPriorityQueue.right_child(current_element_location)
                continue
            # Terminate if there are no further children to swap.
            return to_return

    def magicswap(self, indexA, indexB):
        # Swaps two elements in the priority queue, also updating the lookup table for decreasekey.
        # Time: O(1), as we perform 6 constant-time assignments and 8 index accesses to list elements.
        # Space: O(n) considering self.back. O(1) otherwise.
        temp = self.back[indexA]
        self.back[indexA] = self.back[indexB]
        self.back[indexB] = temp

    def bubble(self, index):
        # The element at the given index bubbles up the priority queue (insert and decreasekey).
        # Time: O(log n), since the parents are determined from repeatedly halving the given
        #       index, and all other operations are O(1) assuming basic math operations are O(1).
        # Space: O(n) considering self.back. O(1) otherwise.
        current_element_location = index
        parent = MagicPriorityQueue.parent(current_element_location)
        while self.costFn(self.back[current_element_location]) < self.costFn(self.back[parent]):
            self.magicswap(current_element_location, parent)
            current_element_location = parent
            parent = MagicPriorityQueue.parent(current_element_location)

    def length(self):
        return len(self.back)

