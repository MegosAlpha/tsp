#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
    from PyQt6.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
from PriorityQueue import *
import tsp

# Augmented TSP Solution.
# Represents a (potentially partial) path, incl. costs
class BnBTSPSolution(TSPSolution):
    def __init__(self, route, costMatrix, lowerBound, cost = -1):
        self.costMatrix = costMatrix
        self.lowerBound = lowerBound
        self.route = route
        # Partial path costs must be manually calculated because
        # disconnected nodes will cause over-aggressive pruning.
        # (TSPSolution adds the connection to origin automatically)
        # Therefore, we do not call the superclass constructor.
        if cost != -1:
            self.cost = cost
        else:
            if len(route) == 1:
                self.cost = 0
            else:
                self.cost = sum([self.route[i].costTo(self.route[i+1]) for i in range(0, len(self.route)-1)])


    def reduceRowCostMatrix(costMatrix):
        # Simple row reduction that tracks the lower bound.
        # Time: O(n^2) from the main loop containing a linear search
        # Space: O(n^2) as we are dealing with the cost matrix.
        lb = 0
        # for every row of the cost matrix...
        for i in range(len(costMatrix)):
            # find the minimum (O(n))
            rowMin = min(costMatrix[i])
            if rowMin > 0 and rowMin != math.inf:
                # subtract the minimum to create a zero
                costMatrix[i] = [cost - rowMin for cost in costMatrix[i]]
                # add this minimum to the lower bound
                lb += rowMin
        return (lb, costMatrix)


    def clone(self):
        # Used to expand search states.
        # Time: O(n^2), as we must copy the cost matrix.
        # Space: O(n^2), since we allocate a copy of the cost matrix.
        return BnBTSPSolution(self.route[:], self.costMatrix.copy(), self.lowerBound, self.cost)


    def addCityToRoute(self, newCity):
        # Updates the state given that a new node was attached to it.
        # Time: O(n^2), since we transpose the cost matrix and reduce it.
        # Space: O(n^2), once again due to the cost matrix.
        # Add cost of the new path
        self.lowerBound += self.costMatrix[self.route[-1]._index][newCity._index]
        self.cost += self.route[-1].costTo(newCity)
        # Update the route itself
        self.route.append(newCity)
        # Early termination condition: path is impossible.
        # Occurs when no connection exists or when forming a premature loop
        if self.lowerBound == math.inf or self.cost == math.inf:
            return
        # Note: practical space optimization available by using two lists of deleted
        # elements and producing minor matrices instead of replacing rows & cols.
        # Last city in route: corresponding row replaced with infinity
        self.costMatrix[self.route[-2]._index] = [math.inf] * len(self.costMatrix)
        # New city: corresponding column replaced with infinity
        self.costMatrix = self.costMatrix.transpose()
        self.costMatrix[newCity._index] = [math.inf] * len(self.costMatrix)
        # Reduce the columns
        lb_col, cm = BnBTSPSolution.reduceRowCostMatrix(self.costMatrix)
        # Reduce the rows
        lb_row, cm = BnBTSPSolution.reduceRowCostMatrix(cm.transpose())
        # Update reduction costs
        self.lowerBound += lb_col + lb_row
        # Update remaining cost matrix
        self.costMatrix = cm


class TSPSolver:
    def __init__( self, gui_view ):
        self._scenario = None

    def setupWithScenario( self, scenario ):
        self._scenario = scenario

    def initializeReducedCostMatrix(self):
        # Time and Space: O(n^2) due to generating, transposing, reducing the cost matrix
        cities = self._scenario.getCities()
        # Generate the Cost Matrix (Time and Space: O(n^2))
        # The cost of city x to city y is costMatrix[x][y]
        costMatrix = np.array([[citySrc.costTo(cityDest) for cityDest in cities] for citySrc in cities])
        # Reduce the rows (O(n^2))
        lb, costMatrix = BnBTSPSolution.reduceRowCostMatrix(costMatrix)
        # Reduce the columns by reducing the rows of the transpose (O(n^2))
        lb_col, costMatrix_T = BnBTSPSolution.reduceRowCostMatrix(costMatrix.transpose())
        # Update cost matrix and lower bound
        lb += lb_col
        self.costs = costMatrix_T.transpose()
        self.lb = lb

    def initializeInfinityRegulatedCostMatrix(self):
        cities = self._scenario.getCities()
        transformInfinity = lambda x: x if x != math.inf else -1
        self.ircmatrix = np.array([[transformInfinity(citySrc.costTo(cityDest)) for cityDest in
        cities] for citySrc in cities])

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution,
        time spent to find solution, number of permutations tried during search, the
        solution found, and three null values for fields not used for this
        algorithm</returns>
    '''

    def defaultRandomTour( self, time_allowance=60.0 ):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time()-start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation( ncities )
            route = []
            # Now build the route using the random permutation
            for i in range( ncities ):
                route.append( cities[ perm[i] ] )
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results


    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this
        algorithm</returns>
    '''

    def greedy( self,time_allowance=60.0 ):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        # NOTE: This is a naive implementation of the greedy algorithm.
        # Will likely be reimplemented in the group project.
        # Currently used as a heuristic for the BSSF if it passes.
        remaining = cities[1:]
        route = [cities[0]]
        # Time: O(n^2) from the inner loop O(n)
        while len(remaining) > 0 and time.time()-start_time < time_allowance:
            minAdjacent = (route[-1].costTo(remaining[0]), 0)
            # Finding the minimum adjacent connection
            # Time: O(n) seeing as we try to connect to all remaining cities.
            for cityIdx in range(1, len(remaining)):
                cost = route[-1].costTo(remaining[cityIdx])
                if cost < minAdjacent[0]:
                    minAdjacent = (cost, cityIdx)
            # Greedily add this minimum on and proceed.
            # Notice that no backtracking / DFS-like functionality is included.
            # As a result, this algorithm can fail to provide a non-infinite path.
            route.append(remaining[minAdjacent[1]])
            remaining = remaining[:minAdjacent[1]] + remaining[minAdjacent[1]+1:]
        if len(remaining) == 0:
            foundTour = True
            bssf = TSPSolution(route)
            # Fall back to random instead of backtracking.
            if bssf._costOfRoute() == math.inf:
                return self.defaultRandomTour()
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results


    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints:
        max queue size, total number of states created, and number of pruned states.</returns>
    '''

    def branchAndBound( self, time_allowance=60.0 ):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundOptimalTour = False
        count = 0
        start_time = time.time()
        # Set up the cost matrix & reduce it
        self.initializeReducedCostMatrix()
        # BSSF initialization via naive greedy algorithm or random tour
        bssf = self.greedy()['soln']
        pruned = 0
        # Maximum queue size
        mqs = 1
        states = 1
        # Priority queue and cost function
        # Cost function is O(1), so it does not affect the big-O in the PQ impl
        pq = MagicPriorityQueue(lambda x: x.lowerBound - len(x.route)*self.lb)
        # create initial state (O(1) since it's a single element)
        pq.makequeue([BnBTSPSolution([cities[0]], self.costs, self.lb)])
        # Worst-case: only complete routes are pruned.
        # Time: O(n!) since every descendant route is assessed and added to the queue.
        #       n! outpaces n^3
        # Space: O(n!) for the same reason. n! outpaces n^2.
        while time.time()-start_time < time_allowance and not foundOptimalTour and len(pq.back)>0:
            # Update the maximum queue size
            if len(pq.back) > mqs:
                mqs = len(pq.back)
            # take the lowest cost node, which we will expand. O(log n) from PQ
            expand = pq.deletemin()
            # automatically prune worse partial paths and impossible paths (O(1))
            if (bssf != None and expand.cost >= bssf.cost) or expand.cost == math.inf:
                pruned += 1
                continue
            # Branching step with time O(n^3) (O(n) outer, O(n^2) inner)
            for city in cities:
                # Refuse to add some obvious bad states to reduce allocation (O(1))
                if city == expand.route[-1] or (len(expand.route) == ncities and city != expand.route[0]):
                    continue
                # Generate a new state corresponding to the chosen city
                states += 1
                # O(n^2) from clone, addCityToRoute
                subproblem = expand.clone()
                subproblem.addCityToRoute(city)
                if len(subproblem.route) == ncities + 1:
                    # Complete route found (entire section O(1))
                    count += 1
                    # Trim end of route to maintain functionality of enumerateEdges.
                    subproblem.route = subproblem.route[:-1]
                    # Early termination: absolute best possible route found.
                    if subproblem.cost == self.lb:
                        bssf = subproblem
                        foundOptimalTour = True
                        break
                    # Update BSSF...
                    if bssf == None or bssf.cost > subproblem.cost:
                        bssf = subproblem
                    else:
                        # ... or prune the completed but worse route
                        pruned += 1
                elif not ((bssf != None and subproblem.lowerBound >= bssf.cost) or subproblem.cost == math.inf or subproblem.lowerBound == math.inf):
                    # If the route is not explicitly worse, add it to the queue.
                    # O(log n) from the priority queue implementation.
                    pq.insert(subproblem)
                else:
                    # The route is pruned, as it is worse than the BSSF or impossible.
                    pruned += 1
        end_time = time.time()
        # Additional cleanup if time expires
        # May also be considered O(n!) in the worst-case,
        # seeing as it inherits whatever states are left above (up to all of the leaf states)
        while len(pq.back) > 0:
            expand = pq.deletemin()
            # prune & count worse partial paths and impossible paths (O(1))
            if (bssf != None and expand.cost >= bssf.cost) or expand.cost == math.inf:
                pruned += 1
        # Set due to the spec
        self._bssf = bssf
        results['cost'] = bssf.cost if bssf != None else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = mqs
        results['total'] = states
        results['pruned'] = pruned
        return results


    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number of solutions found during search, the
        best solution found.  You may use the other three field however you like.
        algorithm</returns>
    '''

    def fancy( self,time_allowance=60.0 ):
        # Held-Karp algorithm
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        count = 1
        start_time = time.time()
        # Set up the cost matrix
        self.initializeInfinityRegulatedCostMatrix();
        # Use convenient package that solves the problem
        route_idxs = tsp.solve_problem(self.ircmatrix, ncities);
        end_time = time.time()
        route = [cities[x] for x in route_idxs][:-1]
        bssf = TSPSolution(route)
        # Set due to the spec
        self._bssf = bssf
        results['cost'] = bssf.cost if bssf != None else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = (ncities - 1) * (2**ncities)
        results['pruned'] = None
        return results

