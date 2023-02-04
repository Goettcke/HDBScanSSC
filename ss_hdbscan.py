import hdbscan
from copy import copy
import numpy as np

from super_heap import SuperHeap, NodeObject

class HDBScanSSC:

    def __init__(self,min_cluster_size=5):
        self.X = None
        self.y = None
        self.transduction_ = []
        self.min_cluster_size = min_cluster_size
        self.metric = "euclidean"
        self.gen_min_span_tree=True
        self.approx_min_span_tree=False
        self.algorithm = "generic"

    def fit(self, X, y):
        """
        Fits the data and performs the label expansion

        Keyword arguments :
        X -- 2d numpy array containing the data to be fitted
        y -- 1d array containing a list of labels for each point in the data. Use -1 for unlabeled points
        """
        self.X = X
        self.y = y
        self.transduction_ = copy(self.y)
        self._transduce()


    def _build_super_heap(self, mstr):
        """ Builds a Superheap object from a minimum spanning tree.
        In this case it will be the MSTr generated by HDBScan*

        Keyword arguments :
        mstr -- A networkx minimum spanning tree

        Returns :
        Superheap object
        """
        assert len(self.X) > 0, "Fit X before transduction"
        l = [mstr.edges[x,y]['weight'] for x,y in mstr.edges]
        sh = SuperHeap(maxsize=len(self.X)+1)

        #Determine the unlabeled points.
        for index in range(len(self.X)):
            for neighbor_index in mstr.neighbors(index):  # For some reason the neighbors indices are now floats
                neighbor_index = int(neighbor_index)  # Converting back to int
                if self.transduction_[neighbor_index] != -1 and self.transduction_[index] == -1:  #Meaning we have a labeled neighbor, and the point itself is unlabeled
                    no = NodeObject(index)
                    edge = (neighbor_index, index)  # labeled point to unlabeled
                    sh.insert({'key': -mstr.edges[neighbor_index, index]['weight'], 'node': no, 'edge': edge, "path":[0]})  # the minus - in front of the key, is because superheap is a maxheap, and we want a min heap in this case.
        return sh


    def _transduce(self):
        """
        Transduce performs the label expansion from labeled points to the unlabeled points
        """
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size,
                                    gen_min_span_tree=self.gen_min_span_tree,
                                    approx_min_span_tree=self.approx_min_span_tree,
                                    algorithm=self.algorithm)
        clusterer.fit(self.X)
        mstr = clusterer.minimum_spanning_tree_.to_networkx()

        sh = self._build_super_heap(mstr)
        # First we pop an element from the queue
        while sh.size > 0:
            popped_element = sh.extractMax()
            elements = [popped_element]

            # Add all ties to elements
            while popped_element['key'] == sh.root()['key'] and sh.size > 0 and popped_element['node'].index != sh.root()['node'].index:
                elements.append(sh.extractMax())


            # Handle tie-breaking in case there are any
            if len(elements) > 1:
                popped_element = None

                path_list = [e['path'] for e in elements] # path list is a list of the individuel weights on the path to a labeled point

                #Extract the largest edge from each path
                # The np.random.choice(np flatnonzero.. ensures, that in the case there is a tie within the smallest largest edge weights we pick a random, and not just the first one, which would be the case with the cleaner np.argmin
                largest_edges = np.array([np.argmax(p) for p in path_list])
                smallest_largest_edge_index = np.random.choice(np.flatnonzero(largest_edges==largest_edges.min()))

                popped_element = elements[smallest_largest_edge_index] # extract the element with the smallest largest edge

                elements.remove(popped_element)
                for e in elements:
                    sh.insert(e)
                elements = [] # Important otherwise if we were in a situation where length of elements was 2, and we remove 1, then the elif will trigger

                neighbor_index = popped_element['edge'][1]
                self.transduction_[neighbor_index] = self.transduction_[popped_element['edge'][0]]  # So we label the unlabeled neighbor which has the minimum mstr distance with the label of its neighbor

            elif len(elements) == 1:
                neighbor_index = popped_element['edge'][1]
                self.transduction_[neighbor_index] = self.transduction_[popped_element['edge'][0]]  # So we label the unlabeled neighbor which has the minimum mstr distance with the label of its neighbor

            # Add all the neighbors of the now labeled point, to the heap, if they are unlabeled.
            for n_neighbor_index in mstr.neighbors(neighbor_index):
                n_neighbor_index = int(n_neighbor_index)
                if self.transduction_[n_neighbor_index] == -1 and (neighbor_index, n_neighbor_index) not in [x['edge'] for x in sh.Heap if x['node'] != None]:
                    no = NodeObject(n_neighbor_index)
                    edge = (neighbor_index, n_neighbor_index) #labeled point to unlabeled
                    key = -mstr.edges[neighbor_index, n_neighbor_index]['weight']
                    path = popped_element['path'] + [key]
                    sh.insert({'key': key, 'node': no, 'edge': edge, 'path': path})

    def get_params(self):
        return {'min_cluster_size': self.min_cluster_size}

    def set_params(self, **params):
        """Default set params method, copied from scikit-learn

        **params -- Dict of parameters
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params()

        nested_params = defaultdict(dict)
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self
