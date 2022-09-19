"""
wl_test.py:  
    Perform WL test on graph structures
    Not presented in the paper.
"""

__author__ = "Weizhe Lin"
__copyright__ = "Copyright 2021, Weizhe Lin"
__version__ = "1.0.0"
__email__ = "wl356@cam.ac.uk"
__status__ = "Published for Github"

import hashlib
from utils.log_system import logger

class WLTestKernel(object):
    """This module performs WL test on the graph, returning graph node representations
    """
    def __init__(self, config, triplets):
        self.config = config
        self.data = {}
        self.max_iter = self.config.model_config.wl_test_iter
        self.node_color_dict = {}
        self.node_neighbor_dict = {}
        node_list = []
        link_list = []
        for h_id, r_id, t_id in tqdm(triplets, ascii=True):
            node_list.append(h_id)
            node_list.append(t_id)
            link_list.append((h_id, t_id))
        node_list = list(set(node_list))
        link_list = list(set(link_list))
        print('max node id', max(node_list))
        for i in range(max(node_list)+1):
            # make sure isolated nodes are in the list as well
            node_list.append(i)
        node_list = list(set(node_list)) # refine list

        self.data['idx'] = node_list
        self.data['edges'] = link_list

        logger.print('WL test kernel initialized.')
        logger.print('#nodes = {}, #links = {}'.format(len(node_list), len(link_list)))

    def setting_init(self, node_list, link_list):
        for node in node_list:
            self.node_color_dict[node] = 1
            self.node_neighbor_dict[node] = {}

        for pair in link_list:
            u1, u2 = pair
            if u1 not in self.node_neighbor_dict:
                self.node_neighbor_dict[u1] = {}
            if u2 not in self.node_neighbor_dict:
                self.node_neighbor_dict[u2] = {}
            self.node_neighbor_dict[u1][u2] = 1
            self.node_neighbor_dict[u2][u1] = 1

    def WL_recursion(self, node_list):
        iteration_count = 1
        while True:
            logger.print('WL test running: iter {}/{} ...'.format(iteration_count, self.max_iter))
            new_color_dict = {}
            for node in node_list:
                neighbors = self.node_neighbor_dict[node]
                neighbor_color_list = [self.node_color_dict[neb] for neb in neighbors]
                color_string_list = [str(self.node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
                color_string = "_".join(color_string_list)
                hash_object = hashlib.md5(color_string.encode())
                hashing = hash_object.hexdigest()
                new_color_dict[node] = hashing
            color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
            for node in new_color_dict:
                new_color_dict[node] = color_index_dict[new_color_dict[node]]
            if self.node_color_dict == new_color_dict or iteration_count == self.max_iter:
                logger.print('WL test converges at iter {}/{}'.format(iteration_count, self.max_iter))
                return
            else:
                self.node_color_dict = new_color_dict
            iteration_count += 1


    def run(self):
        logger.print('WL test running...')
        node_list = self.data['idx']
        link_list = self.data['edges']
        self.setting_init(node_list, link_list)
        self.WL_recursion(node_list)
        logger.print('WL test running fininshed.')
        all_values = list(self.node_color_dict.values())
        logger.print('max color {}, unique color {}'.format(max(all_values),
                                                        len(set(all_values))))
        return self.node_color_dict
