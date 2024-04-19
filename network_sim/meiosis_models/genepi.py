"""
Refactoring malaria genetics tracking code from the previous custom Pf genome data model:
https://github.com/edwenger/genepi/blob/master/genepi/meiosis.py
"""

from collections import defaultdict
import itertools
import logging
import math

import numpy as np

from postparent.core.recombination import BaseRecombinationModel

log = logging.getLogger(__name__)


chromatids = 'AaBb'
chromatid_pairings = ('AB', 'Ab', 'aB', 'ab')  # excludes sister couplings (Aa or Bb)


def pairwise(iterable):
    """ s -> (s0,s1), (s1,s2), (s2, s3), ... """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class MeiosisRecombinationModel(BaseRecombinationModel):

    """
    A concrete implementation of meiotic recombination
    """

    def __init__(self, bp_per_morgan):
        super().__init__()

        self.bp_per_morgan = bp_per_morgan

    def get_edges(self, child_ids, parent_ids, interval):

        super().get_edges(child_ids, parent_ids, interval)

        if len(child_ids) not in range(1, 5):
            raise Exception('Meiosis model expects between 1 and 4 meiotic progeny per recombination')

        begin, end = interval

        if parent_ids[0] == parent_ids[1]:
            return [dict(left=begin, right=end, parent=parent_ids[0], child=child) for child in child_ids]

        """
        Crossover events are randomly assigned between pairs of non-sister chromatids.
        """

        crossover_positions = self._get_crossover_positions(interval)
        chromatid_crosses = [(pos, np.random.choice(chromatid_pairings)) for pos in crossover_positions]
        log.debug(chromatid_crosses)

        chromatid_order = sorted(chromatids, key=lambda k: np.random.random())  # initial shuffle
        initial_order = tuple(chromatid_order)
        log.debug(initial_order)

        crosses_by_chromatid = defaultdict(list)
        for position, crossed_chromatids in chromatid_crosses:
            c1, c2 = [chromatid_order.index(c) for c in crossed_chromatids]
            for c in (c1, c2):
                crosses_by_chromatid[c].append(position)
            chromatid_order[c1], chromatid_order[c2] = chromatid_order[c2], chromatid_order[c1]

        """
        Genetic blocks are assigned to either the first or second parent
        for each of the meiotic siblings crossover positions by chromatid
        """

        edges = []

        for i, c in enumerate(initial_order[:len(child_ids)]):
            log.debug('Cross positions involving chromatid %s (ix=%d): %s', c, i, crosses_by_chromatid[i])

            current_input_genome = itertools.cycle(parent_ids if c in 'Aa' else reversed(parent_ids))

            crossover_boundaries = [begin] + crosses_by_chromatid[i] + [end]

            for (left, right), parent in zip(pairwise(crossover_boundaries), current_input_genome):
                log.debug('(%d - %d: %d -> %d', left, right, parent, child_ids[i])
                edges.append(dict(left=left, right=right, parent=parent, child=child_ids[i]))

        return edges

    def _get_crossover_positions(self, interval):

        """
        Draw independent crossover positions with exponentially drawn inter-arrival distances
        :param: tuple (begin, end) representing chromosome interval over which to draw crossover positions
        :return: list of crossover positions in base pairs
        """

        begin, end = interval

        next_point = begin
        positions = []

        while next_point < end:
            if next_point != begin:
                positions.append(next_point)
            step = int(math.ceil(np.random.exponential(self.bp_per_morgan / 2.0)))  # two sister chromatids
            next_point += step

        return positions