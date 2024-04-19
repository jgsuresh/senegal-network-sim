# Forward dream
import random

import numpy as np


def meiosis(quantum, nsnps, p_oocysts=0.5, bp_per_cM=20):
    """
    Run P.falciparum meiosis on the `quantum` of
    strains that enter the vector, with...

    - Number of oocysts drawn from min[10, Geo(p_o)]
    - Random pairings from `quantum` to produce zygotes
    - Number of cross-overs max[1, Poi(bp_per_cM)]
    - Random pairings from `bivalent` for cross-overs
    - Recomb. rate uniform along chromosome

    ...and returning *all* progeny generated across
    *all* oocysts.

    Parameters
        quantum : ndarray, shape (k, nsnps)
            Array of parasite genomes which will
            undergo meiosis. If `k` is < 2,
            no recombination occurs.
        nsnps : int
            Number of SNPs per parasite genome.
        p_oocysts : float in 0 < p <= 1
            Number of oocysts is drawn from
            ~Geo(p_o), up to a maximum of 10. Note
            that if p = 1, only a single oocyst
            is drawn every time.
        bp_per_cM : float
            Recombination rate. Default of results in an
            average of one CO per bivalent when
            `nsnps`=1000.

    Returns
        progeny: ndarray, shape (n_oocysts*4, nsnps)
            Array of parasite genomes, after they have
            undergone meiosis.

    """

    if len(quantum) > 1:

        # Compute cross-over rate *per bivalent*
        mean_n_co = 2 * nsnps / (bp_per_cM * 100)

        # Draw no. oocysts
        n_oocysts = min([np.random.geometric(p_oocysts), 10])
        print("Number of oocysts: ", n_oocysts)

        # Pair strains to create zygotes, 1 per oocyst
        i = random.choices(range(len(quantum)), k=n_oocysts * 2)
        zygotes = list(zip(i[:-1:2], i[1::2]))

        # Run meiosis for each zygote
        progeny = []
        for zygote in zygotes:
            parentals = np.copy(quantum[zygote, :])

            if not (parentals[0] == parentals[1]).all():  # if identical, no need
                print("Parentals are not identical: doing crossover events")
                # Create bivalent
                bivalent = np.dstack([np.copy(parentals), np.copy(parentals)])

                # Prepare crossover events
                n_co = max([1, np.random.poisson(mean_n_co)])  # enforce at least 1 CO
                print("Number of crossover events: ", n_co)
                co_brks = random.choices(range(nsnps), k=n_co)
                co_brks.sort()
                i = random.choices(range(2), k=n_co * 2)
                co_pairs = list(zip(i[:-1:2], i[1::2]))

                # Resolve crossovers
                for brk, pair in zip(co_brks, co_pairs):
                    bivalent[[0, 1], :brk, pair] = bivalent[[1, 0], :brk, pair[::-1]]

                # Segregate & store progeny
                progeny.append(np.vstack([bivalent[:, :, 1], bivalent[:, :, 0]]))
            else:
                print("Parentals are identical")
                progeny.append(np.vstack([parentals, parentals]))

        # Combine progeny across oocysts
        progeny = np.vstack(progeny)

    else:
        progeny = quantum

    return progeny

# play around with this setup
if __name__ == "__main__":
    nsnps = 10
    quantum = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]])

    # Test case with 4 unique strains and 100 SNPs
    # nsnps = 1000
    # quantum = np.array([[random.choice([0, 1]) for _ in range(nsnps)],
    #                     [random.choice([0, 1]) for _ in range(nsnps)],
    #                     [random.choice([0, 1]) for _ in range(nsnps)],
    #                     [random.choice([0, 1]) for _ in range(nsnps)]])
    p_oocysts = 0.5
    bp_per_cM = 20
    # bp_per_cM = 0.01
    progeny = meiosis(quantum, nsnps, p_oocysts, bp_per_cM)
    print(progeny)