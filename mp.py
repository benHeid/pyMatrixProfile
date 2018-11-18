import numpy as np

import distanceProfile
import order

def _matrixProfile(tsA, m, orderClass, distanceProfileFunction, tsB = None):
    order = orderClass(len(tsA) - m + 1)

    mp = np.full(len(tsA) - m + 1, np.inf)
    mpIndex = np.full(len(tsB) - m + 1, None, dtype = np.float)
    idx = order.next()
    for idx in range(len(tsA)-m+1):
        (distanceProfile, querySegementsId) = distanceProfileFunction(tsA, idx, m, tsB)
        idsToUpdate = distanceProfile < mp
        mpIndex[idsToUpdate] = querySegementsId[idsToUpdate]
        mp[idsToUpdate] = distanceProfile[idsToUpdate]
        #idx = order.next()

    return (mp, mpIndex)

def naiveMP(tsA, m, tsB = None):
    """
    >>> np.round(naiveMP(tsA = np.array([0.0, 1.0, -1.0, 0.0, 0.0]), tsB = np.array([-1, 1, 0, 0, -1, 1]), m = 4), 3)
    array([[ 2.,  2.,  2.],
           [ 0.,  1.,  0.]])
    """
    return _matrixProfile(tsA, m, order.LinearOrder, distanceProfile.naiveDistanceProfile, tsB)

def stmp(tsA, m, tsB = None):
    """
    >>> np.round(stmp(tsA = np.array([0.0, 1.0, -1.0, 0.0, 0.0]), tsB = np.array([-1, 1, 0, 0, -1, 1]), m = 4), 3)
    array([[ 2.,  2.,  2.],
           [ 0.,  1.,  0.]])
    """
    return _matrixProfile(tsA, m, order.LinearOrder, distanceProfile.stampDistanceProfile, tsB)

def stamp(tsA, m, tsB = None):
    """
    >>> np.round(stmp(tsA = np.array([0.0, 1.0, -1.0, 0.0, 0.0]), tsB = np.array([-1, 1, 0, 0, -1, 1]), m = 4), 3)
    array([[ 2.,  2.,  2.],
           [ 0.,  1.,  0.]])
    """
    return _matrixProfile(tsA, m, order.RandomOrder, distanceProfile.stampDistanceProfile, tsB)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
