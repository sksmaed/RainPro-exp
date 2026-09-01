"""Marshall-Palmer dBZ <-> mm/h conversion (Z = 200 R^1.6), as used by the paper
(Sec. A.2) to convert QPESUMS max dBZ into the mm/h intensity classes of App. E.
"""

from __future__ import annotations

import numpy as np

_A, _B = 200.0, 1.6


def dbz_to_mmh(dbz: np.ndarray) -> np.ndarray:
    z = np.power(10.0, dbz / 10.0)
    return np.power(z / _A, 1.0 / _B)


def mmh_to_dbz(mmh: np.ndarray) -> np.ndarray:
    z = _A * np.power(mmh, _B)
    return 10.0 * np.log10(z)
