from federatedml.secureprotol.spdz.tensor.fixedpoint_numpy import FixedPointTensor
from federatedml.secureprotol.spdz import SPDZ
# on guest side(assuming local Party is partys[0]): 
import numpy as np
data = np.array([[1,2,3], [4,5,6]])
with SPDZ() as spdz:
    x = FixedPointTensor.from_source("x", data)
    y = FixedPointTensor.from_source("y", partys[1])

# on host side(assuming PartyId is partys[1]):
import numpy as np
data = np.array([[3,2,1], [6,5,4]])
with SPDZ() as spdz:
    y = FixedPointTensor.from_source("y", data)
    x = FixedPointTensor.from_source("x", partys[0])