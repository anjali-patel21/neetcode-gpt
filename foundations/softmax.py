import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)

        z_stable = z - np.max(z)
        sum = 0
        for i in range(len(z_stable)):
            x = np.exp(z_stable[i])
            sum = x + sum
        
        ans = []
        for i in range(len(z_stable)):
            val = np.exp(z_stable[i])/sum
            ans.append(val)
        return np.round(ans, 4)
        pass
