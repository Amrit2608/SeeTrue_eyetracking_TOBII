from typing import List, Tuple
from collections import defaultdict
import csv
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

# user ID, known image (see task description), x coords, y coords
DataRow = Tuple[int, bool, npt.NDArray, npt.NDArray]

def load_data(filename: str, user_ids: List[int]) -> List[DataRow]:

    result = []

    with open(filename) as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            user_id = int(line[0][1:])
            if user_id in user_ids:
                known = bool(line[1])
                xs = np.array(line[2::2], dtype=np.float64)
                ys = np.array(line[3::2], dtype=np.float64)
                result.append((user_id, known, xs, ys))

    return result

GROUP_12_USERS = [6,8,12,14,20,26,28,34]


if __name__ == "__main__":

    data = load_data("data/train.csv", GROUP_12_USERS)

    for i in range(5):
        plt.plot(data[i][2], data[i][3])

    plt.show()