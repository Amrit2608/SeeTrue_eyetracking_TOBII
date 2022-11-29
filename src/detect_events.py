from typing import List, Tuple
from collections import defaultdict
import csv
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import uneye
from animation import saveGazeAnimation
import pickle

FIXATION = 0
SACCADE = 1

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
                xs_deg, ys_deg = convertPixelPositions2DegreePositions(xs, ys)
                # gaze_points = np.array([line[2::2],line[3::2]], dtype=np.float64).reshape(-1,2)
                result.append(
                    {"id": user_id, "known": known, "xs": xs, "ys": ys, "xs_deg": xs_deg, "ys_deg": ys_deg})
    return result


def our_algorithm(gaze_points, threshold, duration_threshold):
    """
    I-DT algorithm
    """
    events = []
    curr_p = 0
    while curr_p <= len(gaze_points) - duration_threshold:
        # init window
        min_x = max_x = gaze_points[curr_p][0]
        min_y = max_y = gaze_points[curr_p][1]
        window = 2
        while window <= duration_threshold:
            min_x = min(gaze_points[curr_p+window-1][0], min_x)
            max_x = max(gaze_points[curr_p+window-1][0], max_x)
            min_y = min(gaze_points[curr_p+window-1][1], min_y)
            max_y = max(gaze_points[curr_p+window-1][1], max_y)
            window += 1
        dispersion = max_x - min_x + max_y - min_y

        if dispersion < threshold:
            while dispersion < threshold and curr_p + window <= len(gaze_points):
                min_x = min(gaze_points[curr_p+window-1][0], min_x)
                max_x = max(gaze_points[curr_p+window-1][0], max_x)
                min_y = min(gaze_points[curr_p+window-1][1], min_y)
                max_y = max(gaze_points[curr_p+window-1][1], max_y)
                dispersion = max_x - min_x + max_y - min_y
                if dispersion < threshold:
                    window += 1
            window -= 1
            events.extend([FIXATION for _ in range(window)])
            curr_p += window
        else:
            events.append(SACCADE)
            curr_p += 1

    events.extend([SACCADE for _ in range(len(gaze_points) - curr_p)])
    return events


def convertPixelPositions2DegreePositions(xs, ys):
    """"
     Convert gaze points (pixel representation) to degree positions from the subject's eye
     700 pixels correspond to 97.5 mm.
     The distance between the subject and the screen was 450 mm.
    """
    return np.degrees(np.arctan2(xs / 700.0 * 97.5, 450)), \
        np.degrees(np.arctan2(ys / 700.0 * 97.5, 450))
    # return (np.arctan2(xs / 700.0 * 97.5, 450)), \
    #     (np.arctan2(ys / 700.0 * 97.5, 450))


GROUP_12_USERS = [6, 8, 12, 14, 20, 26, 28, 34]

########### Parameters for deeplearning model ###########
min_sacc_dur = 6  # minimum saccade duration in ms
min_sacc_dist = 10  # minimum saccade distance in ms
sampfreq = 1000  # Hz
weights_name = 'weights/weights_synthetic'
##################################


def plotFigure(xs_deg, ys_deg, events_ours_list, events_uneye, param_tag=None, file_name=None):

    color_uneye = ['green' if event ==
                   SACCADE else 'red' for event in events_uneye]
    size_uneye = [1 if event ==
                  SACCADE else 10 for event in events_uneye]

    fig, ax = plt.subplots(ncols=1+len(events_ours_list),
                           nrows=1, figsize=(6*(len(events_ours_list)+1), 6))

    ax[0].set_title('UnEye (Deep Learning)', fontdict={'fontsize': 20})
    ax[0].set_xlabel('x[deg]', fontdict={'fontsize': 10})
    ax[0].set_ylabel('y[deg]', fontdict={'fontsize': 10})
    ax[0].scatter(xs_deg, ys_deg, c=color_uneye, s=size_uneye)

    for i, events_ours in enumerate(events_ours_list):
        color_ours = ['green' if event ==
                      SACCADE else 'red' for event in events_ours]
        size_ours = [1 if event ==
                     SACCADE else 4 for event in events_ours]
        if param_tag is not None:
            ax[i+1].set_title('Our algorithm threshold='+str(param_tag[i]),
                              fontdict={'fontsize': 20})

        ax[i+1].set_xlabel('x[deg]', fontdict={'fontsize': 10})
        ax[i+1].set_ylabel('y[deg]', fontdict={'fontsize': 10})
        ax[i+1].scatter(xs_deg, ys_deg, c=color_ours, s=size_ours)

    if file_name is None:
        plt.plot()
    else:
        plt.savefig(file_name)
    plt.close()


def getGroupedEventIdxs(events):
    """
    return grouped event indexies.
    grouped_events[0]: grouped fixation event indexies
    grouped_events[1]: grouped saccade event indexies
    e.g.
        grouped_events
        >> [ [ [0,10], [20,30] ], [ [10,20], [30,40] ] ]
        it means that 
            index 0 to 9 are fixation  
            index 100 to 109 are fixation
            index 10 to 19 are saccade
            index 30 to 39 are saccade            
    """
    start_point = 0
    current_event = int(events[0])
    grouped_events = [[], []]
    for i, event in enumerate(events):
        if event != current_event or i >= len(events)-1:
            if current_event == 1 or current_event == 0:
                grouped_events[current_event].append([start_point, i])
            start_point = i
            current_event = int(event)
    return grouped_events


if __name__ == "__main__":

    data = load_data("data/train.csv", GROUP_12_USERS)

    uneye_model = uneye.DNN(sampfreq=sampfreq,
                            weights_name=weights_name,
                            min_sacc_dur=min_sacc_dur,
                            min_sacc_dist=min_sacc_dist)
    ours_param = [1.0, 2.0]
    for i, d in enumerate(data):
        xs_deg, ys_deg = d["xs_deg"], d["ys_deg"]
        gaze_points = np.stack([xs_deg, ys_deg], axis=1)

        # Event detection by uneye (deep learning)
        # events: fixation=0, saccades=1
        events_uneye, probability = uneye_model.predict(xs_deg, ys_deg)
        grouped_events_uneye = getGroupedEventIdxs(events_uneye)
        data[i]["events_uneye"] = events_uneye
        data[i]["grouped_events_uneye"] = grouped_events_uneye

        # Event detection by our algorithm
        # events: fixation=0, saccades=1
        data[i]["events_ours_list"] = []
        data[i]["grouped_events_ours_list"] = []
        for p in ours_param:
            events_ours = our_algorithm(
                gaze_points, threshold=p, duration_threshold=100)
            grouped_events_ours = getGroupedEventIdxs(events_ours)
            data[i]["events_ours_list"].append(events_ours)
            data[i]["grouped_events_ours_list"].append(grouped_events_ours)

        data[i]["ours_params"] = ours_param

        # if i % 40 == 0:
        # plotFigure(xs_deg, ys_deg, data[i]["events_ours_list"],
        #            events_uneye, file_name='fig/%d.png' % i, param_tag=ours_param)

    with open("data/data_with_events.pkl", "wb") as f:
        pickle.dump(data, f)
