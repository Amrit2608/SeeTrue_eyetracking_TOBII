import pickle
import numpy as np
import plot
import csv
from matplotlib import pyplot as plt
FIXATION = 0
SACCADE = 1

id_dict = {6: 0, 8: 1, 12: 2, 14: 3, 20: 4, 26: 5, 28: 6, 34: 7}
known_dict = {True: 1, False: 0}

if __name__ == "__main__":
    with open('data/data_with_events.pkl', 'rb') as f:
        data = pickle.load(f)
    """
    data is a list of dictionaries. Each dictionary corresponds to a subject.
    The keys of dictionary are below.
        id: 
            id of the subject.
        known: 
            whether the subject recognized the image or not.
        xs: 
            gaze x positions with pixel representation.
        ys: 
            gaze y positions with pixel representation.
        xs_deg: 
            gaze x positions with degree representation.
        ys_deg: 
            gaze y positions with degree representation.
        events_uneye: 
            detected events by UnEye model. same length as xs, ys, xs_deg and yes_deg.
            0 means fixation event while 1 means saccade event.
        grouped_events_uneye:
            grouped detected events by UnEye model. the first axis length is 2, 
            grouped_events_uneye[0]: grouped fixation event
            grouped_events_uneye[1]: grouped saccade event
            each grouped event have list of indexis of start point and end point.
                e.g.
                grouped_events
                >> [ [ [0,10], [20,30] ], [ [10,20], [30,40] ] ]
                it means that 
                    index 0 to 9 are fixation  
                    index 20 to 29 are fixation
                    index 10 to 19 are saccade
                    index 30 to 39 are saccade    
        events_ours: 
            detected events by our algorithm. 
            events_ours[0] means events detected by our algorithm with param1.
            events_ours[1] means events detected by our algorithm with param2.
            All other axes are the same scheme as events_uneye.
        grouped_events_uneye:
            grouped detected events by our algorithm. 
            events_ours[0] means grouped events detected by our algorithm with param1.
            events_ours[1] means grouped events detected by our algorithm with param2.
            All other axes are the same scheme as grouped_events_uneye.
        ours_params:
            dispersion parameters used on our algorithm. [1.0, 2.0]
    """

    print(data[0].keys())
    fixation_num = np.zeros(3)  # uneye, ours1, ours2
    fixation_time = np.zeros(3)  # uneye, ours1, ours2

    fig = plt.figure()

    c = ['r', 'g', 'b']
    labels = ['U\'n\'Eye', 'I-DT threshold=1.0', 'I-DT threshold=2.0']
    fn = [[], [], []]
    ft = [[], [], []]
    for i, d in enumerate(data):
        fixation_num[0] = len(d['grouped_events_uneye'][FIXATION])
        fixation_num[1] = len(d['grouped_events_ours_list'][0][FIXATION])
        fixation_num[2] = len(d['grouped_events_ours_list'][1][FIXATION])

        fixation_time[0] = sum([fixation_idx[-1]-fixation_idx[0]
                                for fixation_idx in d['grouped_events_uneye'][FIXATION]])
        fixation_time[1] = sum([fixation_idx[-1]-fixation_idx[0]
                                for fixation_idx in d['grouped_events_ours_list'][0][FIXATION]])
        fixation_time[2] = sum([fixation_idx[-1]-fixation_idx[0]
                                for fixation_idx in d['grouped_events_ours_list'][1][FIXATION]])
        # for i in range(3):
        #     plt.scatter(fixation_num[i],
        #                 fixation_time[i]/fixation_num[i], c=c[i],label=labels[i])

        for i in range(3):
            fn[i].append(fixation_num[i])
            ft[i].append(fixation_time[i]/fixation_num[i]/1000)
    for i in range(3):
        plt.scatter(fn[i], ft[i], c=c[i], label=labels[i], marker='+')
    plt.legend()
    plt.xlabel('Number of fixations')
    plt.ylabel('Mean fixation duration [sec]')
    plt.title('Comparison of A and B with two different parameters')
    plt.savefig('comparison_UnEye_I-DT.png')

    print("UnEye, Ours1, Ours2")
    print("AVG number of fixation", fixation_num / len(data))
    print("AVG fixation duration [sec]", fixation_time / fixation_num / 1000)
