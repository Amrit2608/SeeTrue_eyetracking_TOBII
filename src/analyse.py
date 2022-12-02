import pickle
import numpy as np
import plot
import csv
FIXATION = 0
SACCADE = 1

id_dict = {6:0, 8:1, 12:2, 14:3, 20:4, 26:5, 28:6, 34:7}
known_dict = {True:1, False:0}

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

    # print(data[0])
    fixation_num  = np.zeros((8, 2, 3))  # uneye, ours1, ours2
    fixation_time = np.zeros((8, 2, 3))  # uneye, ours1, ours2
    saccade_num  = np.zeros((8, 2, 3))  # uneye, ours1, ours2
    saccade_time = np.zeros((8, 2, 3))  # uneye, ours1, ours2

    fixation = {}
    saccade = {}
    for key, val in id_dict.items():
        fixation[key] = [[[],[],[]], [[],[],[]]]
        saccade[key] = [[[],[],[]], [[],[],[]]]

    for i, d in enumerate(data):
        
        # fixation_num[id_dict[d['id']]][int(d['known'])][0] += len(d['grouped_events_uneye'][FIXATION])
        # fixation_num[id_dict[d['id']]][int(d['known'])][1] += len(d['grouped_events_ours_list'][0][FIXATION])
        # fixation_num[id_dict[d['id']]][int(d['known'])][2] += len(d['grouped_events_ours_list'][1][FIXATION])

        # fixation_time[id_dict[d['id']]][int(d['known'])][0] += sum([fixation_idx[-1]-fixation_idx[0]
        #                         for fixation_idx in d['grouped_events_uneye'][FIXATION]])
        # fixation_time[id_dict[d['id']]][int(d['known'])][1] += sum([fixation_idx[-1]-fixation_idx[0]
        #                         for fixation_idx in d['grouped_events_ours_list'][0][FIXATION]])
        # fixation_time[id_dict[d['id']]][int(d['known'])][2] += sum([fixation_idx[-1]-fixation_idx[0]
        #                         for fixation_idx in d['grouped_events_ours_list'][1][FIXATION]])
        
        fixation[d['id']][int(d['known'])][0].extend([fixation_idx[-1]-fixation_idx[0]
                                for fixation_idx in d['grouped_events_uneye'][FIXATION]])
        fixation[d['id']][int(d['known'])][1].extend([fixation_idx[-1]-fixation_idx[0]
                                for fixation_idx in d['grouped_events_ours_list'][0][FIXATION]])
        fixation[d['id']][int(d['known'])][2].extend([fixation_idx[-1]-fixation_idx[0]
                                for fixation_idx in d['grouped_events_ours_list'][1][FIXATION]])

        # saccade_num[id_dict[d['id']]][int(d['known'])][0] += len(d['grouped_events_uneye'][SACCADE])
        # saccade_num[id_dict[d['id']]][int(d['known'])][1] += len(d['grouped_events_ours_list'][0][SACCADE])
        # saccade_num[id_dict[d['id']]][int(d['known'])][2] += len(d['grouped_events_ours_list'][1][SACCADE])

        # saccade_time[id_dict[d['id']]][int(d['known'])][0] += sum([saccade_idx[-1]-saccade_idx[0]
        #                         for saccade_idx in d['grouped_events_uneye'][SACCADE]])
        # saccade_time[id_dict[d['id']]][int(d['known'])][1] += sum([saccade_idx[-1]-saccade_idx[0]
        #                         for saccade_idx in d['grouped_events_ours_list'][0][SACCADE]])
        # saccade_time[id_dict[d['id']]][int(d['known'])][2] += sum([saccade_idx[-1]-saccade_idx[0]
        #                         for saccade_idx in d['grouped_events_ours_list'][1][SACCADE]])

        saccade[d['id']][int(d['known'])][0].extend([saccade_idx[-1]-saccade_idx[0]
                                for saccade_idx in d['grouped_events_uneye'][SACCADE]])
        saccade[d['id']][int(d['known'])][1].extend([saccade_idx[-1]-saccade_idx[0]
                                for saccade_idx in d['grouped_events_ours_list'][0][SACCADE]])
        saccade[d['id']][int(d['known'])][2].extend([saccade_idx[-1]-saccade_idx[0]
                                for saccade_idx in d['grouped_events_ours_list'][1][SACCADE]])

        for fixation_idx in d['grouped_events_ours_list'][0][FIXATION]:
            start_idx = fixation_idx[0]
            end_idx = fixation_idx[-1]
            c_x = np.average(d['xs_deg'][start_idx:end_idx])
            c_y = np.average(d['ys_deg'][start_idx:end_idx])

            # print("id: ", id_dict[d['id']], "Centroid point:",
            #       (c_x, c_y), (end_idx-start_idx)/1000.0, "[sec]")
    # print(fixation_time/fixation_num)

    # MFD = fixation_time/fixation_num
    # MSA = saccade_time/saccade_num

    print(np.sum(fixation[6][0][1])/len(fixation[6][0][1]))

    MFD = {}
    MSA = {}
    MFD_SD = {}
    MSA_SD = {}
    MFD_overall = {}
    MSA_overall = {}
    MFD_overall_SD = {}
    MSA_overall_SD = {}

    # for now, we're only calculating the algorthm for ours 1
    algorithm = 1
    for idx, val in id_dict.items():
        MFD[idx] = [np.sum(fixation[idx][0][algorithm])/len(fixation[idx][0][algorithm]), 
                    np.sum(fixation[idx][1][algorithm])/len(fixation[idx][1][algorithm])]

        MSA[idx] = [np.sum(saccade[idx][0][algorithm])/len(saccade[idx][0][algorithm]), 
                        np.sum(saccade[idx][1][algorithm])/len(saccade[idx][1][algorithm])]

        MFD_SD[idx] = [np.std(fixation[idx][0][algorithm]),np.std(fixation[idx][1][algorithm])]
        MSA_SD[idx] = [np.std(saccade[idx][0][algorithm]),np.std(saccade[idx][1][algorithm])]

        overall_fixation = fixation[idx][0][algorithm]
        overall_fixation.extend(fixation[idx][1][algorithm])
        overall_saccade = saccade[idx][0][algorithm]
        overall_saccade.extend(saccade[idx][1][algorithm])

        MFD_overall[idx] = np.sum(overall_fixation)/(len(overall_fixation))
        MSA_overall[idx] = np.sum(overall_saccade)/(len(overall_saccade))

        MFD_overall_SD[idx] = np.std(overall_fixation)
        MSA_overall_SD[idx] = np.std(overall_saccade)

    # MFD_SD = np.std(MFD[:,:,1],axis=1)

    legend = ['subject_id','MFD_true','MFD_SD_true','MFD_false','MFD_SD_false','MSA_true','MSA_SD_true','MSA_false',
                'MSA_SD_false','MFD_overall','MFD_overall_SD','MSA_overall','MSA_overall_SD']

    with open('data/result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(legend)
        for idx,val in id_dict.items():
            writer.writerow([idx,MFD[idx][0], MFD[idx][1], MFD_SD[idx][0], MFD_SD[idx][1], 
                                MSA[idx][0], MSA[idx][1], MSA_SD[idx][0], MSA_SD[idx][1], 
                                MFD_overall[idx], MFD_overall_SD[idx], MSA_overall[idx], MSA_overall_SD[idx]])

    # print(MFD[:,:,1])
    # plot_data = np.append(MFD[:,:,1], MSA[:,:,1], axis=1)
    # plot.plot_data(plot_data, legend, id_dict.keys())

    # print("UnEye, Ours1, Ours2")
    # print("AVG number of fixation", fixation_num / len(data))
    # print("AVG fixation duration [sec]", fixation_time / fixation_num / 1000)

    