import matplotlib.pyplot  as plt
import numpy


def plot_data(data, legend, x_ticks):
    # Setup
    plt.figure(figsize=(16,6))
    X_axis = numpy.arange(len(data))
    width = 0.2
    colors = ['#BB5566', '#DDAA33', '#004488', '#004438']

    # Get the minimum and max values
    minmax_metrics = []
    minmax_metrics.extend([
        min(data[:][0]), max(data[:][0]),
        min(data[:][1]), max(data[:][1]),
        min(data[:][2]), max(data[:][2]),
        min(data[:][3]), max(data[:][3])
        ]
    )
    y_min = min(minmax_metrics)
    y_max = max(minmax_metrics)
    offset = 0.025

    # Plot bars
    b1 = plt.bar(X_axis-width, height=data[:,0], width=width, label='Accuracy', color=colors[0])
    b2 = plt.bar(X_axis, height=data[:,1], width=width, label='Precision', color=colors[1])
    b3 = plt.bar(X_axis+width, height=data[:,2], width=width, label='Recall', color=colors[2])
    b4 = plt.bar(X_axis+width*2, height=data[:,3], width=width, label='Recall', color=colors[3])

    # # Plot lines
    l1 = plt.plot(X_axis-width, data[:,0], color=colors[0])
    l2 = plt.plot(X_axis, data[:,1], color=colors[1])
    l3 = plt.plot(X_axis+width, data[:,2], color=colors[2])
    l4 = plt.plot(X_axis+width*2, data[:,3], color=colors[3])

    # # Plot scatter (for the dots)
    s1 = plt.scatter(X_axis-width, data[:,0], color=colors[0], s=200, marker='|')
    s2 = plt.scatter(X_axis, data[:,1], color=colors[1], s=200, marker='|')
    s3 = plt.scatter(X_axis+width, data[:,2], color=colors[2], s=200, marker='|')
    s4 = plt.scatter(X_axis+width*2, data[:,3], color=colors[3], s=200, marker='|')

    # Finishing up the plotting
    ax = plt.gca()
    ax.set_ylim([y_min - offset, y_max + offset])
    plt.xlabel("Min samples split")
    plt.xticks(X_axis, x_ticks, rotation='vertical')
    plt.ylabel("Score")
    plt.title("Metrics for the decision tree classifiers")
    plt.legend((b1, b2, b3, b4), legend)
    plt.show()