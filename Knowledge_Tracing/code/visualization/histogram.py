import plotly.figure_factory as ff
import plotly.express as px
import numpy as np


def histogram(data, path):
    fig = px.histogram(data, x="total_bill")
    fig.show()
    fig.write_image(path + "images/" + "histogram.png")


def histogram2(data, label_x, path):
    # create the bins
    counts, bins = np.histogram(data, bins=range(0, 60, 5))
    bins = 0.5 * (bins[:-1] + bins[1:])

    fig = px.bar(x=bins, y=counts, labels={'x': label_x, 'y': 'count'})
    # fig.show()
    fig.write_image(path + "images/" + label_x + ".png")


def histogram_percentage(data, path):
    fig = px.histogram(data, x="total_bill", histnorm='probability density')
    fig.show()
    fig.write_image(path + "images/" + "histogram_percentage.png")


def comparison_histogram(data, labels, bin_size, path, name):
    fig = ff.create_distplot(data, group_labels=labels, bin_size=bin_size)
    fig.update_xaxes(range=[0, 40])
    # fig.show()
    fig.write_image(path + "comparison/" + name + ".png")
