import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
from mlflow import log_artifact

from Knowledge_Tracing.code.utils.utils import make_dir


def histogram(data, path):
    fig = px.histogram(data, x="total_bill")
    fig.show()
    fig.write_image(path + "images/" + "histogram.png")


def histogram2(data, label_x, path, bins_range=range(0, 60, 5)):
    # create the bins
    counts, bins = np.histogram(data, bins=bins_range)
    bins = 0.5 * (bins[:-1] + bins[1:])

    fig = px.bar(x=bins, y=counts, labels={'x': label_x, 'y': 'count'})
    # fig.show()
    file = path + "images/" + label_x + ".png"
    make_dir(path + "images/")
    fig.write_image(path + "images/" + label_x + ".png")
    #log_artifact(path + "images/" + label_x + ".png")

def histogram_percentage(data, path):
    fig = px.histogram(data, x="total_bill", histnorm='probability density')
    fig.show()
    fig.write_image(path + "images/" + "histogram_percentage.png")


def comparison_histogram(data, labels, bin_size, path, name):
    fig = ff.create_distplot(data, group_labels=labels, bin_size=bin_size)
    fig.update_xaxes(range=[0, 100])
    # fig.show()
    fig.write_image(path + name + ".png")
    log_artifact(path + name + ".png")
