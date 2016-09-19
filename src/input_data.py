import numpy as np

def read_data(file_name):
    file = open(file_name, 'r')
    arr = np.loadtxt(file, delimiter=',')
    return arr