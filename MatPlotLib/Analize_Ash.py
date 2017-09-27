import xlrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

xlsx = 'ash.xlsx'

WorkBook = xlrd.open_workbook(xlsx)

sheet_0 = WorkBook.sheets()[0]

rows = sheet_0.nrows

def value_of_ash():
    data = []
    for row in range(int(rows)):
        data.append([float(sheet_0[row][0], float(sheet_0[row][1]), float(sheet_0[row][3]), float(sheet_0[row][5]))])
    data = np.array(data)
    return data

print(value_of_ash())