import os
import numpy as np
import pandas as pd
from Functions_file import *

y_predict = [0, 2, 2, 1, 0, 0, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2]
y_actual = [1, 1, 1, 0, 0, 0, 1, 2, 0, 2, 0, 2, 1, 1, 2, 0, 2]

conf_matrix = confusion_matrix(y_actual, y_predict, y_actual)
print(conf_matrix)
acc = accuracy(y_actual, y_predict, y_actual)
print(acc)
pres = precision(y_actual, y_predict, y_actual)
print(pres)
rec = recall(y_actual, y_predict, y_actual)
print(rec)
f1_score = f1_score(y_actual, y_predict, y_actual)
print(f1_score)
