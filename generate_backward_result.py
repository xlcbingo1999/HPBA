import os
import re
import numpy as np
from utils.global_variable import RESULT_PATH
from utils.data_operator import final_operate_data


all_current_test_all_dirs = [
    "schedule-review-simulation-05-16-19-42-41"    
]
for dir in all_current_test_all_dirs:
    final_operate_data(dir)