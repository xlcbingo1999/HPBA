import os
import re
import numpy as np
from utils.global_variable import RESULT_PATH
from utils.data_operator import final_log_result


all_current_test_all_dirs = [
    "schedule-review-simulation-05-17-09-48-57"  
]
all_result_file_name = "all_result.log"
for dir in all_current_test_all_dirs:
    final_log_result(dir, all_result_file_name)