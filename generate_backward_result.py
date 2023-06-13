import os
import re
import numpy as np
from utils.global_variable import RESULT_PATH
from utils.data_operator import final_log_result


all_current_test_all_dirs = [
    "schedule-review-testbed-06-08-00-44-33",
    "schedule-review-testbed-06-08-01-11-39",
    "schedule-review-testbed-06-08-00-59-55",
    "schedule-review-testbed-06-08-14-01-35",
    "schedule-review-testbed-06-08-00-45-23",
    "schedule-review-testbed-06-08-00-59-00",
    "schedule-review-testbed-06-08-14-02-50",
    "schedule-review-testbed-06-08-14-30-04",
    "schedule-review-testbed-06-08-01-24-45",
    "schedule-review-testbed-06-08-01-21-41",
    "schedule-review-testbed-06-08-01-22-32",
    "schedule-review-testbed-06-08-01-18-50",
    "schedule-review-testbed-06-08-01-20-06",
    "schedule-review-testbed-06-08-14-28-23",
    "schedule-review-testbed-06-08-15-09-52",
    "schedule-review-testbed-06-08-16-37-04",
    "schedule-review-testbed-06-08-15-37-09",
    "schedule-review-testbed-06-08-16-04-41",
    "schedule-review-testbed-06-08-16-13-15",
    "schedule-review-testbed-06-08-16-20-58",
    "schedule-review-testbed-06-08-16-25-31",
    "schedule-review-testbed-06-08-15-14-39",
    "schedule-review-testbed-06-08-16-38-30",
    "schedule-review-testbed-06-08-15-38-14",
    "schedule-review-testbed-06-08-16-00-48",
    "schedule-review-testbed-06-08-16-13-57",
    "schedule-review-testbed-06-08-16-22-00",
    "schedule-review-testbed-06-08-16-26-53",
    "schedule-review-testbed-06-08-15-10-45",
    "schedule-review-testbed-06-08-16-39-21",
    "schedule-review-testbed-06-08-15-39-39",
    "schedule-review-testbed-06-09-10-57-04",
    "schedule-review-testbed-06-08-16-16-53",
    "schedule-review-testbed-06-08-16-23-31",
    "schedule-review-testbed-06-08-16-27-45",
    "schedule-review-testbed-06-08-15-13-15",
    "schedule-review-testbed-06-08-16-39-53",
    "schedule-review-testbed-06-08-15-40-19",
    "schedule-review-testbed-06-08-16-06-44",
    "schedule-review-testbed-06-08-16-17-23",
    "schedule-review-testbed-06-08-16-24-10",
    "schedule-review-testbed-06-08-16-36-09",
]
all_result_file_name = "all_result.log"
for dir in all_current_test_all_dirs:
    final_log_result(dir, all_result_file_name)