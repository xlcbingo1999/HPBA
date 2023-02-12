import os
import threading
import sys
import time
import subprocess

def do_thread():
    def do_thread_func():
        # os.system("python long_time_output.py")
        result = subprocess.run(['python', 'long_time_output.py'], stdout=subprocess.PIPE)
        print(result.stdout.decode('utf-8'))
    p = threading.Thread(target=do_thread_func, daemon=True)
    p.start()

do_thread()
print("what i want to do!")
time.sleep(8)
sys.exit(0)

# label_distributions_strs = []
# for k, v in label_distributions.items():
#     label_distributions_strs.append("{}={}".format(k, v))
# final_label_distributions_str = ','.join(label_distributions_strs)
# execute_cmds.append("--label_distributions {}".format(final_label_distributions_str))
# train_configs_strs = []
# for k, v in train_configs.items():
#     train_configs_strs.append("{}={}".format(k, v))
# final_train_configs_str = ','.join(train_configs_strs)
# execute_cmds.append("--train_configs {}".format(final_train_configs_str))