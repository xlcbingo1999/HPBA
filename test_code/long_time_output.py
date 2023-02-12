import time

dtimes = 5
begin_time = time.time()
while time.time() - begin_time < dtimes:
    print("do sth thing")
    time.sleep(1)
print("finished! stop---")