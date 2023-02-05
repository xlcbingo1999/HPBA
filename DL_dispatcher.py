import zerorpc
import time

def dispatch_jobs(client, sched_ip, sched_port, global_job_id):
    jobs_detail = [
        [global_job_id, {'target_func': 'add',
            'x': 2,
            'y': 3,
            'sched_ip': sched_ip,
            'sched_port': sched_port}],
        [global_job_id+1, {'target_func': 'add',
            'x': 4,
            'y': 5,
            'sched_ip': sched_ip,
            'sched_port': sched_port}],
        [global_job_id+2, {'target_func': 'add',
            'x': 8,
            'y': 5,
            'sched_ip': sched_ip,
            'sched_port': sched_port}],
        [global_job_id+3, {'target_func': 'add',
            'x': 4,
            'y': 9,
            'sched_ip': sched_ip,
            'sched_port': sched_port}],
    ]
    client.add_jobs(jobs_detail)

def sched_dispatch(client):
    client.sched_dispatch()

if __name__ == '__main__':
    global_job_id = 0
    sched_ip = "172.18.162.3"
    sched_port = "16200"
    
    client = zerorpc.Client()
    client.connect("tcp://{}:{}".format(sched_ip, sched_port))

    origin_time = time.time()
    temp_time = time.time()

    
    # while True:
    #     if time.time() - origin_time >= 60:
    #         print("over")
    #         break
    #     if time.time() - temp_time >= 2:
    #         print("sched_dispatch")
    #         sched_dispatch(client)
    #         dispatch_jobs(client, sched_ip, sched_port, global_job_id)
    #         global_job_id = global_job_id + 4
    #         temp_time = time.time()

    
    
