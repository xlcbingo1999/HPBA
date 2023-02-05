import json
import itertools


def read_gpu_status(all_devices_logic_id):
    all_gpu_status_paths = [
        "/mnt/linuxidc_client/gpu_states/{}.json".format(logic_id) for logic_id in all_devices_logic_id
    ]
    results = {}
    for index in range(len(all_devices_logic_id)):
        current_gpu_status = {}
        with open(all_gpu_status_paths[index], 'r') as f:
            current_gpu_status = json.load(f)
        results[all_devices_logic_id[index]] = current_gpu_status
    return results

if __name__ == "__main__":
    ip_adds = ["172.18.162.3", "172.18.162.4", "172.18.162.6"]
    local_device_ids = [0, 1, 2, 3]
    all_devices = itertools.product(ip_adds, local_device_ids)
    all_devices_logic_id = [
        "{}-{}".format(ip_device_tuple[0], ip_device_tuple[1]) for ip_device_tuple in all_devices
    ]
    results = read_gpu_status(all_devices_logic_id)
    print(results)
