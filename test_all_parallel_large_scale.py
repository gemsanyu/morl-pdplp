import multiprocessing as mp
import subprocess

def test_proc(instance_name, num_vehicles, title, num_ray):
    process_args = ["python",
                    "test_phn.py",
                    "--test-instance-name",
                    instance_name,
                    "--device",
                    "cuda",
                    "--test-num-vehicles",
                    str(num_vehicles),
                    "--title",
                    title,
                    "--num-ray",
                    str(num_ray)]
    subprocess.run(process_args)

def test_all_parallel(instances_name_list, num_vehicles_list, title, num_ray):
    config_list = [(instance_name, num_vehicles, title, num_ray) for instance_name in instances_name_list for num_vehicles in num_vehicles_list]
    # for config in config_list:
    #     test_proc(config[0], config[1], config[2], config[3])
    with mp.Pool(5) as pool:
       pool.starmap(test_proc, config_list)
    # for config in config_list:
    #     test(*config)

if __name__ == "__main__":
    title="hnc-phn-po-init"
    num_ray=200
    idx_list = [1,2,3,4,5,6]
    nlist = [1000,3000,5000]
    graph_list = ["bar", "ber", "poa"]
    instances_name_list = [g+"-n"+str(n)+"-"+str(idx) for g in graph_list for n in nlist for idx in idx_list]
    num_vehicles_list = [10]
    test_all_parallel(instances_name_list, num_vehicles_list, title, num_ray)
