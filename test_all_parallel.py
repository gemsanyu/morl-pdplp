import multiprocessing as mp
import subprocess

def test_proc(instance_name, num_vehicles, title, num_ray):
    process_args = ["python",
                    "test_phn.py",
                    "--test-instance-name",
                    instance_name,
                    "--test-num-vehicles",
                    str(num_vehicles),
                    "--title",
                    title,
                    "--num-ray",
                    str(num_ray)]
    subprocess.run(process_args)

def test_all_parallel(instances_name_list, num_vehicles_list, title, num_ray):
    config_list = [(instance_name, num_vehicles, title, num_ray) for instance_name in instances_name_list for num_vehicles in num_vehicles_list]
    with mp.Pool(20) as pool:
        pool.starmap(test_proc, config_list)
    # for config in config_list:
    #     test(*config)

if __name__ == "__main__":
    title="hnc-phn"
    num_ray=200

    instances_name_list = [
                        "bar-n100-1",
                        "bar-n100-2",
                        "bar-n100-3",
                        "bar-n100-4",
                        "bar-n100-5",
                        "bar-n100-6",
                        "bar-n200-1",
                        "bar-n200-2",
                        "bar-n200-3",
                        "bar-n200-4",
                        "bar-n200-5",
                        "bar-n200-6",
                        "bar-n400-1",
                        "bar-n400-2",
                        "bar-n400-3",
                        "bar-n400-4",
                        "bar-n400-5",
                        "bar-n400-6",
                        "ber-n100-1",
                        "ber-n100-2",
                        "ber-n100-3",
                        "ber-n100-4",
                        "ber-n100-5",
                        "ber-n100-6",
                        "ber-n200-1",
                        "ber-n200-2",
                        "ber-n200-3",
                        "ber-n200-4",
                        "ber-n200-5",
                        "ber-n200-6",
                        "ber-n400-1",
                        "ber-n400-2",
                        "ber-n400-3",
                        "ber-n400-4",
                        "ber-n400-5",
                        "ber-n400-6"
                    ]
    num_vehicles_list = [1,3,5,10]
    test_all_parallel(instances_name_list, num_vehicles_list, title, num_ray)
