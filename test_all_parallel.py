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
    print(process_args)
    exit()
    subprocess.run(process_args)

def test_all_parallel(instances_list, title, num_ray):
    config_list = []
    for instance in instances_list:
        instance_name, num_vec = instance
        config_list += [[instance_name, num_vec, title, num_ray]]
    # print(len(config_list))
    # exit()
    # for config in config_list:
    #     test_proc(config[0], config[1], config[2], config[3])
    with mp.Pool(20) as pool:
       pool.starmap(test_proc, config_list)
    # for config in config_list:
    #     test(*config)

if __name__ == "__main__":
    title="moco-hnc-pdptw"
    num_ray=200

    instances_list = [
        ("bar-n100-1",6),
        ("bar-n100-2",5),
        ("bar-n100-3",6),
        ("bar-n100-4",12),
        ("bar-n100-5",6),
        ("bar-n100-6",3),
        ("ber-n100-1",13),
        ("ber-n100-2",6),
        ("ber-n100-3",3),
        ("ber-n100-4",3),
        ("ber-n100-5",5),
        ("ber-n100-6",14),
        ("poa-n100-1",12),
        ("poa-n100-2",15),
        ("poa-n100-3",10),
        ("poa-n100-4",7),
        ("poa-n100-5",6),
        ("poa-n100-6",3),
        ("bar-n200-1",22),
        ("bar-n200-2",23),
        ("bar-n200-3",8),
        ("bar-n200-4",13),
        ("bar-n200-5",5),
        ("bar-n200-6",9),
        ("ber-n200-1",27),
        ("ber-n200-2",12),
        ("ber-n200-3",9),
        ("ber-n200-4",5),
        ("ber-n200-5",27),
        ("ber-n200-6",9),
        ("poa-n200-1",25),
        ("poa-n200-2",12),
        ("poa-n200-3",22),
        ("poa-n200-4",10),
        ("poa-n200-5",15),
        ("poa-n200-6",27),
        ("bar-n400-1",32),
        ("bar-n400-2",30),
        ("bar-n400-3",11),
        ("bar-n400-4",17),
        ("bar-n400-5",41),
        ("bar-n400-6",21),
        ("ber-n400-1",34),
        ("ber-n400-2",33),
        ("ber-n400-3",43),
        ("ber-n400-4",19),
        ("ber-n400-5",26),
        ("ber-n400-6",19),
        ("poa-n400-1",24),
        ("poa-n400-2",41),
        ("poa-n400-3",40),
        ("poa-n400-4",19),
        ("poa-n400-5",14),
        ("poa-n400-6",42),
                    ]
    num_vehicles_list = [1,3,5,10]
    test_all_parallel(instances_list, title, num_ray)
