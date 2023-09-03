#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd
import datetime
import configparser
import json

sys.path.append("../../../utils") 
from check import from_txt, check_to_txt
from perf import gem5_get_perf_data, vcs_get_perf_data, generate_perf_report
from utils import WorkloadManager

title = "Diffent Optimization levels for matmul operator"


workload_config = ""
if len(sys.argv) > 1:
    workload_config = sys.argv[1]

parent_dir = os.path.dirname(__file__)

def states_fn(params):
    m, k, n = tuple(params)

    return f"M{m}-K{k}-N{n}"

def golden_check(m, k, n):

    vs1 = np.ones((m, k)).astype('uint8')
    vs2 = np.ones((k, n)).astype('uint8')
    vd = np.matmul(vs1, vs2, dtype=np.uint8)
    print("vd\n",vd)

    vs1.tofile("src1.bin")
    vs2.tofile("src2.bin")
    vd.tofile('golden.bin')


    return  np.matrix(vs1), np.matrix(vs2), np.matrix(vd)

def gen_mnemonic(outfile, *arg):
    vs1, vs2, vd = arg
    with open(outfile,'w') as f:
        f.write("Vs1\n")
        for line in vs1:
            np.savetxt(f, line, fmt='%5d')

        f.write("Vs2\n")
        for line in vs2:
            np.savetxt(f, line, fmt='%5d')

        f.write("Vd\n")
        for line in vd:
            np.savetxt(f, line, fmt='%5d')


def test(num, params, defs, cpu_opts_list, wm, root_dir=""):
    m, k, n = params

    
    cpu_make_ops = "MINOR_OPTS=\'"
    for opt in cpu_opts_list:
        opt_name = opt["opt_name"]
        val = opt["val"]
        cpu_make_ops += f"--{opt_name} {val} "
    cpu_make_ops += "\' "

    print(cpu_make_ops)

    cur_timestamp = wm.get_timestamp()
    gem5_bin = wm.get("gem5_binaries")
    print(gem5_bin)
    gem5_dir = gem5_bin["bin_path"]
    gem5_mne_name = gem5_bin["name"]
    gem5_outlog = f"{gem5_mne_name}-{cur_timestamp}.run.log"
    simulator = wm.get("simulator")
    

    golden = golden_check(m, k, n)

    os.system(f"make clean")
    os.system(f"make " + 
        f"MAIN={parent_dir} " + 
        f"DEFS='-DM={m} -DK={k} -DN={n} -D__RVM__ {defs}' " +
        f"GEM5='{gem5_dir}' " +
        cpu_make_ops +
        f"GEM5_OUTLOG={gem5_outlog} run SIM={simulator}")

    os.system(f"make dump")

    stat_dir = os.path.join(root_dir, f'{num}-{wm.gen_stats_name(params)}.stats')
    
    os.makedirs(stat_dir, exist_ok=True)
    os.system(f"pwd");
    # f_name = "./classify/"+num+"_gem5.sig"
    # os.system(f"cp  ./m5out/gem5.sig  {f_name}")
    os.system(f"cp -r m5out {stat_dir}/")
    os.system(f"mv {gem5_outlog} {stat_dir}/")
    os.system(f"mv test.dump {stat_dir}/")

    with open(os.path.join(stat_dir, f"config.ini"), "w") as file:
        cpu_make_ops = cpu_make_ops.replace("--", "\n")
        file.write(cpu_make_ops)

    os.system(f"cp ../gem5_configs/{wm.get_config()} {stat_dir}/")


    golden_mnemonic = f'{stat_dir}/golden.txt'
    gen_mnemonic(golden_mnemonic, *golden)
    print("Test done")



if __name__ == "__main__":

    workloads_json = json.load(open(workload_config, "r"))

    out_root = workloads_json["output_dir"]

    os.makedirs(out_root, exist_ok=True)

    for workload in workloads_json["workloads"]:

        wm = WorkloadManager(workload, states_fn)

        cur_timestamp = wm.get_timestamp()
        simulator = wm.get("simulator")
        stats_root_dir = cur_timestamp + "-" +\
            wm.get("description") + "-" +\
            wm.get("extension") + "-" +\
            wm.get("config")
        minor_ops = wm.load_config()

        print(minor_ops)
        root_dir = os.path.join(out_root, stats_root_dir)
        os.makedirs(root_dir, exist_ok=True)

        compiler_opt = wm.get("compiler_opt")
        for opt in compiler_opt:
            defs = opt["val"]
            key = opt["name"]
            params = wm.get("params")
            for i, param in enumerate(params):
                test(key+'-'+str(i), param, defs, minor_ops, wm, root_dir=root_dir)
