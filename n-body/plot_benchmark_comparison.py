import matplotlib.pyplot as plt
from subprocess import Popen, PIPE

from taichi_nbody_orig import run_nbody as run_orig
from taichi_nbody_block import run_nbody as run_block
from taichi_nbody_block_loop import run_nbody as run_block_loop

def shell_exec():
    return

def run_reference_block(nBodies=1024, exec_path="./mini-nbody/cuda/nbody-block"):
    p = Popen([exec_path, str(nBodies)], stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    output = output.decode('utf-8').strip()
    numbers = output.split(",")
    speed = float(numbers[-1].strip())
    return speed


x = []
ref_block_res = []
ref_naive_res = []
ref_roof_res = []
taichi_res = []
taichi_naive_res = []
nBodies = 128
for k in range(13):
    print("#{}".format(nBodies))
    ref_speed = run_reference_block(nBodies=nBodies)
    ref_n_speed = run_reference_block(nBodies=nBodies, exec_path="./mini-nbody/cuda/nbody-orig")
    ref_roof_speed = run_reference_block(nBodies=nBodies, exec_path="./mini-nbody/cuda/nbody-unroll")
    _, taichi_speed_1 = run_block(nBodies)
    _, taichi_speed_2 = run_block_loop(nBodies)
    _, taichi_n_speed = run_orig(nBodies)
    ref_block_res.append(ref_speed)
    ref_naive_res.append(ref_n_speed)
    ref_roof_res.append(ref_roof_speed)
    #taichi_1_res.append(taichi_speed_1)
    #taichi_2_res.append(taichi_speed_2)
    taichi_res.append(max(taichi_speed_1, taichi_speed_2))
    taichi_naive_res.append(taichi_n_speed)
    print("reference=\t{}".format(ref_speed))
    print("taichi_orig=\t{:.3f}".format(taichi_n_speed))
    print("taichi_0=\t{:.3f}".format(taichi_speed_1))
    print("taichi_1=\t{:.3f}".format(taichi_speed_2))
    x.append(nBodies)
    nBodies *= 2

plt.plot(x, ref_naive_res)
plt.plot(x, taichi_naive_res)
plt.plot(x, ref_block_res)
plt.plot(x, taichi_res)
plt.plot(x, ref_roof_res)
plt.xscale('log')
plt.xlabel("#Bodies")
plt.ylabel("Speed (billion bodies per second)")
plt.legend(["CUDA Ref (Naive)", "Taichi (Naive)", "CUDA Ref (Block)", "Taichi (Block)", "CUDA Ref (Roofline)"], loc='lower right')
plt.title("Nbody benchmark")
plt.savefig("bench.png")
