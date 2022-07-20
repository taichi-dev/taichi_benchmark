import taichi as ti
import math
import time

@ti.func
def warp_inclusive_add_cuda(val:ti.template()):
    global_tid = ti.global_thread_idx()
    lane_id = global_tid % 32
    # Intra-warp scan, manually unroll
    offset_j = 1
    n = ti.simt.warp.shfl_up_f32(ti.simt.warp.active_mask(), val, offset_j)
    if (lane_id >= offset_j):
        val += n
    offset_j = 2
    n = ti.simt.warp.shfl_up_f32(ti.simt.warp.active_mask(), val, offset_j)
    if (lane_id >= offset_j):
        val += n
    offset_j = 4
    n = ti.simt.warp.shfl_up_f32(ti.simt.warp.active_mask(), val, offset_j)
    if (lane_id >= offset_j):
        val += n
    offset_j = 8
    n = ti.simt.warp.shfl_up_f32(ti.simt.warp.active_mask(), val, offset_j)
    if (lane_id >= offset_j):
        val += n
    offset_j = 16
    n = ti.simt.warp.shfl_up_f32(ti.simt.warp.active_mask(), val, offset_j)
    if (lane_id >= offset_j):
        val += n
    offset_j = 32
    n = ti.simt.warp.shfl_up_f32(ti.simt.warp.active_mask(), val, offset_j)
    if (lane_id >= offset_j):
        val += n
    return val

target = ti.cuda
if target == ti.cuda:
    inclusive_add = warp_inclusive_add_cuda
    barrier = ti.simt.block.sync
elif target == ti.vulkan:
    inclusive_add = ti.simt.subgroup.inclusive_add
    barrier = ti.simt.subgroup.barrier
else:
    raise RuntimeError(f"Arch {target} not supported for parallel scan.")

ti.init(arch=target)

WARP_SZ = 32
BLOCK_SZ = 128
n_elements = BLOCK_SZ * 16
GRID_SZ = int((n_elements + BLOCK_SZ - 1) / BLOCK_SZ)

# Declare input array and all partial sums
ele_num = n_elements
ele_nums = [ele_num]
while (ele_num > 1):
    ele_num = int((ele_num + BLOCK_SZ - 1) / BLOCK_SZ)
    ele_nums.append(ele_num)

arrs = []
for en in ele_nums:
    arr = ti.field(ti.f32, shape=en)
    arrs.append(arr)
arr_golden = ti.field(ti.f32, shape=n_elements)

# This should be replaced real smem, size is block_size/32+1
smem = ti.field(ti.f32, shape=(int(GRID_SZ), 64))


@ti.kernel
def shfl_scan(arr_in: ti.template(), sum_smem: ti.template(),
              partial_sums: ti.template(), single_block: ti.template()):
    ti.loop_config(block_dim=BLOCK_SZ)
    for i in arr_in:
        val = arr_in[i]

        thread_id = i % BLOCK_SZ
        block_id = int(i // BLOCK_SZ)
        lane_id = i % WARP_SZ
        warp_id = thread_id // WARP_SZ

        val = inclusive_add(val)
        barrier()

        # Put warp scan results to smem
        if (thread_id % WARP_SZ == WARP_SZ - 1):
            sum_smem[block_id, warp_id] = val
        barrier()

        # Inter-warp scan, use the first thread in the first warp
        if (warp_id == 0 and lane_id == 0):
            for k in range(1, BLOCK_SZ / WARP_SZ):
                sum_smem[block_id, k] += sum_smem[block_id, k - 1]
        barrier()

        # Update data with warp sums
        warp_sum = 0.0
        if (warp_id > 0):
            warp_sum = sum_smem[block_id, warp_id - 1]
        val += warp_sum
        arr_in[i] = val

        # Update partial sums
        if not single_block and (thread_id == BLOCK_SZ - 1):
            partial_sums[block_id] = val


@ti.kernel
def uniform_add(arr_in: ti.template(), nele: ti.template(),
                partial_sums: ti.template()):

    ti.loop_config(block_dim=BLOCK_SZ)
    for i in range(nele - BLOCK_SZ):
        ii = i + BLOCK_SZ  # skip the first block
        block_id = int(ii // BLOCK_SZ)
        arr_in[ii] += partial_sums[block_id - 1]


# Ground truth for comparison
def scan_golden(arr_in: ti.template()):
    cur_sum = 0.0
    for i in range(n_elements):
        cur_sum += arr_in[i]
        arr_in[i] = cur_sum


def initialize():
    for i in range(n_elements):
        arrs[0][i] = 1.0
        arr_golden[i] = 1.0


# dry run
initialize()
for i in range(len(ele_nums) - 1):
    if i == len(ele_nums) - 2:
        shfl_scan(arrs[i], smem, arrs[i + 1], True)
    else:
        shfl_scan(arrs[i], smem, arrs[i + 1], False)

for i in range(len(ele_nums) - 2, -1, -1):
    uniform_add(arrs[i], ele_nums[i], arrs[i + 1])
ti.sync()

# measure average
time_tot = 0
for _ in range(10):
    initialize()
    t = time.perf_counter()
    for i in range(len(ele_nums) - 1):
        if i == len(ele_nums) - 2:
            shfl_scan(arrs[i], smem, arrs[i + 1], True)
        else:
            shfl_scan(arrs[i], smem, arrs[i + 1], False)

    for i in range(len(ele_nums) - 2, -1, -1):
        uniform_add(arrs[i], ele_nums[i], arrs[i + 1])
    ti.sync()
    time_tot += time.perf_counter() - t

time_in_ms = time_tot / 10 * 1000
print("Average execution time in ms", time_in_ms)

# compute ground truth
scan_golden(arr_golden)

# Compare
for i in range(n_elements):
    if arr_golden[i] != arrs[0][i]:
        #print("", i)
        print(f"Failed at pos {i} arr_golden {arr_golden[i]} vs arr {arrs[0][i]}")
        #break

print("Done")
