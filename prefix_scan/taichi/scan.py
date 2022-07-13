import taichi as ti
import math
import time 

ti.init(arch=ti.cuda)

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

n_elements = 100000 
WARP_SZ = 32
BLOCK_SZ = int(next_power_of_2(math.ceil(math.sqrt(n_elements))))
BLOCK_SZ = ti.max(ti.min(1024, BLOCK_SZ), 64)
GRID_SZ = int((n_elements + BLOCK_SZ - 1) / BLOCK_SZ)
n_partialSum = GRID_SZ
P_BLOCK_SZ = ti.min(GRID_SZ, BLOCK_SZ)
P_GRID_SZ = int((n_partialSum + P_BLOCK_SZ - 1) / P_BLOCK_SZ)
P_BLOCK_SZ = next_power_of_2(P_BLOCK_SZ)

arr = ti.field(ti.f32, shape=n_elements)
arr_golden = ti.field(ti.f32, shape=n_elements)
partial_sums = ti.field(ti.f32, shape=int(GRID_SZ))

# This should be replaced real smem, size is block_size/32
smem = ti.field(ti.f32, shape=(int(GRID_SZ), 64))

@ti.kernel
def scan_inplace(arr_in: ti.template(), sum_smem: ti.template(), partial_sums: ti.template()):

    # Phase One
    ti.loop_config(block_dim=BLOCK_SZ)
    for i in range(n_elements):
        val = arr_in[i]

        thread_id = i % BLOCK_SZ
        block_id = int(i // BLOCK_SZ)
        lane_id = i % WARP_SZ
        warp_id = thread_id // WARP_SZ

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

        # Put warp scan results to smem
        if (thread_id % WARP_SZ == WARP_SZ - 1):
            sum_smem[block_id, warp_id] = val
        ti.simt.block.sync()

        # Inter-warp scan, use the first thread in the first warp
        if (warp_id == 0 and lane_id == 0):
            for k in range(1, BLOCK_SZ / WARP_SZ):
                sum_smem[block_id, k] += sum_smem[block_id, k-1]
        ti.simt.block.sync()

        # Update data with warp sums
        warp_sum = 0.0
        if (warp_id > 0):
            warp_sum = sum_smem[block_id, warp_id - 1]
        val += warp_sum
        arr_in[i] = val
        
        # Update partial sums
        if (thread_id == BLOCK_SZ - 1):
            partial_sums[block_id] = val

    # Phase Two
    ti.loop_config(block_dim=P_BLOCK_SZ)
    for i in range(n_partialSum):
        val = partial_sums[i]

        thread_id = i % BLOCK_SZ
        block_id = int(i // BLOCK_SZ)
        lane_id = i % WARP_SZ
        warp_id = thread_id // WARP_SZ

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

        # Put warp scan results to smem
        if (thread_id % WARP_SZ == WARP_SZ - 1):
            sum_smem[block_id, warp_id] = val
        ti.simt.block.sync()

        # Inter-warp scan, use the first thread in the first warp
        if (warp_id == 0 and lane_id == 0):
            for k in range(1, BLOCK_SZ / WARP_SZ):
                sum_smem[block_id, k] += sum_smem[block_id, k-1]
        ti.simt.block.sync()

        # Update data with warp sums
        warp_sum = 0.0
        if (warp_id > 0):
            warp_sum = sum_smem[block_id, warp_id - 1]
        val += warp_sum
        partial_sums[i] = val

    # Phase Three: Uniform Add
    ti.loop_config(block_dim=BLOCK_SZ)
    for i in range(n_elements - BLOCK_SZ):
        ii = i + BLOCK_SZ # skip the first block
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
        arr[i] = 1.0
        arr_golden[i] = 1.0
    for i in range(8):
        partial_sums[i] = 0.0

# dry run
initialize()
scan_inplace(arr, smem, partial_sums)
ti.sync()

# measure average
time_tot = 0
for _ in range(10):
    initialize()
    t = time.perf_counter()
    scan_inplace(arr, smem, partial_sums)
    ti.sync()
    time_tot += time.perf_counter() - t

time_in_ms = time_tot / 10 * 1000
print ("Average execution time in ms", time_in_ms)

# compute ground truth
scan_golden(arr_golden)

# Compare
for i in range(n_elements):
    if arr_golden[i] != arr[i]:
        print("Failed at pos", i)
        print("arr_golden", arr_golden[i])
        print("arr", arr[i])
        break

print("Done")
