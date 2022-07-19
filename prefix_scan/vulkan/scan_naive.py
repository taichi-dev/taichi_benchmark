import taichi as ti
import math
import time

ti.init(arch=ti.vulkan)

#TODO: This implementation requires the element size to be multiple of block_sz 
#      otherwise requires padding the field to the nearest multiple
BLOCK_SZ = 128
n_elements = BLOCK_SZ*50
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

# This should be replaced real smem
smem = ti.field(ti.f32, shape=(int(GRID_SZ), 64))

@ti.kernel
def intra_block_sum(arr_in: ti.template(), offset_j: ti.template()):
    ti.loop_config(block_dim=BLOCK_SZ)
    for i in arr_in:
        thread_id = i % BLOCK_SZ
        if (thread_id >= offset_j):
            arr_in[i] += arr_in[i-offset_j] 

@ti.kernel
def inplace_scan(arr_in: ti.template(), partial_sums: ti.template(), single_block: ti.template()):
    ti.loop_config(block_dim=BLOCK_SZ)
    for i in arr_in:
        thread_id = i % BLOCK_SZ
        block_id = int(i // BLOCK_SZ)
 
        # Update partial sums
        if not single_block and (thread_id == BLOCK_SZ - 1):
            partial_sums[block_id] = arr_in[i]


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


# run
initialize()
for i in range(len(ele_nums) - 1):
    if i == len(ele_nums) - 2:
        for j_offset in [ 2**j for j in range(0,8) ]:
            intra_block_sum(arrs[i], j_offset)
        inplace_scan(arrs[i], arrs[i + 1], True)
    else:
        for j_offset in [ 2**j for j in range(0,8) ]:
            intra_block_sum(arrs[i], j_offset)
        inplace_scan(arrs[i], arrs[i + 1], False)
    ti.sync()

for i in range(len(ele_nums) - 2, -1, -1):
    uniform_add(arrs[i], ele_nums[i], arrs[i + 1])
    ti.sync()

# compute ground truth
scan_golden(arr_golden)

# Compare
for i in range(n_elements):
    if arr_golden[i] != arrs[0][i]:
        print("Failed at pos", i)
        print("arr_golden", arr_golden[i])
        print("arr", arrs[0][i])
        break

print("Done")

