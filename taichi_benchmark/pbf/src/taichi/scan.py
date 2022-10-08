import taichi as ti


@ti.kernel
def parallel_inclusive_scan_inplace_kernel(
    arr: ti.template(),
    warp_sum: ti.template(),
    block_sum: ti.template(),
    real_block_id: ti.template(),
    block_counter_0: ti.template(),
    block_counter_1: ti.template(),
    N: ti.i32,
    WARP_SIZE: ti.i32,
    BLOCK_SIZE: ti.i32,
    BLOCK_NUM: ti.i32,
):

    FULL_MASK = ti.u32(0xFFFFFFFF)

    # __device__ ???
    # int block_counter_0 = 0;
    # int block_counter_1 = 0;

    # __shared__ ???
    # ti.block_local(real_block_id)
    # ti.block_local(warp_sum)

    ti.loop_config(block_dim=1024)
    for i in range(BLOCK_SIZE * BLOCK_NUM):

        ltid = i % BLOCK_SIZE
        bid = i // BLOCK_SIZE

        if ltid == 0:
            real_block_id[bid] = ti.atomic_add(block_counter_0[None], 1)
        ti.simt.block.sync()

        gtid = ltid + real_block_id[bid] * BLOCK_SIZE
        bid = real_block_id[bid]

        if gtid >= N:
            continue

        lane_id = ltid % WARP_SIZE
        warp_id = ltid // WARP_SIZE
        warp_num = (BLOCK_SIZE + WARP_SIZE - 1) // WARP_SIZE

        val = arr[gtid]
        temp0 = 0

        d = 1
        while d < WARP_SIZE:
            temp0 = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, d)
            if lane_id >= d:
                val += temp0
            d <<= 1

        ti.simt.block.sync()

        if lane_id == WARP_SIZE - 1:
            warp_sum[bid, warp_id] = val

        ti.simt.block.sync()

        if ltid < WARP_SIZE:

            temp0 = 0
            if ltid < warp_num:
                temp0 = warp_sum[bid, ltid]

            d = 1
            while d < warp_num:
                temp1 = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), temp0, d)
                if lane_id >= d:
                    temp0 += temp1
                d <<= 1

            if ltid < warp_num:
                warp_sum[bid, ltid] = temp0

        ti.simt.block.sync()

        if ltid >= WARP_SIZE:
            val += warp_sum[bid, warp_id - 1]

        arr[gtid] = val
        if ltid == BLOCK_SIZE - 1:
            block_sum[bid + 1] = val

        ti.simt.block.sync()

        if ltid == 0:
            while ti.atomic_add(block_counter_1[None], 0) != bid:
                pass

            ti.simt.grid.memfence()

            if bid == 0:
                block_sum[bid] = 0
            else:
                block_sum[bid] += block_sum[bid - 1]

            ti.simt.grid.memfence()

            ti.atomic_add(block_counter_1[None], 1)

        ti.simt.block.sync()
        arr[gtid] += block_sum[bid]


block_counter_0 = None
block_counter_1 = None
warp_sum = None
block_sum = None
real_block_id = None


def parallel_inclusive_scan_inplace(arr):

    N = arr.shape[0]
    WARP_SIZE = 32
    BLOCK_SIZE = 1024
    BLOCK_NUM = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    MAX_WARP_NUM_PER_BLOCK = 32

    assert BLOCK_SIZE == 1024
    assert MAX_WARP_NUM_PER_BLOCK >= (BLOCK_SIZE + WARP_SIZE - 1) // WARP_SIZE

    global block_counter_0
    global block_counter_1
    global warp_sum
    global block_sum
    global real_block_id
    if block_sum is None or block_sum.shape[0] != BLOCK_NUM:
        block_counter_0 = ti.field(ti.i32, ())
        block_counter_1 = ti.field(ti.i32, ())
        warp_sum = ti.field(ti.i32, (BLOCK_NUM, MAX_WARP_NUM_PER_BLOCK))
        block_sum = ti.field(ti.i32, BLOCK_NUM)
        real_block_id = ti.field(ti.i32, BLOCK_NUM)

    block_counter_0.fill(0)
    block_counter_1.fill(0)
    warp_sum.fill(0)
    block_sum.fill(0)

    parallel_inclusive_scan_inplace_kernel(
        arr,
        warp_sum,
        block_sum,
        real_block_id,
        block_counter_0,
        block_counter_1,
        N,
        WARP_SIZE,
        BLOCK_SIZE,
        BLOCK_NUM,
    )
