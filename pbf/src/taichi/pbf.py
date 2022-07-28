# All the PBF implements I can find on the internet do not provide the `physically-real` parameters.
# To make the parameter more meaningful, I try to use a physically-real parameters in this implementation.
# However, there are still some problems existed:
# TODO:
# 1. particle vibration: use solid particles as the boundary
# 2. viscosity is too high
from random import random
from re import template
import timeit
from typing import List
from git import base

import taichi as ti
import numpy as np
import json
import math
from scan import parallel_inclusive_scan_inplace

ti.init(arch=ti.cuda, random_seed = 0, kernel_profiler=False)


screen_res = (800, 800)

# pbf-opt parameters

# large scale
# boundary_box_np = np.array([[-2, 0, -2], [2, 20, 2]])
# spawn_box_np = np.array([[-1.3, 0.3, -1.3], [1.3, 18.7, 1.3]])

# large scale
# boundary_box_np = np.array([[-2, 0, -2], [2, 30, 2]])
# spawn_box_np = np.array([[-0.3, 0.3, -0.3], [0.3, 28.7, 0.3]])

# middle scale
boundary_box_np = np.array([[0, 0, 0], [2, 6, 2]])
spawn_box_np = np.array([[0.2, 0.3, 0.2], [0.8, 5.7, 0.8]])

# small scale
# boundary_box_np = np.array([[0, 0, 0], [2, 2, 2]])
# spawn_box_np = np.array([[0.3, 0.3, 0.3], [0.35, 0.35, 0.35]])

# Every time change particle_radius, k and relaxation_parameter should be fine-tuned too,
# radius=0.025, relaxation_parameter=1000.0, pressure_k=0.0002 seem to be a best configuration.
particle_radius = 0.01
particle_diameter = particle_radius * 2
N_np = ((spawn_box_np[1] - spawn_box_np[0]) / particle_diameter + 1).astype(int)

h = 4.0 * particle_radius

# def next_power_of_2(x):  
#     return 1 if x == 0 else 2**(x - 1).bit_length()

# for i in range(3):
#     N_np[i] = next_power_of_2(int(N_np[i]))

particle_num = N_np[0] * N_np[1] * N_np[2]
print(particle_num)
pos = ti.Vector.field(3, ti.f32, shape=particle_num)
prev_pos = ti.Vector.field(3, ti.f32, shape=particle_num)
delta_p = ti.Vector.field(3, ti.f32, shape=particle_num)
delta_v = ti.Vector.field(3, ti.f32, shape=particle_num)
vel = ti.Vector.field(3, ti.f32, shape=particle_num)
den = ti.field(ti.f32, shape=particle_num)
lambd = ti.field(ti.f32, shape=particle_num)


boundary_box = ti.Vector.field(3, ti.f32, shape=2)
spawn_box = ti.Vector.field(3, ti.f32, shape=2)
N = ti.Vector([N_np[0], N_np[1], N_np[2]])

boundary_box.from_numpy(boundary_box_np)
spawn_box.from_numpy(spawn_box_np)

grid_N_np = ((boundary_box_np[1] - boundary_box_np[0]) / h + 3).astype(int)
grid_N = ti.Vector([grid_N_np[0], grid_N_np[1], grid_N_np[2]])
cell_ids = ti.field(ti.i32, shape=particle_num)
cell_pnums = ti.field(ti.i32, shape=grid_N_np[0] * grid_N_np[1] * grid_N_np[2])
cell_pnums_temp = ti.field(ti.i32, shape=grid_N_np[0] * grid_N_np[1] * grid_N_np[2])

cell_ids_aux = ti.field(ti.i32, shape=particle_num)
pos_aux = ti.Vector.field(3, ti.f32, shape=particle_num)
prev_pos_aux = ti.Vector.field(3, ti.f32, shape=particle_num)
vel_aux = ti.Vector.field(3, ti.f32, shape=particle_num)

new_ids = ti.field(ti.i32, shape=particle_num)


gravity = ti.Vector([0.0, -9.8, 0.0])
# for debugging
# gravity = ti.Vector([0.0, 0.0, 0.0])

rest_density = 1000.0
mass = rest_density * particle_diameter * particle_diameter * particle_diameter
maxIte = 5
dt = 0.016 / maxIte
eps = 1e-6
pi = math.pi
relaxation_parameter = 1000.0
relaxation_parameter_2 = 0.0
viscosity = 0.1
pressure_k = 0.00002
pressure_n = 4

debugging = False


@ti.func
def W_poly6(R, h):
    r = R.norm()
    res = 0.0
    if r <= h:
        h2 = h * h
        h4 = h2 * h2
        h9 = h4 * h4 * h
        h2_r2 = h2 - r * r
        res = 315.0 / (64 * pi * h9) * h2_r2 * h2_r2 * h2_r2
    else:
        res = 0.0
    return res


@ti.func
def W_spiky_gradient(R, h):
    r = R.norm()
    res = ti.Vector([0.0, 0.0, 0.0])
    if r < eps:
        res = ti.Vector([0.0, 0.0, 0.0])
    elif r <= h:
        h3 = h * h * h
        h6 = h3 * h3
        h_r = h - r
        res = -45.0 / (pi * h6) * h_r * h_r * (R / r)
    else:
        res = ti.Vector([0.0, 0.0, 0.0])
    return res


W = W_poly6
W_gradient = W_spiky_gradient


@ti.func
def flatten_cell_id(cell_id: ti.template()) -> ti.i32:
    return int(cell_id[0] * grid_N[1] * grid_N[2] + cell_id[1] * grid_N[2] + cell_id[2])


@ti.func
def get_3d_cell_id(pos: ti.template()) -> ti.i32:
    cell_id = (pos - boundary_box[0]) // h + 1
    cell_id = ti.min(ti.max(cell_id, 1), grid_N)
    cell_id = int(cell_id)
    return cell_id


@ti.func
def get_flatten_cell_id(pos: ti.template()) -> ti.i32:
    return flatten_cell_id(get_3d_cell_id(pos))


@ti.kernel
def initialize_particle(
    pos: ti.template(),
    prev_pos: ti.template(),
    delta_p: ti.template(),
    N: ti.template(),
):
    for i in range(particle_num):
        pos[i] = (
            ti.Vector(
                [int(i % N[0]), int(i / N[0]) % N[1], int(i / N[0] / N[1] % N[2])]
            )
            * particle_diameter
            + spawn_box[0]
        )
        # pos[i] = ti.Vector([ti.random(), ti.random(), ti.random()])
        prev_pos[i] = pos[i]
        delta_p[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def update_cell_id(pos: ti.template(), cell_ids: ti.template(), cell_pnums: ti.template(), cell_pnums_temp: ti.template()):
    for I in ti.grouped(cell_pnums):
        cell_pnums[I] = 0
    for I in ti.grouped(pos):
        cid = get_flatten_cell_id(pos[I])
        cell_ids[I] = cid
        ti.atomic_add(cell_pnums[cid], 1)
    for I in ti.grouped(cell_pnums):
        cell_pnums_temp[I] = cell_pnums[I]


@ti.kernel
def counting_sort(cell_pnums: ti.template(), cell_pnums_temp: ti.template(), new_ids: ti.template(), cell_ids: ti.template(), cell_ids_aux: ti.template(), pos: ti.template(), pos_aux: ti.template(), prev_pos: ti.template(), prev_pos_aux: ti.template(), vel: ti.template(), vel_aux: ti.template()):
    # ti.loop_config(serialize=True)
    for i in range(particle_num):
        I = particle_num - 1 - i
        base_offset = 0
        if cell_ids[I] - 1 >= 0:
            base_offset = cell_pnums[cell_ids[I]-1]
        new_ids[I] = ti.atomic_sub(cell_pnums_temp[cell_ids[I]], 1) - 1 + base_offset
    for I in ti.grouped(cell_ids):
        cell_ids_aux[new_ids[I]] = cell_ids[I]
        pos_aux[new_ids[I]] = pos[I]
        prev_pos_aux[new_ids[I]] = prev_pos[I]
        vel_aux[new_ids[I]] = vel[I]
    for I in ti.grouped(pos):
        cell_ids[I] = cell_ids_aux[I]
        pos[I] = pos_aux[I]
        prev_pos[I] = prev_pos_aux[I]
        vel[I] = vel_aux[I]

arr_in = ti.field(ti.i32, particle_num)
arr_out = ti.field(ti.i32, particle_num)
arr_out_aux = ti.field(ti.i32, particle_num)
histograms = ti.field(ti.i32, (4, 256))
sums = ti.field(ti.i32, 4)


# @ti.kernel
# def sort_particles(cell_ids: ti.template(), arr_out: ti.template(), arr_out_aux: ti.template(), histograms: ti.template(), sums: ti.template()):
#     radix_sort(cell_ids, arr_out, arr_out_aux, histograms, sums)


# @ti.kernel
# def reorder_particles(permu: ti.template(), cell_ids: ti.template(), cell_ids_aux: ti.template(), pos: ti.template(), pos_aux: ti.template(), prev_pos: ti.template(), prev_pos_aux: ti.template(), vel: ti.template(), vel_aux: ti.template()):

#     for I in ti.grouped(permu):
#         cell_ids_aux[I] = cell_ids[permu[I]]
#         pos_aux[I] = pos[permu[I]]
#         prev_pos_aux[I] = prev_pos[permu[I]]
#         vel_aux[I] = vel[permu[I]]
#     for I in ti.grouped(pos):
#         cell_ids[I] = cell_ids_aux[I]
#         pos[I] = pos_aux[I]
#         prev_pos[I] = prev_pos_aux[I]
#         vel[I] = vel_aux[I]


# @ti.kernel
# def prefix_sum(cell_pnums: ti.template()):
#     cell_num = grid_N[0] * grid_N[1] * grid_N[2]
#     ti.loop_config(serialize=True)
#     for i in range(cell_num):
#         if i > 0:
#             cell_pnums[i] += cell_pnums[i-1]



@ti.func
def do_neighbor_job(i: ti.i32, func: ti.template()):
    p_cell_id = get_3d_cell_id(pos[i])
    for j1, j2, j3 in ti.ndrange(3, 3, 3):
        offset = ti.Vector([-1 + j1, -1 + j2, -1+j3])
        cid = flatten_cell_id(p_cell_id + offset)
        for j in range(cell_pnums[ti.max(0, cid-1)], cell_pnums[cid]):
            func(i, j)

@ti.func
def do_grad_C_job(i: ti.i32, grad_pi_C: ti.template(), grad_C_sum: ti.template()):
    p_cell_id = get_3d_cell_id(pos[i])
    for j1, j2, j3 in ti.ndrange(3, 3, 3):
        offset = ti.Vector([-1 + j1, -1 + j2, -1+j3])
        cid = flatten_cell_id(p_cell_id + offset)
        for j in range(cell_pnums[ti.max(0, cid-1)], cell_pnums[cid]):
            R = pos[i] - pos[j]
            grad_pj_C = (
                -mass * W_gradient(R, h) / rest_density
            )  ## To avoid numerical error, we need to multiply [mass/rest_density] term here to keep [grad_pj_C] not very big. Do not multiply this term until the end.
            grad_pi_C += -grad_pj_C
            grad_C_sum += grad_pj_C.norm_sqr()

@ti.func
def calc_density_job(i: ti.i32, j: ti.i32):
    R = pos[i] - pos[j]
    den[i] += mass * W(R, h)

@ti.kernel
def update_density(pos: ti.template(), den: ti.template()):
    for i in range(particle_num):
        den[i] = 0
        do_neighbor_job(i, calc_density_job)


@ti.kernel
def semi_euler(pos: ti.template(), prev_pos: ti.template(), vel: ti.template()):
    for i in range(particle_num):
        vel[i] += gravity * dt
        prev_pos[i] = pos[i]
        pos[i] += vel[i] * dt

@ti.func
def s_corr_job(i: ti.i32, j: ti.i32):
    R = pos[i] - pos[j]
    dq = ti.Vector([0.3 * h, 0.0, 0.0])
    s_corr = -pressure_k * ti.pow(W(R, h) / W(dq, h), pressure_n)
    delta_p[i] += (
        mass * (lambd[i] + lambd[j] + s_corr) * W_gradient(R, h) / rest_density
    )


@ti.func
def grad_C_job(i: ti.i32, j: ti.i32, grad_pi_C: ti.template(), grad_C_sum: ti.template()):
    R = pos[i] - pos[j]
    grad_pj_C = (
        -mass * W_gradient(R, h) / rest_density
    )  ## To avoid numerical error, we need to multiply [mass/rest_density] term here to keep [grad_pj_C] not very big. Do not multiply this term until the end.
    grad_pi_C += -grad_pj_C
    grad_C_sum += grad_pj_C.norm_sqr()

@ti.kernel
def solve_density_constraint(
    pos: ti.template(), den: ti.template(), delta_p: ti.template(), lambd: ti.template()
):
    # density constraint
    for i in range(particle_num):
        grad_C_sum = 0.0
        grad_pi_C = ti.Vector([0.0, 0.0, 0.0])

        do_grad_C_job(i, grad_pi_C, grad_C_sum)


        grad_C_sum += grad_pi_C.norm_sqr()
        # grad_C_sum *= mass / rest_density # This leads to numerical error.

        C = den[i] / rest_density - 1.0

        lambd[i] = -C / (grad_C_sum + relaxation_parameter)

    for i in range(particle_num):
        do_neighbor_job(i, s_corr_job)


restitution = 0.5
kk = 1.0
boundary_dir = ti.field(ti.f32, ())
boundary_dir[None] = 1.0


@ti.kernel
def boundary_handle(
    pos: ti.template(),
    vel: ti.template(),
    boundary_box: ti.template(),
    boundary_dir: ti.template(),
):
    for i in range(particle_num):
        collision_normal = ti.Vector([0.0, 0.0, 0.0])
        for j in ti.static(range(3)):
            if pos[i][j] < boundary_box[0][j]:
                pos[i][j] = boundary_box[0][j]
                collision_normal[j] += -1.0
        for j in ti.static(range(3)):
            if pos[i][j] > boundary_box[1][j] and j != 1:
                pos[i][j] = boundary_box[1][j]
                collision_normal[j] += 1.0
        collision_normal_length = collision_normal.norm()
        if collision_normal_length > eps:
            collision_normal /= collision_normal_length
            vel[i] -= (1.0 + restitution) * collision_normal.dot(vel[i]) * collision_normal



@ti.func
def xsph_job(i: ti.i32, j: ti.i32):
    delta_v[i] += (
                    -viscosity * mass / den[j] * (vel[i] - vel[j]) * W(pos[i] - pos[j], h)
                )


@ti.kernel
def xsph(vel: ti.template(), delta_v: ti.template()):
    for i in range(particle_num):
        do_neighbor_job(i, xsph_job)

    for i in range(particle_num):
        vel[i] += delta_v[i]


@ti.kernel
def correct_position(pos: ti.template(), prev_pos: ti.template()):
    for i in range(particle_num):
        pos[i] += delta_p[i]


@ti.kernel
def update_velocity(pos: ti.template(), prev_pos: ti.template(), vel: ti.template()):
    for i in range(particle_num):
        vel[i] = (pos[i] - prev_pos[i]) / dt


initialize_particle(pos, prev_pos, delta_p, N)

window = ti.ui.Window("PBF", screen_res, vsync=False)
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(0.0, 3.0, 5.0)
camera.up(0.0, 1.0, 0.0)
# camera.lookat(0.5, 0.5, 0.5)
camera.lookat(0, 0, 0)
camera.fov(70)
scene.set_camera(camera)
canvas = window.get_canvas()
movement_speed = 0.02

frame = 0
pause = False

while window.running:

    if not pause:
        semi_euler(pos, prev_pos, vel)
        for i in range(maxIte):
            update_cell_id(pos, cell_ids, cell_pnums, cell_pnums_temp)
            
            # naive prefix sum
            # sort_particles(cell_ids, arr_out_aux, arr_out, histograms, sums)
            # reorder_particles(arr_out, cell_ids, cell_ids_aux, pos, pos_aux, prev_pos, prev_pos_aux, vel, vel_aux)
            # prefix_sum(cell_pnums)

            # parallel scan
            parallel_inclusive_scan_inplace(cell_pnums)
            counting_sort(cell_pnums, cell_pnums_temp, new_ids, cell_ids, cell_ids_aux, pos, pos_aux, prev_pos, prev_pos_aux, vel, vel_aux)
            
            update_density(pos, den)
            if debugging:
                break
            delta_p.fill(0.0)
            solve_density_constraint(pos, den, delta_p, lambd)
            correct_position(pos, prev_pos)
            boundary_handle(pos, vel, boundary_box, boundary_dir)
        update_velocity(pos, prev_pos, vel)
        delta_v.fill(0)
        xsph(vel, delta_v)  # it need more xsph iterations to get better result

    # user controlling of camera
    if window.is_pressed(' '):
            pause = False
    camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.LMB)
    scene.set_camera(camera)

    scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
    scene.particles(pos, radius=particle_radius, color=(0.4, 0.7, 1.0))
    canvas.scene(scene)
    window.show()
    frame += 1
    if debugging and frame > 0:
        break
    if frame > 100000:
        break
    # break

ti.profiler.print_kernel_profiler_info()