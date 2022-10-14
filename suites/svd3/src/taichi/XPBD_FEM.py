"""
Then we use XPBD-FEM (gpu version, Jacobian solver) to simulate the deformation of 2D object
"""
import taichi as ti
import time
from taichi.lang.ops import sqrt

ti.init(arch=ti.gpu, kernel_profiler=True)

h = 0.003  # timestep size

compliance = 1.0e-3  # Fat Tissuse compliance, for more specific material,please see: http://blog.mmacklin.com/2016/10/12/xpbd-slides-and-stiffness/
alpha = compliance * (1.0 / h / h
                      )  # timestep related compliance, see XPBD paper
N = 10
NF = 2 * N**2  # number of faces
NV = (N + 1)**2  # number of vertices
pos = ti.Vector.field(2, float, NV)
oldPos = ti.Vector.field(2, float, NV)
vel = ti.Vector.field(2, float, NV)  # velocity of particles
invMass = ti.field(float, NV)  #inverse mass of particles
f2v = ti.Vector.field(3, int, NF)  # ids of three vertices of each face
B = ti.Matrix.field(2, 2, float, NF)  # D_m^{-1}
F = ti.Matrix.field(2, 2, float, NF)  # deformation gradient
lagrangian = ti.field(float, NF)  # lagrangian multipliers
gravity = ti.Vector([0, -1.2])
MaxIte = 20
NumSteps = 5

gradient = ti.Vector.field(2,float, 3 * NF)
dLambda = ti.field(float, NF)
omega = 0.5
# For validation
dualResidual = ti.field(float,())
primalResidual = ti.field(float,())

attractor_pos = ti.Vector.field(2, float, ())
attractor_strength = ti.field(float, ())


@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, N + 1):
        k = i * (N + 1) + j
        pos[k] = ti.Vector([i, j]) * 0.05 + ti.Vector([0.25, 0.25])
        oldPos[k] = pos[k]
        vel[k] = ti.Vector([0, 0])
        invMass[k] = 1.0
    for i in range(N + 1):
        k = i * (N + 1) + N
        invMass[k] = 0.0
    k0 = N
    k1 = (N + 2) * N
    invMass[k0] = 0.0
    invMass[k1] = 0.0
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        B_i_inv = ti.Matrix.cols([b - a, c - a])
        B[i] = B_i_inv.inverse()


@ti.kernel
def init_mesh():
    for i, j in ti.ndrange(N, N):
        k = (i * N + j) * 2
        a = i * (N + 1) + j
        b = a + 1
        c = a + N + 2
        d = a + N + 1
        f2v[k + 0] = [a, b, c]
        f2v[k + 1] = [c, d, a]


@ti.kernel
def resetLagrangian():
    for i in range(NF):
        lagrangian[i] = 0.0

@ti.func
def computeGradient(idx, U, S, V):
    isSuccess = True
    sumSigma = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2)
    if sumSigma < 1.0e-6:
        isSuccess = False

    dcdS = 1.0 / sumSigma * ti.Vector([S[0, 0] - 1, S[1, 1] - 1
                                       ])  # (dcdS11, dcdS22)
    dsdx2 = ti.Vector([
        B[idx][0, 0] * U[0, 0] * V[0, 0] + B[idx][0, 1] * U[0, 0] * V[1, 0],
        B[idx][0, 0] * U[0, 1] * V[0, 1] + B[idx][0, 1] * U[0, 1] * V[1, 1]
    ])  #(ds11dx2, ds22dx2)
    dsdx3 = ti.Vector([
        B[idx][0, 0] * U[1, 0] * V[0, 0] + B[idx][0, 1] * U[1, 0] * V[1, 0],
        B[idx][0, 0] * U[1, 1] * V[0, 1] + B[idx][0, 1] * U[1, 1] * V[1, 1]
    ])  #(ds11dx3, ds22dx3)
    dsdx4 = ti.Vector([
        B[idx][1, 0] * U[0, 0] * V[0, 0] + B[idx][1, 1] * U[0, 0] * V[1, 0],
        B[idx][1, 0] * U[0, 1] * V[0, 1] + B[idx][1, 1] * U[0, 1] * V[1, 1]
    ])  #(ds11dx4, ds22dx4)
    dsdx5 = ti.Vector([
        B[idx][1, 0] * U[1, 0] * V[0, 0] + B[idx][1, 1] * U[1, 0] * V[1, 0],
        B[idx][1, 0] * U[1, 1] * V[0, 1] + B[idx][1, 1] * U[1, 1] * V[1, 1]
    ])  #(ds11dx5, ds22dx5)
    dsdx0 = -(dsdx2 + dsdx4)
    dsdx1 = -(dsdx3 + dsdx5)
    # constraint gradient
    dcdx0 = dcdS.dot(dsdx0)
    dcdx1 = dcdS.dot(dsdx1)
    dcdx2 = dcdS.dot(dsdx2)
    dcdx3 = dcdS.dot(dsdx3)
    dcdx4 = dcdS.dot(dsdx4)
    dcdx5 = dcdS.dot(dsdx5)

    g0 = ti.Vector([dcdx0, dcdx1]) # constraint gradient with respect to x0
    g1 = ti.Vector([dcdx2, dcdx3]) # constraint gradient with respect to x1
    g2 = ti.Vector([dcdx4, dcdx5]) # constraint gradient with respect to x2

    return g0, g1, g2, isSuccess

@ti.kernel
def semiEuler():
    # semi-Euler update pos & vel
    for i in range(NV):
        if (invMass[i] != 0.0):
            vel[i] += h * gravity + attractor_strength[None] * (
                attractor_pos[None] - pos[i]).normalized(1e-5)
            oldPos[i] = pos[i]
            pos[i] += h * vel[i]
@ti.kernel
def computeGradientVector():
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        invM0, invM1, invM2 = invMass[ia], invMass[ib], invMass[ic]
        sumInvMass = invM0 + invM1 + invM2
        if sumInvMass < 1.0e-6:
            print("wrong invMass function")
        D_s = ti.Matrix.cols([b - a, c - a])
        F[i] = D_s @ B[i]
        U, S, V = ti.svd(F[i])
        constraint = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2)
        g0, g1, g2, isSuccess = computeGradient(i, U, S, V)
        if isSuccess:
            l = invM0 * g0.norm_sqr() + invM1 * g1.norm_sqr(
            ) + invM2 * g2.norm_sqr()
            dLambda[i] = -(constraint + alpha * lagrangian[i]) / (l + alpha)
            lagrangian[i] = lagrangian[i] + dLambda[i]
            gradient[3 * i + 0]  = g0
            gradient[3 * i + 1]  = g1
            gradient[3 * i + 2]  = g2

@ti.kernel
def updatePos():
    for i in range(NF):
        ia, ib, ic = f2v[i]
        invM0, invM1, invM2 = invMass[ia], invMass[ib], invMass[ic]
        if (invM0 != 0.0):
            pos[ia] += omega * invM0 * dLambda[i] * gradient[3 * i + 0]
        if (invM1 != 0.0):
            pos[ib] += omega * invM1 * dLambda[i] * gradient[3 * i + 1]
        if (invM2 != 0.0):
            pos[ic] += omega * invM2 * dLambda[i] * gradient[3 * i + 2]

@ti.kernel
def updteVelocity():
    # update velocity
    for i in range(NV):
        if (invMass[i] != 0.0):
            vel[i] = (pos[i] - oldPos[i]) / h

@ti.func
def computeConstriant(idx, x0, x1, x2):
    D_s = ti.Matrix.cols([x1 - x0, x2 - x0])
    F = D_s @ B[idx]
    U, S, V = ti.svd(F)
    constraint = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2)
    return constraint


init_mesh()
init_pos()
pause = False
gui = ti.GUI('XPBD-FEM')
first = True
drawTime = 0
sumTime = 0
while gui.running:
    realStart = time.time()
    for e in gui.get_events():
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == gui.SPACE:
            pause = not pause
    mouse_pos = gui.get_cursor_pos()
    attractor_pos[None] = mouse_pos
    attractor_strength[None] = gui.is_pressed(gui.LMB) - gui.is_pressed(
        gui.RMB)

    gui.circle(mouse_pos, radius=15, color=0x336699)
    if not pause:
        for i in range(NumSteps):
            semiEuler()
            resetLagrangian()
            for ite in range(MaxIte):
                computeGradientVector()
                updatePos()
            updteVelocity()
    ti.sync()
    start = time.time()
    faces = f2v.to_numpy()
    for i in range(NF):
        ia, ib, ic = faces[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        gui.triangle(a, b, c, color=0x00FF00)

    positions = pos.to_numpy()
    gui.circles(positions, radius=2, color=0x0000FF)
    for i in range(N + 1):
        k = i * (N + 1) + N
        staticVerts = positions[k]
        gui.circle(staticVerts, radius=5, color=0xFF0000)
    gui.show()
    end = time.time()
    drawTime = (end - start)
    sumTime = (end - realStart)
    print("Draw Time Ratio; ", drawTime / sumTime )
