import numpy as np
import matplotlib.pyplot as plt
plt.title("MPM Benchmark (No GUI)", fontsize=20)

# plot cuda
plt.plot([8192, 32768], [2048/2.4, 2048/4.7], 'go', label='CUDA 2D', markersize=10)
plt.plot([8192, 65536], [2048/6.7, 2048/16.2], 'gs', label='CUDA 3D', markersize=10)

# plot taichi
plt.plot([8192, 32768], [2048/0.8, 2048/2.5], 'ro', label='Taichi 2D', markersize=10)
plt.plot([8192, 65536], [2048/3.9, 2048/21.9], 'rs', label='Taichi 3D', markersize=10)

# xaxis properties
plt.xscale('log', base=2)
plt.xlabel('Number of Particles', fontsize=15)
plt.xticks(fontsize=10)

#plt.yticks(np.arange(2048/21.9, 2048/0.8, 100))
plt.yscale('log', base=2)
plt.ylabel('FPS', fontsize=15)
plt.yticks(fontsize=10)

plt.margins(0.05)
plt.legend()
plt.savefig('mpm.png')

