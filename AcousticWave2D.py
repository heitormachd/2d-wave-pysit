import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import laplace
from PIL import Image
import cv2


def create_gif_and_video(time):
    images_for_gif = []
    images_for_video = []
    for t in range(time):
        images_for_gif.append(Image.open(f"images/plot_{t}.png"))

        img = cv2.imread(f"images/plot_{t}.png")
        height, width, layers = img.shape
        size = (width, height)
        images_for_video.append(img)

    out = cv2.VideoWriter(
        'project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

    for i in range(len(images_for_video)):
        out.write(images_for_video[i])
    out.release()

    images_for_gif[0].save('simulation.gif', format='GIF',
                           append_images=images_for_gif[1:],
                           save_all=True,
                           duration=50, loop=0)


T = 3  # [s]
nt = 4301  # [n]
dt = (T/nt)  # [s/n]

xmin = 0.1
xmax = 1.0

zmin = 0.1
zmax = 0.8

p = 1

Lx = (xmax - xmin)
Lz = (zmax - zmin)  # [m]
nx = 91 * p
nz = 71 * p  # [n]
dx = (Lx/nx)
dz = (Lz/nz)  # [m/n]

xrow = np.linspace(xmin, xmax, nx)
zrow = np.linspace(zmin, zmax, nz)
tup = np.meshgrid(zrow, xrow)
grid = tuple([(x.reshape(nx*nz)) for x in tup])

Z = grid[0]
drop_threshold = 1e-7


def _gaussian_derivative_pulse(ZZ, threshold):
    """ Derivative of a Gaussian at a specific sigma """
    P = -100.0 * ZZ * np.exp(-(ZZ ** 2) / 1e-4)
    P[np.where(abs(T) < threshold)] = 0
    return P


# Usado para criar os refletores:
dC = np.zeros(nx*nz)
for d in (0.45, 0.65):
    # depth is a percentage of the length
    depth = zmin + d * Lz
    dC += _gaussian_derivative_pulse(Z - depth, drop_threshold)

dC = dC.reshape(nx, nz)
dC = dC.T
plt.plot(dC)
plt.show()



def gaussian(t, a, style):
    if style == 'ricker':
        x = np.exp(-t**2/a)*(4*t**2/a**2-2/a)
    elif style == 'gaussian_1stderivative':
        x= np.exp(-t**2/a)*(-2*t/a)
    return x/np.max(np.abs(x))


# atraso do pulso
Tt = np.linspace(0, T-dt, nt)
tt = Tt - T/21
am = 1e-3  # amplitude
#source_style = 'ricker'
source_style = 'gaussian_1stderivative'
source = gaussian(tt, am, source_style)
plt.figure()
plt.plot(source)
plt.show()

p_present = np.zeros((nz, nx))
p_past = np.zeros((nz, nx))
p_future = np.zeros((nz, nx))

c0 = 2  # [m/s]

c = np.zeros((nz * nx))
c = c.reshape(nx, nz)
c = c.T
c[:] = c0
c[:] += dC



# shots
nr = np.arange(nx)  # n do receptor

zpos = int(((1./9.)*zmax)/dz) * p   # [n]

pos_rec_x = nr.copy()
pos_rec_z = zpos * np.ones_like(nr)  # matriz posição do receptor

source_x = (nx // 2)
source_z = zpos  # posição da fonte

act = np.zeros((nt, nx))  # matriz das aquisições no tempo
# actp = source

for nt in range(nt):

    p_future = (c ** 2) * laplace(p_present) * (dt ** 2) / (dx ** 2)

    p_future += 2 * p_present - p_past

    p_past = p_present
    p_present = p_future

    p_future[source_z, source_x] += source[nt]

    p_future[0, :] = 0
    p_future[-1, :] = 0
    p_future[:, 0] = 0
    p_future[:, -1] = 0

    act[nt, :] = p_future[pos_rec_z, pos_rec_x]
    act[nt, :] *= (-1)

    # actp[nt] = p_future[source_z,source_x]

#     plt.imsave(f"images/plot_{nt}.png", p_future, cmap='gray')
#
# create_gif_and_video(nt)

# plot shot
plt.figure()
plt.imshow(act, origin='upper', aspect='auto')
plt.colorbar()
plt.figure()
plt.plot(actp)
plt.show()

# Para colocar breakpoint
pass
