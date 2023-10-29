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

p = 3

Lx = (xmax - xmin) * p
Lz = (zmax - zmin) * p  # [m]
nx = 91 * p
nz = 71 * p  # [n]
dx = (Lx/nx)
dz = (Lz/nz)  # [m/n]


Tt = np.linspace(0, T-dt, nt)


xrow = np.linspace(xmin, xmax, nx)
zrow = np.linspace(zmin, zmax, nz)
tup = np.meshgrid(xrow, zrow)
grid = tuple([(x.reshape(nx*nz)) for x in tup])


Z = grid[-1]
drop_threshold = 1e-7


def _gaussian_derivative_pulse(ZZ, threshold, **kwargs):
    """ Derivative of a Gaussian at a specific sigma """
    T = -100.0*ZZ*np.exp(-(ZZ**2)/1e-4)
    T[np.where(abs(T) < threshold)] = 0
    return T


gaussian_derivative = _gaussian_derivative_pulse(Z, drop_threshold)


def ricker(t, a):
    # x = np.diff(np.diff(np.exp(-t**2/a))/dt)/dt
    x = np.exp(-t**2/a)*(4*t**2/a**2-2/a)
    return x/np.max(np.abs(x))


tt = Tt-T/21
am = 1e-3
source = ricker(tt, am)

# plt.figure()
# plt.plot(source)
# plt.show()

p_present = np.zeros((nz, nx))
p_past = np.zeros((nz, nx))
p_future = np.zeros((nz, nx))


c0 = 2  # [m/s]

# MELHORAR A PARTE ABAIXO:

c = np.zeros((nz, nx))
c[:] = c0
c[28 * p, :] = 2.000016747910872
c[29 * p, :] = 2.0048261353405694
c[30 * p, :] = 2.158098836842796
c[31 * p, :] = 2.3894003915357027
c[32 * p, :] = 1.6105996084642973
c[33 * p, :] = 1.841901163157204
c[34 * p, :] = 1.9951738646594308
c[35 * p, :] = 1.9999832520891276

c[42 * p, :] = 2.0000167479108724
c[43 * p, :] = 2.0048261353405694
c[44 * p, :] = 2.158098836842796
c[45 * p, :] = 2.3894003915357027
c[46 * p, :] = 1.6105996084642973
c[47 * p, :] = 1.841901163157204
c[48 * p, :] = 1.9951738646594308
c[49 * p, :] = 1.9999832520891276


# shots
nr = np.arange(nx)  # n do receptor

zpos = int(((1./9.)*zmax)/dz) * p   # [n]

pos_rec_x = nr.copy()
pos_rec_z = zpos * np.ones_like(nr)  # matriz posição do receptor

source_x = (nx // 2)
source_z = zpos  # posição da fonte

# pos_rec = pos_rec.astype(int)

act = np.zeros((nt, nx))  # matriz das aquisições no tempo


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

    plt.imsave(f"images/plot_{nt}.png", p_future, cmap='gray')

create_gif_and_video(nt)

# plot shot
plt.figure()
plt.imshow(act, origin='upper', aspect='auto')
plt.colorbar()
plt.show()
print('done')
