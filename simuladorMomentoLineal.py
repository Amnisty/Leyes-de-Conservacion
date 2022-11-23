from itertools import combinations
from matplotlib import animation
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
import cgitb
cgitb.enable()


class Particle:

    def __init__(self, x, y, vx, vy, radius=0.01, styles=None):

        self.r = np.array((x, y))
        self.v = np.array((vx, vy))
        self.radius = radius

        self.styles = styles
        if not self.styles:

            self.styles = {'edgecolor': 'b', 'fill': False}

    @property
    def x(self):
        return self.r[0]

    @x.setter
    def x(self, value):
        self.r[0] = value

    @property
    def y(self):
        return self.r[1]

    @y.setter
    def y(self, value):
        self.r[1] = value

    @property
    def vx(self):
        return self.v[0]

    @vx.setter
    def vx(self, value):
        self.v[0] = value

    @property
    def vy(self):
        return self.v[1]

    @vy.setter
    def vy(self, value):
        self.v[1] = value

    def sobrepone(self, other):

        return np.hypot(*(self.r - other.r)) < self.radius + other.radius

    def dibujado(self, ax):

        circle = Circle(xy=self.r, radius=self.radius, **self.styles)
        ax.add_patch(circle)
        return circle

    def movimiento(self, dt):

        self.r += self.v * dt

        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx
        if self.x + self.radius > 1:
            self.x = 1-self.radius
            self.vx = -self.vx
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = -self.vy
        if self.y + self.radius > 1:
            self.y = 1-self.radius
            self.vy = -self.vy


class Simulacion:

    def __init__(self, n, radius=0.01, styles=None):

        self.init_particles(n, radius, styles)

    def init_particles(self, n, radius, styles=None):

        try:
            iterator = iter(radius)
            assert n == len(radius)
        except TypeError:

            def r_gen(n, radius):
                for i in range(n):
                    yield radius
            radius = r_gen(n, radius)

        self.n = n
        self.particles = []
        for i, rad in enumerate(radius):

            while True:

                x, y = rad + (1 - 2*rad) * np.random.random(2)

                vr = 0.1 * np.random.random() + 0.05
                vphi = 2*np.pi * np.random.random()
                vx, vy = vr * np.cos(vphi), vr * np.sin(vphi)
                particle = Particle(x, y, vx, vy, rad, styles)

                for p2 in self.particles:
                    if p2.sobrepone(particle):
                        break
                else:
                    self.particles.append(particle)
                    break

    def manejo_colisiones(self):

        def cambio_velocidades(p1, p2):

            m1, m2 = p1.radius**2, p2.radius**2
            M = m1 + m2
            r1, r2 = p1.r, p2.r
            d = np.linalg.norm(r1 - r2)**2
            v1, v2 = p1.v, p2.v
            u1 = v1 - 2*m2 / M * np.dot(v1-v2, r1-r2) / d * (r1 - r2)
            u2 = v2 - 2*m1 / M * np.dot(v2-v1, r2-r1) / d * (r2 - r1)
            p1.v = u1
            p2.v = u2

        pairs = combinations(range(self.n), 2)
        for i, j in pairs:
            if self.particles[i].sobrepone(self.particles[j]):
                cambio_velocidades(self.particles[i], self.particles[j])

    def animacion_movimiento(self, dt):

        for i, p in enumerate(self.particles):
            p.movimiento(dt)
            self.circles[i].center = p.r
        self.manejo_colisiones()
        return self.circles

    def movimiento(self, dt):
        for i, p in enumerate(self.particles):
            p.movimiento(dt)
        self.manejo_colisiones()

    def init(self):

        self.circles = []
        for particle in self.particles:
            self.circles.append(particle.dibujado(self.ax))
        return self.circles

    def animar(self, i):

        self.animacion_movimiento(0.01)
        return self.circles

    def ajuste_animacion(self):
        self.fig, self.ax = plt.subplots()
        for s in ['top', 'bottom', 'left', 'right']:
            self.ax.spines[s].set_linewidth(2)
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])

    def save_or_show_animation(self, anim, save, filename='collision.mp4'):
        if save:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, bitrate=1800)
            anim.save(filename, writer=writer)
        else:
            plt.show()

    def hacer_animacion(self, save=False, interval=1, filename='collision.mp4'):
        self.ajuste_animacion()
        anim = animation.FuncAnimation(self.fig, self.animar,
                                       init_func=self.init, frames=800, interval=interval, blit=True)
        self.save_or_show_animation(anim, save, filename)


if __name__ == '__main__':
    nparticles = 50
    radii = np.random.random(nparticles)*0.03+0.02
    styles = {'edgecolor': 'C0', 'linewidth': 2, 'fill': None}
    sim = Simulacion(nparticles, radii, styles)
    sim.hacer_animacion(save=False)
