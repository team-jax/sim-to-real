"""NumPy port of README 7.1 / v2_arm_raise Catmull-Rom and grid lookup."""
import numpy as np
import torch

TIMES = np.array([0, 5, 5.6, 6.2, 6.8, 7.4, 8.0])
GEOMETRIC = np.deg2rad(np.array([
    [0]*8,
    [-28.8, 0, 2.5, 89.1, -5.6, 20.8, -6.6, -170.1],
    [-24.6, 0, 2.5, 87, -5.6, 18.7, -6.6, -170.1],
    [-20.4, 0, 2.5, 84.4, -5.6, 17, -6.6, -170.1],
    [-16.1, 0, 2.5, 81.4, -5.6, 15.8, -6.6, -170.1],
    [-11.7, 0, 2.5, 77.8, -5.6, 15, -6.6, -170.1],
    [-7.3, 0, 2.5, 73.7, -5.6, 14.7, -6.6, -170.1],
]))
COMMAND = GEOMETRIC.copy()
COMMAND[1:] += np.deg2rad([1.9, 0, 0, 2.1, 0, 0.7, -0.2, 0])


def smooth(x):
    x = np.clip(x, 0, 1)
    return x*x*(3-2*x)


def hermite(ts, values, t):
    ts, values = np.asarray(ts), np.asarray(values)
    slopes = np.zeros_like(values)
    slopes[1:-1] = (values[2:]-values[:-2]) / (ts[2:]-ts[:-2])[:, None]
    k = int(np.clip(np.searchsorted(ts, t, side="right")-1, 0, len(ts)-2))
    h = ts[k+1]-ts[k]
    x = np.clip((t-ts[k])/h, 0, 1)
    return ((2*x**3-3*x*x+1)*values[k] + (x**3-2*x*x+x)*h*slopes[k]
            + (-2*x**3+3*x*x)*values[k+1] + (x**3-x*x)*h*slopes[k+1])


class Grid:
    def __init__(self, directory, xy, dz):
        self.tables = []
        for suffix in ["basec", "m3c", "p3c"]:
            table = torch.load(directory/f"pregrasp_grid_v49_{suffix}.pt", map_location="cpu", weights_only=True)
            self.tables.append({k: v.numpy() for k, v in table.items()})
        self.xy, self.dz = np.asarray(xy, dtype=float), dz
        if abs(dz) > 0.03:
            raise ValueError("Table height outside available +/-3 cm grids")
        for axis, value in zip(("gx", "gy"), self.xy):
            g = self.tables[0][axis]
            # Include the 2.5 cm insertion lookup in the validated domain.
            if value < g[0] or value > g[-1] or (axis == "gx" and value-0.025 < g[0]):
                raise ValueError("Cube position outside available pregrasp grids")
        self.knots = self.lookup(self.xy)

    def lookup(self, xy):
        outputs = []
        for table in self.tables:
            indices, weights = [], []
            for axis, value in zip(("gx", "gy"), xy):
                g = table[axis]
                f = (np.clip(value, g[0], g[-1])-g[0])/(g[1]-g[0])
                i = int(np.clip(np.floor(f), 0, len(g)-2))
                indices.append(i)
                weights.append(f-i)
            i, j = indices
            a, b = weights
            q = table["dq"]
            outputs.append((1-a)*(1-b)*q[:, i, j] + a*(1-b)*q[:, i+1, j]
                           + (1-a)*b*q[:, i, j+1] + a*b*q[:, i+1, j+1])
        base, minus, plus = outputs
        return base + (abs(self.dz)/0.03)*((minus if self.dz < 0 else plus)-base)

    def shift(self, t):
        ts = np.r_[0, 2, self.tables[0]["ts"]]
        values = np.vstack([np.zeros(8), self.knots[0], self.knots])
        return hermite(ts, values, t)

    def path(self, t, geometric=False):
        return hermite(TIMES, GEOMETRIC if geometric else COMMAND, t) + self.shift(t)

    def insertion(self):
        delta = self.lookup(self.xy + [-0.025, 0])[-1] - self.knots[-1]
        delta[6] += np.deg2rad(12)
        return delta
