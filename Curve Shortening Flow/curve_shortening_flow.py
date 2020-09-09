import numpy as np
import matplotlib.pyplot as plt


class CurveShorteningFlow:
    def __init__(self, curve, variant='A', step=0.001, duration=150, draw_rate=0.1):
        self._number_of_steps = duration
        self._step = step
        self._draw_rate = draw_rate
        self._curve = np.array(curve)

        if variant not in ['A', 'B', 'C', 'D']:
            raise AttributeError

        self._curvature_definition = variant

    def _get_size(self):
        return len(self._curve)

    def _get_vertex(self, i):
        return self._curve[i % self._get_size()]

    def _get_turning_angle(self, i):
        e1 = self._get_vertex(i) - self._get_vertex(i - 1)
        e2 = self._get_vertex(i + 1) - self._get_vertex(i)

        return np.math.atan2(np.linalg.det([e1, e2]), np.dot(e1, e2))

    def _get_curvature(self, i):
        if self._curvature_definition == 'A':
            return self._get_turning_angle(i)
        elif self._curvature_definition == 'B':
            return 2 * np.math.sin(self._get_turning_angle(i) / 2)
        elif self._curvature_definition == 'C':
            return 2 * np.math.tan(self._get_turning_angle(i) / 2)
        elif self._curvature_definition == 'D':
            e = self._get_vertex(i + 1) - self._get_vertex(i - 1)
            w = np.linalg.norm(e)

            return 2 * np.math.sin(self._get_turning_angle(i)) / w

    def _get_normal(self, i):
        sign = np.sign(self._get_turning_angle(i))

        if self._curvature_definition in ['A', 'B', 'C']:
            e1 = self._get_vertex(i - 1) - self._get_vertex(i)
            e2 = self._get_vertex(i + 1) - self._get_vertex(i)

            return sign * (e1 + e2) / np.linalg.norm(e1 + e2)

        elif self._curvature_definition == 'D':
            triangle = np.array([self._get_vertex(i - 1), self._get_vertex(i),
                                self._get_vertex(i + 1)])
            e = self._get_circumcenter(triangle) - self._get_vertex(i)

            if e is None:
                return np.array([1., 0.])

            return sign * e / np.linalg.norm(e)

    def _get_flow(self):
        flow = [self._get_curvature(i) * self._get_normal(i) for i in range(self._get_size())]

        return np.array(flow)

    def _draw(self):
        polygon = plt.Polygon(self._curve, fill=None)
        plt.gca().add_line(polygon)

    @staticmethod
    def _show():
        plt.show()

    def transform(self):
        for i in range(self._number_of_steps):
            if i % int(self._draw_rate * self._number_of_steps) == 0:
                self._draw()

            self._curve += self._step * self._get_flow()

        self._show()

    @staticmethod
    def _get_circumcenter(triangle):
        d = 2 * np.sum(np.cross(triangle[:, 0], triangle[:, 1]))
        norms = np.array([np.dot(triangle[i], triangle[i]) for i in range(3)])
        u = [np.sum(np.cross(triangle[:, i], norms)) for i in range(2)]

        if np.isclose(d, 0):
            return (triangle[0] + triangle[1]) / 2

        return np.array([-u[1], u[0]]) / d

    def is_round(self):
        if self._get_size() < 3:
            return False

        circumcenter = self._get_circumcenter(self._curve[:3])
        radius = np.linalg.norm(self._get_vertex(1) - circumcenter)

        for i in range(self._get_size()):
            if not np.isclose(np.linalg.norm(self._get_vertex(i) - circumcenter), radius):
                return False

            if np.sign(self._get_turning_angle(i) * self._get_turning_angle(i - 1)) < 0:
                return False

        return True

    def get_mass_center(self):
        weights = [0 for _ in range(self._get_size())]

        for i in range(self._get_size()):
            e1 = self._get_vertex(i - 1) - self._get_vertex(i)
            e2 = self._get_vertex(i + 1) - self._get_vertex(i)

            weights[i] = (np.linalg.norm(e1) + np.linalg.norm(e2)) / 2

        return np.average(self._curve, axis=0, weights=weights)

    def get_total_curvature(self):
        total_curvature = 0

        for i in range(self._get_size()):
            total_curvature += self._get_curvature(i)

        return total_curvature
