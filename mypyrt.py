#! /usr/bin/python

import sys
import math

from collections import namedtuple

import png


class Point:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, p):
        if not isinstance(p, Point):
            raise ValueError('Expected a Point, got a', type(p))

        return Vector(self.x - p.x,
                      self.y - p.y,
                      self.z - p.z)


class Vector:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def _vector_mul(self, v):
        return   self.x * v.x \
               + self.y * v.y \
               + self.z * v.z

    def _const_mul(self, k):
        return Vector(k * self.x,
                      k * self.y,
                      k * self.z)

    def __mul__(self, o):

        if isinstance(o, Vector):
            return self._vector_mul(o)

        try:
            k = float(o)
            return self._const_mul(k)
        except ValueError:
            pass

        raise ValueError('Expected a Point or a number, got a', type(o))

    def __div__(self, k):
        val = float(k)  # Let conversion fail for incorrect types.
        return self * (1 / val)

    def norm(self):
        return math.sqrt(self * self)

    def normalize(self):
        return self / self.norm()


Line = namedtuple('Line', 'origin direction')
Color = namedtuple('Color', 'red green blue')


class Sphere:

    def __init__(self, center, radius, color=Color(255, 255, 255)):
        if not isinstance(center, Point):
            raise ValueError('Expected a Point as first arg, got a',
                    type(center))
        self.center = center
        self.radius = float(radius)
        self.color = color

    def intersect(self, line):
        o = line.origin
        l = line.direction

        c = self.center
        r = self.radius

        oc = o - c
        mul1 = l * oc
        discr = mul1 * mul1 - oc * oc + r * r

        if discr < 0:
            return -1

        shortest_dist = - (l * oc) - math.sqrt(discr)
        return shortest_dist

    def rendered_pixel(self, point, ray):
        origin = ray.origin

        pc = (point - self.center).normalize()
        op = (origin - point).normalize()
        coeff = abs(pc * op)

        c = self.color
        return [int(c.red * coeff),
                int(c.green * coeff),
                int(c.blue * coeff)]


class Plane:

    def __init__(self, point, normal):
        if not isinstance(point, Point):
            raise ValueError('Expected Point as first arg, got',
                    type(point))

        if not isinstance(normal, Vector):
            raise ValueError('Expected Vector as second arg, got',
                    type(normal))

        self.point = point
        self.normal = normal

    def intersect(self, line):
        o = line.origin
        l = line.direction

        p = self.point
        n = self.normal

        denum = l * n
        if denum == 0:
            return -1

        dist = (p - o) * n / denum
        return dist

    def rendered_pixel(self, point, ray):
        # Plane is darker with the distance.
        if point.z > 0:
            attenuation = 0.0
        else:
            attenuation = - point.z / 50.0

        # Tiled rendering.
        if int(math.floor(point.x / 2) + math.floor(point.z / 2)) & 1:
            base = 255
        else:
            base = 128

        return [int(base / (1.0 + attenuation))] * 3


BACKGROUND_COLOR = Color(0, 0, 64)  # Dark Blue

camera_pos = Point(0.0, 0.0, 10.0)
camera_dir = Vector(0.0, 0.0, -1.0)
screen_dist = 3.0  # Distance from camera to screen window.
screen_size = Point(4.0, 3.0, 0)

image_size = Point(320, 240, 0)

scene = [
        Sphere(Point(0.0, 0.0, 0.0), 3.0, Color(255, 255, 0)),
        Sphere(Point(-1.0, 2.0, 2.0), 1.5, Color(255, 0, 0)),
        Sphere(Point(-5.0, -4.0, -5.0), 3.0, Color(0, 255, 128)),
        Sphere(Point(6.0, 3.0, -5.0), 3.0, Color(0, 72, 255)),
        Plane(Point(0.0, 4.0, 0.0), Vector(0.0, 1.0, 0.0))  # Floor
        ]


def make_ray(x_scr, y_scr):
    x = screen_size.x * x_scr / image_size.x - screen_size.x / 2
    y = screen_size.y * y_scr / image_size.y - screen_size.y / 2
    z = camera_pos.z + camera_dir.z * screen_dist

    screen_point = Point(x, y, z)
    return Line(camera_pos, (screen_point - camera_pos).normalize())


def send_ray(ray):
    touched = []
    for obj in scene:
        d = obj.intersect(ray)
        if d > 0:
            touched.append((d, obj))

    if not touched:
        return BACKGROUND_COLOR

    touched.sort()  # Tuples are sorted by first item, then second item, etc.

    l = ray.direction
    o = camera_pos

    d, obj = touched[0]
    p = Point(o.x + l.x * d, o.y + l.y * d, o.z + l.z * d)

    return obj.rendered_pixel(p, ray)


def main(argv=None):
    pixels = []
    for y in range(image_size.y):
        for x in range(image_size.x):
            ray = make_ray(x, y)
            color = send_ray(ray)
            pixels.extend(color)

    with open('result.png', 'wb') as f:
        w = png.Writer(image_size.x, image_size.y)
        w.write_array(f, pixels)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
