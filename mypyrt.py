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


Sphere = namedtuple('Sphere', 'center radius')
Line = namedtuple('Line', 'origin direction')
Plane = namedtuple('Place', 'point normal')

camera_pos = Point(0.0, 0.0, 10.0)
camera_dir = Vector(0.0, 0.0, -1.0)
screen_dist = 3.0  # Distance from camera to screen window.
screen_size = Point(4.0, 3.0, 0)

image_size = Point(320, 240, 0)

scene = [
        Sphere(Point(0.0, 0.0, 0.0), 3.0),
        Sphere(Point(1.0, 2.0, 2.0), 1.5),
        Sphere(Point(-4.0, -3.0, -5.0), 3.0),
        Plane(Point(0.0, 4.0, 0.0), Vector(0.0, 1.0, 0.0))  # Floor
        ]


def sphere_intersection(line, sphere):
    o = line.origin
    l = line.direction

    c = sphere.center
    r = sphere.radius

    oc = o - c
    mul1 = l * oc
    discr = mul1 * mul1 - oc * oc + r * r

    if discr < 0:
        return -1

    dist = - (l * oc) - math.sqrt(discr)
    return dist


def place_intersection(line, plane):
    o = line.origin
    l = line.direction

    p = plane.point
    n = plane.normal

    denum = l * n
    if denum == 0:
        return -1

    dist = (p - o) * n / denum
    return dist


def make_ray(x_scr, y_scr):
    x = screen_size.x * x_scr / image_size.x - screen_size.x / 2
    y = screen_size.y * y_scr / image_size.y - screen_size.y / 2
    z = camera_pos.z + camera_dir.z * screen_dist

    screen_point = Point(x, y, z)
    return Line(camera_pos, (screen_point - camera_pos).normalize())


def send_ray(ray):
    touched = []
    for obj in scene:
        if isinstance(obj, Sphere):
            d = sphere_intersection(ray, obj)
        if isinstance(obj, Plane):
            d = place_intersection(ray, obj)
        if d > 0:
            touched.append((d, obj))

    if not touched:
        return [0] * 3
        #return [0, 127, 0]  # DEBUG

    touched.sort()  # Tuples are sorted by first item, then second item, etc.

    l = ray.direction
    o = camera_pos

    d, obj = touched[0]
    p = Point(o.x + l.x * d, o.y + l.y * d, o.z + l.z * d)

    if isinstance(obj, Sphere):
        sp = (p - obj.center).normalize()
        op = (o - p).normalize()

        return [int(255 * abs(sp * op))] * 3

    if isinstance(obj, Plane):
        if p.z > 0:
            attenuation = 0.0
        else:
            attenuation = - p.z / 50.0
        if int(math.floor(p.x / 2) + math.floor(p.z / 2)) & 1:
            base = 255
        else:
            base = 128
        return [int(base / (1.0 + attenuation))] * 3


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
