#! /usr/bin/python

import sys
import math

from collections import namedtuple

import png


Point = namedtuple('Point', 'x y z')
Vect = namedtuple('Vect', 'x y z')
Sphere = namedtuple('Sphere', 'center radius')
Line = namedtuple('Line', 'origin direction')

camera_pos = Point(0.0, 0.0, 10.0)
camera_dir = Vect(0.0, 0.0, -1.0)
screen_dist = 3.0  # Distance from camera to screen window.
screen_size = Point(4.0, 3.0, 0)

image_size = Point(64, 48, 0)

scene_objects = [Sphere(Point(0.0, 0.0, 0.0), 3.0)]


def mul(v1, v2):
    return   v1.x * v2.x \
           + v1.y * v2.y \
           + v1.z * v2.z


def minus(p1, p2):
    return Vect(p2.x - p1.x,
                p2.y - p1.y,
                p2.z - p1.z)


def norm(v):
    return math.sqrt(mul(v, v))


def normalize(v):
    n = norm(v)
    return Vect(v.x / n, v.y / n, v.z / n)


def sphere_intersection(line, sphere):
    o = line.origin
    l = line.direction
    c = sphere.center
    r = sphere.radius

    oc = minus(o, c)
    mul1 = mul(l, oc)
    discr = mul1 * mul1 - mul(oc, oc) + r * r

    return discr >= 0


def make_ray(x_scr, y_scr):
    x = screen_size.x * x_scr / image_size.x - screen_size.x / 2
    y = screen_size.y * y_scr / image_size.y - screen_size.y / 2
    z = camera_pos.z - screen_dist

    screen_point = Point(x, y, z)
    return Line(camera_pos, normalize(minus(screen_point, camera_pos)))


def send_ray(ray):
    if sphere_intersection(ray, scene_objects[0]):
        return [255] * 3
    else:
        return [0] * 3


def main(argv=None):
    pixels = []
    for y in range(image_size.y):
        line = []
        for x in range(image_size.x):
            ray = make_ray(x, y)
            color = send_ray(ray)
            line.extend(color)
        pixels.append(line)

    with open('result.png', 'wb') as f:
        w = png.Writer(image_size.x, image_size.y)
        w.write(f, pixels)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
