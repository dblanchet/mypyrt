#! /usr/bin/python

# Ideas:
#
# - External scene description
# - Ray reflection limit
# - Handle several lights
# - Support various camera position and orientation
# - Add transparent Sphere
# - Make light a full visible scene object
# - Give color to lights

import sys
import math

from collections import namedtuple

import multiprocessing
from time import time

import png


class Point:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, p):
        if not isinstance(p, Point):
            raise ValueError('Expected a Point, got a', type(p))

        # Difference between points makes a Vector.
        return Vector(self.x - p.x,
                      self.y - p.y,
                      self.z - p.z)

    def __add__(self, v):
        if not isinstance(v, Vector):
            raise ValueError('Expected a Vector, got a', type(v))

        return Point(self.x + v.x,
                     self.y + v.y,
                     self.z + v.z)


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

        # If argument is a Vector, return dot product.
        if isinstance(o, Vector):
            return self._vector_mul(o)

        try:
            # If argument is a num scalar, return a Vector.
            k = float(o)
            return self._const_mul(k)
        except ValueError:
            pass

        raise ValueError('Expected a Point or a number, got a', type(o))

    def __rmul__(self, o):
        return self.__mul__(o)

    def __add__(self, o):
        if not isinstance(o, Vector):
            raise ValueError('Expected a Vector, got a', type(o))

        return Vector(self.x + o.x,
                      self.y + o.y,
                      self.z + o.z)

    def __div__(self, k):
        # Division is correct with num scalar only.
        val = float(k)  # Let conversion fail for incorrect types.
        return self * (1 / val)

    def norm(self):
        return math.sqrt(self * self)

    def normalize(self):
        return self / self.norm()

    def reflected(self, normal):
        if not isinstance(normal, Vector):
            raise ValueError('Expected a Vector, got a', type(normal))

        # http://www.3dkingdoms.com/weekly/weekly.php?a=2
        #
        # Return inverted symetrical based on arg vector.
        direction = (- 2 * (self * normal)) * normal + self
        return direction.normalize()


Line = namedtuple('Line', 'origin direction')
Color = namedtuple('Color', 'red green blue')
Light = namedtuple('Light', 'position')


class SceneObject:

    def visible_lights(self, point):
        # Return list of scene lights that are directly
        # in light of sight from the given object point.
        visible = []

        for light in scene.lights:
            ray_dir = light.position - point

            # Own shadow.
            #
            # Check if surface normal at given point
            # is oriented towards this light.
            if self.normal_at(point) * ray_dir < 0:
                continue

            # Cast shadow.
            #
            # Check if light source is visible from
            # considered point.
            ray = Line(point, ray_dir.normalize())
            touched = touched_objects(ray, exclude=[self])

            if touched:

                # Computed distance from point to light source.
                light_dist = ray_dir.norm()

                # Find closest object distance.
                touched.sort()
                obj_dist, _ = touched[0]

                # Check if the closest object stands
                # between point and light source.
                if 0 < obj_dist < light_dist:
                    continue

            visible.append(light)

        return visible

    def normal_at(self, point):
        return None  # Trigger an exception if not overridden.


class Sphere(SceneObject):

    def __init__(self, center, radius, color=Color(255, 255, 255)):
        if not isinstance(center, Point):
            raise ValueError('Expected a Point as first arg, got a',
                    type(center))
        self.center = center
        self.radius = float(radius)
        self.color = color

    def normal_at(self, point):
        return (point - self.center).normalize()

    def intersect(self, line):
        # http://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
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

        # Change brightness according to visible lights.
        lights = self.visible_lights(point)

        if len(lights) > 0:
            # Take first light source only.
            light = lights[0]

            # Change brightness according to ray
            # to normal angle.
            light_dir = (light.position - point).normalize()
            attenuation = abs(light_dir * self.normal_at(point))

            coeff = AMBIANT_LIGHT + (255 - AMBIANT_LIGHT) * attenuation
        else:
            coeff = AMBIANT_LIGHT

        coeff /= 255.0

        # Apply brightness variation to sphere color.
        c = self.color
        return [int(c.red * coeff),
                int(c.green * coeff),
                int(c.blue * coeff)]


class ReflectingSphere(Sphere):

    def rendered_pixel(self, point, ray):
        # Coming ray is reflected according to
        # sphere surface normal and color is taken
        # from touched object mixed with Sphere's
        # own color.

        reflect_dir = ray.direction.reflected(self.normal_at(point))

        reflected = Line(point, reflect_dir)
        r, g, b = send_ray(reflected, exclude=[self])

        c = self.color
        return [int(r * c.red / 255),
                int(g * c.green / 255),
                int(b * c.blue / 255)]


class Plane(SceneObject):

    def __init__(self, point, normal):
        if not isinstance(point, Point):
            raise ValueError('Expected Point as first arg, got',
                    type(point))

        if not isinstance(normal, Vector):
            raise ValueError('Expected Vector as second arg, got',
                    type(normal))

        self.point = point
        self.normal = normal

    def normal_at(self, point):
        return self.normal

    def intersect(self, line):
        # http://en.wikipedia.org/wiki/Line-plane_intersection
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
            base = 64

        # Change brightness according to visible lights.
        lights = self.visible_lights(point)
        coeff = AMBIANT_LIGHT + (255 - AMBIANT_LIGHT) * len(lights)
        coeff /= 255.0

        # RGB are all treated the same, making
        # light and dark gray tiles.
        return [int(base / (1.0 + attenuation) * coeff)] * 3


AMBIANT_LIGHT = 64
BACKGROUND_COLOR = Color(0, 0, AMBIANT_LIGHT)  # Dark Blue


class Camera:

    def __init__(self, position, direction, screen_dist, screen_size):
        self.position = position
        self.direction = direction
        self.screen_dist = screen_dist
        self.screen_size = screen_size

    def make_primary_ray(self, x_scr, y_scr):
        ss = self.screen_size
        x =   ss.x * x_scr / image_size.x - ss.x / 2
        y = - ss.y * y_scr / image_size.y + ss.y / 2
        z = self.position.z + self.direction.z * self.screen_dist

        screen_point = Point(x, y, z)
        ray_dir = (screen_point - camera.position).normalize()
        return Line(camera.position, ray_dir)

camera = Camera(
        position=Point(0.0, 0.0, 10.0),
        direction=Vector(0.0, 0.0, -1.0),
        screen_dist=3.0,  # Distance from camera to screen window.
        screen_size=Point(4.0, 3.0, 0))


# Scene setup.
Scene = namedtuple('Scene', 'objects lights')

yellowSphere = Sphere(
        center=Point(0.0, 0.0, 0.0),
        radius=3.0,
        color=Color(255, 255, 0))
redSphere = Sphere(
        center=Point(-1.0, -2.0, 2.0),
        radius=1.5,
        color=Color(255, 0, 0))
greenSphere = ReflectingSphere(
        center=Point(-6.0, 4.0, -3.0),
        radius=3.0,
        color=Color(0, 255, 128))
blueSphere = Sphere(
        center=Point(8.0, -2.0, -5.0),
        radius=3.0,
        color=Color(0, 72, 255))

tiledFloor = Plane(
        point=Point(0.0, -4.0, 0.0),
        normal=Vector(0.0, 1.0, 0.0))

objects = [yellowSphere, redSphere, greenSphere, blueSphere, tiledFloor]
lights = [Light(position=Point(-5.0, 10.0, 10.0))]

scene = Scene(objects=objects, lights=lights)


# Image setup.
image_size = Point(1024, 768, 0)


def touched_objects(ray, exclude=None):
    touched = []

    for obj in scene.objects:

        # We may want to exclude objects:
        # when computing cast shadows, we want to
        # exclude current illuminated object.
        if exclude is not None and obj in exclude:
            continue

        d = obj.intersect(ray)

        # We do not care about objects
        # behind camera.
        if d > 0:
            touched.append((d, obj))

    return touched


def send_ray(ray, exclude=None):

    # Find out scene objects reach by the ray.
    touched = touched_objects(ray, exclude)
    if not touched:
        return BACKGROUND_COLOR

    # If several objects intersected,
    # take first one.
    #
    # Tuples are sorted by first item, then second item, etc.
    touched.sort()
    distance, obj = touched[0]

    # Compute the point where ray and objects met.
    point = ray.origin + distance * ray.direction

    # Ask touched object for a color.
    return obj.rendered_pixel(point, ray)


def render_line(y):
    line = []

    # Send a ray for each image pixel
    # and find the corresponding color.
    for x in range(image_size.x):
        ray = camera.make_primary_ray(x, y)
        color = send_ray(ray)
        line.extend(color)

    return line


def main(argv=None):

    subprocesses = int(argv[1]) if len(argv) > 1 else None

    # Compute pixel colors.
    start = time()  # Take a time reference before.

    if subprocesses == 0:
        # Use current process for line rendering.
        lines = map(render_line, range(image_size.y))
    else:
        # Distribute line rendering over available CPUs.
        pool = multiprocessing.Pool(processes=subprocesses)
        lines = pool.map(render_line, range(image_size.y))
        pool.close()
        pool.join()

        # Doc says multiprocessing defaults to machine CPU count.
        if subprocesses is None:
            subprocesses = multiprocessing.cpu_count()

    # Print timing information.
    elapsed = time() - start
    px_count = image_size.x * image_size.y
    proc_count = subprocesses if subprocesses > 0 else 1
    print '%d pixels with %d subprocesses in %d seconds ' \
            '(%d px/sec, %d px/proc/sec)' % (
            px_count, subprocesses, elapsed,
            px_count // elapsed, px_count // elapsed // proc_count)

    # Write pixels to easily read file format.
    with open('result.png', 'wb') as f:
        w = png.Writer(image_size.x, image_size.y)
        w.write(f, lines)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
