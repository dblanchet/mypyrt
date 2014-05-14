#! /usr/bin/python

# Ideas:
#
# - External scene description
# - Ray reflection limit
# - Handle several lights
# - Support various camera position and orientation
# - Add transparent Sphere
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

        # A point plus a vector makes another Point.
        return Point(self.x + v.x,
                     self.y + v.y,
                     self.z + v.z)


class Vector:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def _dot_product(self, v):
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
            return self._dot_product(o)

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

    def __sub__(self, o):
        return self.__add__(-o)

    def __neg__(self):
        return self * (-1.0)

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


class Line:

    def __init__(self, origin, direction):
        if not isinstance(origin, Point):
            raise ValueError('Expected Point as first arg, got', type(origin))

        if not isinstance(direction, Vector):
            raise ValueError('Expected Vector as second arg, got',
                    type(direction))

        self.origin = origin
        self.direction = direction

    def distance(self, point):
        # http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        if not isinstance(point, Point):
            raise ValueError('Expected Point as arg, got', type(point))

        op = self.origin - point
        proj_vect = op - (op * self.direction) * self.direction

        return proj_vect.norm()


Color = namedtuple('Color', 'red green blue')


class Ray(Line):

    MAX_BOUNCE_COUNT = 16

    def __init__(self, origin, direction, bounce_left=MAX_BOUNCE_COUNT):
        Line.__init__(self, origin, direction)
        self.bounce_left = bounce_left


class SceneObject:

    def visible_lights(self, point):
        if not isinstance(point, Point):
            raise ValueError('Expected Point as first arg, got', type(point))

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
            #
            # Exclude all light objects so they
            # do not stop sent rays.
            ray = Ray(point, ray_dir.normalize())
            touched = touched_objects(ray, exclude=scene.lights + [self])

            if touched:

                # Computed distance from point to light source.
                light_dist = ray_dir.norm()

                # Find closest object distance.
                obj_distances, _ = touched[0]

                # Check if the closest object stands
                # between point and light source.
                if 0 < obj_distances[0] < light_dist:
                    continue

            visible.append(light)

        return visible

    def normal_at(self, point):
        return None  # Trigger an exception if not overridden.

    def intersect(self, line):
        return None  # Trigger an exception if not overridden.

    def rendered_pixel(self, point, ray):
        return None  # Trigger an exception if not overridden.

    def apply_own_color(self, r, g, b):
        c = self.color
        return [int(r * c.red / 255.0),
                int(g * c.green / 255.0),
                int(b * c.blue / 255.0)]


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
        projection = l * oc
        discr = projection * projection - oc * oc + r * r

        if discr < 0:
            # No intersection.
            return -1,

        # Return all positive values, largest
        # only otherwise.
        longest = - (projection) + math.sqrt(discr)
        if longest > 0:

            shortest = - (projection) - math.sqrt(discr)
            if shortest > 0:
                return shortest, longest

        return longest,

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
        return self.adjust_with_lights(self.color, light_coeffs)


class ReflectingSphere(Sphere):

    def rendered_pixel(self, point, ray):

        # No reflection if bounce count is reached.
        if ray.bounce_left == 0:
            return Sphere.rendered_pixel(self, point, ray)

        # Coming ray is reflected according to
        # sphere surface normal and color is taken
        # from touched object mixed with Sphere's
        # own color.

        reflect_dir = ray.direction.reflected(self.normal_at(point))
        reflected = Ray(point, reflect_dir, ray.bounce_left - 1)

        r, g, b = send_ray(reflected, exclude=[self])

        # Apply sphere own color to reflected color.
        return self.apply_own_color(r, g, b)


class TransparentSphere(Sphere):

    def __init__(self, center, radius, color=Color(255, 255, 255),
            refr_idx=1.5):
        Sphere.__init__(self, center, radius, color)
        self.refr_idx = refr_idx

    def rendered_pixel(self, point, ray):

        # No reflection/transmission if bounce_left
        # count is reached.
        if ray.bounce_left == 0:
            return Sphere.rendered_pixel(self, point, ray)

        # Coming ray is reflected and transmitted
        # according to sphere surface normal and
        # internal material refractive index.
        #
        # Color is taken from touched objects mixed
        # with Sphere's own color.

        # Find ray projection on normal.
        n = self.normal_at(point)
        proj = ray.direction * n

        # Result depends on the ray entering
        # or exiting the sphere.
        entering = proj < 0

        if entering:

            # Set proper refraction indices.
            n1, n2 = 1.0, self.refr_idx

            # Reflected ray.
            reflect_dir = ray.direction.reflected(self.normal_at(point))
            reflected = Ray(point, reflect_dir, ray.bounce_left - 1)
            refl_r, refl_g, refl_b = send_ray(reflected, exclude=[self])

        else:

            # Set proper refraction indices.
            n1, n2 = self.refr_idx, 1.0

            # We do not want a reflected ray.
            refl_r, refl_g, refl_b = 0, 0, 0

        # Compute transmission intermediate value.
        #
        # http://www.cs.rpi.edu/~cutler/classes/advancedgraphics/
        #                                     F05/lectures/13_ray_tracing.pdf
        ratio = n1 / n2
        det = 1.0 - (ratio * ratio) * (1.0 - proj * proj)

        if det < 0:

            # When negative, reflection is total,
            # no transmission.
            #
            # Should never occur when n2 > n1.
            refl_coeff = 1
            trans_coeff = 0

            trans_r, trans_g, trans_b = 0, 0, 0

        else:

            # Keep reflection to a reasonable level.
            refl_coeff = 0.1
            trans_coeff = 1.0 - refl_coeff

            # Build transmitted ray.
            n_comp = ratio * proj - math.sqrt(det)
            direction = n_comp * n + ratio * ray.direction
            transmitted = Ray(point, direction.normalize(),
                    bounce_left=ray.bounce_left - 1)

            if entering:
                # When transmitted ray enters the sphere,
                # the only relevant object is the sphere
                # itself.
                excluded = scene.objects[:].remove(self)

                # Send transmitted ray.
                #
                # We want to ensure the intersection point
                # is the farthest, so it is not confused
                # with current "point" argument.
                trans_r, trans_g, trans_b = send_ray(transmitted, excluded,
                        farthest=True)
            else:
                # When transmitted ray exits the sphere,
                # all objects are relevant except itself.
                excluded = [self]

                # Send transmitted ray.
                trans_r, trans_g, trans_b = send_ray(transmitted, excluded)

        # Compute resulting color.
        #
        # Partition result between transmitted and reflected rays.
        r = refl_coeff * refl_r + trans_coeff * trans_r
        g = refl_coeff * refl_g + trans_coeff * trans_g
        b = refl_coeff * refl_b + trans_coeff * trans_b

        # Apply sphere own color to reflected/transmitted color.
        c = self.color
        return [int(r * c.red / 255),
                int(g * c.green / 255),
                int(b * c.blue / 255)]


class Light(Sphere):

    def __init__(self, position, radius=1.0, color=Color(255, 255, 255)):
        Sphere.__init__(self, position, radius, color)
        self.position = position

    def rendered_pixel(self, point, ray):
        # Light does not stop the ray.
        #
        # It adds its own light as a halo over
        # object/background located behind.

        transmitted = Ray(point, ray.direction, ray.bounce_left)
        r, g, b = send_ray(transmitted, exclude=[self])

        dist = ray.distance(self.position)
        coeff = 1.0 - pow(dist / self.radius, 0.5)

        # Apply light own color.
        c = self.color
        return [int(min(r + c.red * coeff, 255)),
                int(min(g + c.green * coeff, 255)),
                int(min(b + c.blue * coeff, 255))]


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
            return -1,

        dist = (p - o) * n / denum
        return dist,

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
        x = + ss.x * x_scr / image_size.x - ss.x / 2
        y = - ss.y * y_scr / image_size.y + ss.y / 2
        z = self.position.z + self.direction.z * self.screen_dist

        screen_point = Point(x, y, z)
        ray_dir = (screen_point - camera.position).normalize()
        return Ray(camera.position, ray_dir)

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
reflecting = ReflectingSphere(
        center=Point(-6.0, 4.0, -3.0),
        radius=3.0,
        color=Color(0, 255, 128))
blueSphere = Sphere(
        center=Point(8.0, -2.0, -5.0),
        radius=3.0,
        color=Color(0, 72, 255))
transparent = TransparentSphere(
        center=Point(-4.0, -1.0, 1.0),
        radius=1.5,
        refr_idx=10.5)

tiledFloor = Plane(
        point=Point(0.0, -4.0, 0.0),
        normal=Vector(0.0, 1.0, 0.0))

light = Light(position=Point(-5.0, 10.0, 10.0))
objects = [
        yellowSphere, redSphere, blueSphere,
        reflecting,
        transparent,
        tiledFloor,
        light]

lights = [light]

scene = Scene(objects=objects, lights=lights)


# Image setup.
image_size = Point(1024, 768, 0)


def touched_objects(ray, exclude=None):
    if not isinstance(ray, Ray):
        raise ValueError('Expected Ray as first arg, got', type(ray))

    touched = []

    for obj in scene.objects:

        # We may want to exclude objects:
        # when computing cast shadows, we want to
        # exclude current illuminated object.
        if exclude is not None and obj in exclude:
            continue

        # Distance is returned as a tuples of
        # distances. This tuple may contain one
        # or more values. Values are sorted.
        d = obj.intersect(ray)

        # We do not care about objects
        # behind camera.
        if d[0] > 0:
            touched.append((d, obj))

    # Sort touched object according to distance.
    #
    # Tuples are sorted by first item, then second item, etc.
    touched.sort()

    return touched


def send_ray(ray, exclude=None, farthest=False):
    if not isinstance(ray, Ray):
        raise ValueError('Expected Ray as first arg, got', type(ray))

    # Find out scene objects reach by the ray.
    touched = touched_objects(ray, exclude)
    if not touched:
        return BACKGROUND_COLOR

    # If several objects intersected,
    # take first one.
    distances, obj = touched[0]

    # Compute the point where ray and objects met.
    #
    # Caller may want the closest or the farthest point.
    dist_idx = -1 if farthest else 0
    point = ray.origin + distances[dist_idx] * ray.direction

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
