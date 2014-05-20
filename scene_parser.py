#! /usr/bin/python


def parse_scene_setup(data, namespace):

    def parse(json_obj):

        # Check if object requires a dedicated constructor.
        try:
            classname = json_obj['type']
        except TypeError:
            return json_obj

        # Get constructor arguments.
        kwargs = {}
        for key, value in json_obj.items():
            if key == 'type':
                continue
            kwargs[key] = parse(value)

        # Create instance.
        return namespace[classname](**kwargs)

    def parse_named(json_obj):

        # First level object are named.
        obj_name, content = json_obj.items()[0]
        instance = parse(content)
        instance.name = obj_name
        return instance

    lights = [parse_named(obj) for obj in data['lights']]
    objects = [parse_named(obj) for obj in data['objects']]

    return lights, objects
