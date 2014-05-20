

def parse_scene_setup(data, namespace):

    def parse(json_obj):

        # Check if object requires a dedicated constructor.
        try:
            classname = json_obj['type']
        except TypeError:
            return json_obj

        # Remove now useless type information.
        del json_obj['type']

        # Parse arguments if necessary.
        kwargs = {key: parse(value) for key, value in json_obj.items()}

        # Create instance.
        return namespace[classname](**kwargs)

    lights = [parse(obj) for obj in data['lights']]
    objects = [parse(obj) for obj in data['objects']]

    return lights, objects
