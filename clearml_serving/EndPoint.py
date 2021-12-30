from attr import attrib, attrs, asdict


@attrs
class EndPoint(object):
    serving_url = attrib(type=str)
    model_ids = attrib(type=list)
    model_project = attrib(type=str)
    model_name = attrib(type=str)
    model_tags = attrib(type=list)
    model_config_blob = attrib(type=str, default=None)
    max_num_revisions = attrib(type=int, default=None)
    versions = attrib(type=dict, default={})

    def as_dict(self):
        return asdict(self)
