def process_kwargs(instance, kwargs, acceptable_kws=()):
    assert isinstance(acceptable_kws, (list, tuple))

    instance_kwargs = {
        key: kwargs[key]
        for key in kwargs if key in acceptable_kws
    }

    instance.__dict__.update(instance_kwargs)
