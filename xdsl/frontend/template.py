# TODO: calling this a template is very misleading, we probably should instead
# call it `evaluate` or `specialize` or similar. If we do change the name, we have
# to also adapt error messages and tests!
def template(*params):
    """
    Marks function as a template. Parameters specify which function arguments are used to
    instantiate the template.
    """
    def decorate(f):
        pass
    return decorate
