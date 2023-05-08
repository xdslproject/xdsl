# TODO: Clean this up after prototyping
class Frontend:
    map = {}

    def add_mapping(self, func, operation):
        # adds a mapping from a function to an operation
        self.map[func] = operation


def frontend_op(frontend: Frontend, operation):
    def decorator(func):
        frontend.add_mapping(func, operation)
        return func

    return decorator
