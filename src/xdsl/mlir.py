module = __import__('mlir', fromlist=['*'])

if hasattr(module, '__all__'):
    all_names = module.__all__
else:
    all_names = [name for name in dir(module) if not name.startswith('_')]

ir = __import__('mlir.ir', fromlist=['*'])

globals().update({name: getattr(module, name) for name in all_names})