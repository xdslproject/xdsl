
# LANG=C find xdsl -type f -name "*.py" -exec sed -E -i '' -f remove_annotated.sed {} +
s/(.+): ParameterDef\[(.+)\]/\1 = param_def(\2)/g
