import sys
import ruamel.yaml

foo = {
    'name': 'foo',
    'my_list': [{'foo': 'test', 'bar': 'test2'}, {'foo': 'test3', 'bar': 'test4'}],
    'hello': 'world'
}


yaml = ruamel.yaml.YAML()
yaml.indent(sequence=4, offset=2)
yaml.dump(foo, sys.stdout)