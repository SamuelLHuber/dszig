"""Observe DSPy 3.5 BoundingBox validation, schemas and adapter wire behavior."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import dspy
from pydantic import TypeAdapter
import reference as ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    assert dspy.__version__ == '3.5.0', dspy.__version__
    box_type = dspy.BoundingBox
    adapter = TypeAdapter(box_type)
    ordinary = {'top': 0.125, 'bottom': 0.875, 'left': 0.25, 'right': 0.75}
    cases = [ordinary, {'top': 0, 'bottom': 1, 'left': 0, 'right': 1}]
    for field in ordinary:
        for bad in (-0.001, 1.001, '0.25', False, None, [], float('nan'), float('inf'), -float('inf')):
            cases.append({**ordinary, field: bad})
        cases.append({key: val for key, val in ordinary.items() if key != field})
    cases.extend([{**ordinary, 'bottom': ordinary['top']}, {**ordinary, 'right': ordinary['left']},
                  {**ordinary, 'top': 0.9}, {**ordinary, 'left': 0.9},
                  {**ordinary, 'confidence': 0.99}, [], json.dumps(ordinary), None])
    observations = []
    for value in cases:
        try:
            model = adapter.validate_python(value)
            observed = {'valid': True, 'dump': model.model_dump(), 'format': model.format(),
                        'str': str(model), 'repr': repr(model)}
        except Exception as exc:
            observed = {'valid': False, 'error': type(exc).__name__,
                        'error_types': [item['type'] for item in exc.errors()]}
        observations.append({'input': value, **observed})
    class Locate(dspy.Signature):
        """Return the selected rectangle without clamping its edges."""
        reference: box_type = dspy.InputField()
        region: box_type = dspy.OutputField()
    box = adapter.validate_python(ordinary)
    adapters = {}
    for name, wire in [('chat', dspy.ChatAdapter()), ('json', dspy.JSONAdapter()), ('xml', dspy.XMLAdapter())]:
        output = json.dumps(ordinary)
        if name == 'chat':
            output = f'[[ ## region ## ]]\n{output}\n\n[[ ## completed ## ]]'
        elif name == 'json':
            output = json.dumps({'region': ordinary})
        else:
            output = '<region><top>0.125</top><bottom>0.875</bottom><left>0.25</left><right>0.75</right></region>'
        try:
            result = wire.parse(Locate, output)
            parsed = {'region': result['region'].model_dump()}
        except Exception as exc:
            parsed = {'error': type(exc).__name__, 'message': str(exc)}
        adapters[name] = {'messages': ref.messages_for(wire.format(Locate, demos=[], inputs={'reference': box})),
                          'raw': output, 'parsed': parsed}
    frozen = None
    try:
        box.top = 0.5
    except Exception as exc:
        frozen = [item['type'] for item in exc.errors()]
    source = Path(inspect.getsourcefile(box_type))
    receipt = {'dspy': dspy.__version__, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'schema': box_type.model_json_schema(), 'cases': observations, 'adapters': adapters,
               'frozen_assignment_errors': frozen}
    # Nonfinite inputs are diagnostic strings in this receipt, not pretend JSON numbers.
    def finite(value):
        if isinstance(value, float) and not __import__('math').isfinite(value):
            return {'nonfinite': repr(value)}
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [finite(item) for item in value]
        return value
    encoded = json.dumps(finite(receipt), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(encoded)
        print(json.dumps({'output': str(args.output), 'cases': len(cases),
                          'sha256': hashlib.sha256(encoded.encode()).hexdigest()}))
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
