"""The Inspector shows catalog defaults but sends only edited values: Python defaults must match the UI."""
import re
import unittest
from pathlib import Path

from backend.dl_tools.core import registry

TS = (Path(__file__).resolve().parents[2] / 'src/renderer/types/blueprint.ts').read_text()


def ui_defaults(kind: str) -> dict[str, object]:
    start = TS.index(f'  {kind}: {{')
    block = TS[start:TS.index('\n  },\n', start)]
    out: dict[str, object] = {}
    for key, value in re.findall(r'\{ key: "(\w+)", label: "[^"]*", value: ("[^"]*"|true|false|-?[\d.]+)', block):
        out[key] = value.strip('"') if value.startswith('"') else value == 'true' if value in ('true', 'false') else float(value)
    return out


class DefaultsMatch(unittest.TestCase):
    def test_python_defaults_equal_inspector_defaults(self):
        for kind in ('dl_tabular_forecast', 'dl_sequence_classifier', 'dl_anomaly_autoencoder', 'dl_recurrent_compare'):
            fields = registry.get(kind).Params.model_fields
            for key, ui in ui_defaults(kind).items():
                if key not in fields or key == 'layers':
                    continue
                py = fields[key].default
                ui_cast = type(py)(ui) if not isinstance(py, bool) and py is not None else ui
                self.assertEqual(py, ui_cast, f'{kind}.{key}: Python {py!r} ≠ Inspector {ui!r}')


    def test_adapter_defaults_equal_inspector(self):
        from backend.engine.nodes.dl_tool_nodes import RECURRENT_FEATURES
        self.assertEqual(ui_defaults('dl_recurrent_compare')['features'], RECURRENT_FEATURES)
        self.assertEqual(len(RECURRENT_FEATURES.split(',')), 11)


if __name__ == '__main__':
    unittest.main()
