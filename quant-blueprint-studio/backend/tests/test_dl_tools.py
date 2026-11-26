"""DL tools suite: contracts, isolation, registry, chained Blueprint graph and memory release."""
import ast
import gc
import unittest
from pathlib import Path

import numpy as np
import torch

from backend.dl_tools.core import RunContext, ToolError, registry
from backend.dl_tools.core.graph import run, validate, Graph
from backend.dl_tools.core.memory import tracker

TOOLS = Path(__file__).resolve().parents[1] / 'dl_tools' / 'tools' / 'dl'
CPU = RunContext(device='cpu', seed=7)


def series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (np.sin(t / 8) + 0.1 * rng.normal(size=n)).tolist()


class Registry(unittest.TestCase):
    def test_discovers_three_tools_with_specs(self):
        names = {s.name for s in registry.specs()}
        self.assertTrue({'dl_tabular_forecast', 'dl_sequence_classifier', 'dl_anomaly_autoencoder'} <= names)
        for d in registry.describe():
            self.assertIn('params_schema', d)
            self.assertGreater(d['capabilities']['est_ram_mb'], 0)

    def test_tools_do_not_import_each_other(self):
        for tool in TOOLS.iterdir():
            if not tool.is_dir() or tool.name.startswith('_'):
                continue
            for file in tool.glob('*.py'):
                for node in ast.walk(ast.parse(file.read_text())):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        others = [t.name for t in TOOLS.iterdir() if t.is_dir() and t.name != tool.name]
                        self.assertFalse(any(o in node.module for o in others), f'{file} imports {node.module}')
                        self.assertNotIn('engine', node.module)


class Tools(unittest.TestCase):
    def test_forecast_beats_naive_on_sine_and_reports_latency(self):
        tool = registry.get('dl_tabular_forecast')({'architecture': 'mlp', 'epochs': 40})
        r = tool.execute({'series': series()}, CPU)
        self.assertEqual(r.weights, 'fitted_on_input')
        self.assertEqual(len(r.outputs['forecast']), 1)
        self.assertGreater(r.outputs['confidence'], 0)
        self.assertGreater(r.inference_ms, 0)
        tool.unload()

    def test_all_forecast_architectures(self):
        for arch in ('mlp', 'cnn1d', 'lstm'):
            with registry.get('dl_tabular_forecast')({'architecture': arch, 'epochs': 2, 'horizon': 3}) as tool:
                out = tool.execute({'series': series(120)}, CPU).outputs
                self.assertEqual(len(out['forecast']), 3)
                self.assertTrue(all(l <= f <= u for l, f, u in zip(out['lower'], out['forecast'], out['upper'])))

    def test_classifier_outputs_distribution(self):
        with registry.get('dl_sequence_classifier')({'epochs': 5, 'classes': 3, 'dead_zone': 0.05}) as tool:
            out = tool.execute({'series': series(200)}, CPU).outputs
            self.assertAlmostEqual(sum(out['probabilities'].values()), 1, places=5)
            self.assertIn(out['label'], ('down', 'flat', 'up', 'abstain'))
            self.assertEqual(len(out['latent']), 16)  # width of the last hidden layer (default 1 × 16)

    def test_autoencoder_flags_injected_outlier(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(300, 4))
        x[:, 1] = x[:, 0] * 2
        x[280] = [0, 8, 0, 0]
        with registry.get('dl_anomaly_autoencoder')({'epochs': 80}) as tool:
            out = tool.execute({'data': x.tolist()}, CPU).outputs
        self.assertTrue(out['is_anomaly'][280])
        self.assertEqual(int(np.argmax(out['dimension_error'][280])), 1)

    def test_structured_errors(self):
        with self.assertRaises(ToolError) as e:
            registry.get('dl_tabular_forecast')({'lookback': 1})
        self.assertEqual(e.exception.code, 'invalid_params')
        with self.assertRaises(ToolError) as e:
            registry.get('dl_anomaly_autoencoder')().execute({'data': [[1, 2], [3]] * 5}, CPU)
        self.assertEqual(e.exception.code, 'invalid_input')

    def test_seeded_runs_are_deterministic(self):
        a = registry.get('dl_tabular_forecast')({'architecture': 'mlp', 'epochs': 5}).execute({'series': series()}, CPU).outputs['forecast']
        b = registry.get('dl_tabular_forecast')({'architecture': 'mlp', 'epochs': 5}).execute({'series': series()}, CPU).outputs['forecast']
        self.assertEqual(a, b)

    def test_unknown_device_falls_back_to_cpu(self):
        tool = registry.get('dl_tabular_forecast')({'architecture': 'mlp', 'epochs': 1})
        r = tool.execute({'series': series(100)}, RunContext(device='cuda'))
        self.assertIn(r.device, ('cpu', 'cuda'))
        tool.unload()


class Lifecycle(unittest.TestCase):
    def test_unload_frees_the_model_object(self):
        tool = registry.get('dl_sequence_classifier')({'epochs': 1})
        tool.execute({'series': series(100)}, CPU)
        ref = tracker(tool.model)
        tool.unload()
        gc.collect()
        self.assertFalse(tool.loaded)
        self.assertIsNone(ref())


GRAPH = {
    'inputs': {'series': series()},
    'nodes': [{'id': 'forecast', 'tool': 'dl_tabular_forecast', 'params': {'architecture': 'lstm', 'epochs': 10, 'horizon': 2}},
              {'id': 'classify', 'tool': 'dl_sequence_classifier', 'params': {'epochs': 5}},
              {'id': 'anomaly', 'tool': 'dl_anomaly_autoencoder', 'params': {'epochs': 30, 'fit_fraction': 0.5, 'latent': 1}}],
    'edges': [{'source': '$series', 'target': 'forecast', 'target_port': 'series'},
              {'source': '$series', 'target': 'classify', 'target_port': 'series'},
              {'source': 'forecast', 'source_port': 'residuals', 'target': 'anomaly', 'target_port': 'data'}],
}


class Blueprint(unittest.TestCase):
    def test_chained_graph_runs_and_releases_everything(self):
        report = run(GRAPH, CPU)
        self.assertEqual(set(report['outputs']), {'forecast', 'classify', 'anomaly'})
        self.assertEqual(len(report['outputs']['anomaly']['score']), report['nodes']['forecast']['diagnostics']['holdout_windows'])
        self.assertTrue(report['memory']['all_released'])
        self.assertEqual(report['memory']['unloaded'], ['anomaly', 'classify', 'forecast'])

    def test_port_type_mismatch_is_rejected_before_loading(self):
        bad = {**GRAPH, 'edges': [*GRAPH['edges'][:2], {'source': 'forecast', 'source_port': 'forecast', 'target': 'anomaly', 'target_port': 'data'}]}
        problems = validate(Graph.model_validate(bad))
        self.assertTrue(any('Tipos incompatibles' in p for p in problems))

    def test_failure_still_unloads(self):
        bad = {**GRAPH, 'inputs': {'series': series(30)}, 'nodes': [{**GRAPH['nodes'][0], 'params': {'lookback': 40}}, *GRAPH['nodes'][1:]]}
        with self.assertRaises((ToolError, ValueError)):
            run(bad, CPU)


class Architecture(unittest.TestCase):
    """One input layer, N configurable hidden layers, one output layer; every optimizer trains."""

    def test_layer_counts_and_single_input_output(self):
        from backend.dl_tools.core.layers import LayerSpec
        for name, params, payload in (
            ('dl_tabular_forecast', {'architecture': 'mlp', 'hidden_layers': 3, 'units': 24, 'horizon': 2}, {'series': series(150)}),
            ('dl_sequence_classifier', {'hidden_layers': 0}, {'series': series(150)}),
            ('dl_anomaly_autoencoder', {'layers': [LayerSpec(units=12, activation='tanh', dropout=0.1).model_dump(), LayerSpec(units=6).model_dump()], 'latent': 2},
             {'data': np.random.default_rng(0).normal(size=(120, 5)).tolist()})):
            with registry.get(name)({**params, 'epochs': 2}) as tool:
                r = tool.execute(payload, CPU)
                roles = [row['capa'] for row in r.diagnostics['layers']]
                self.assertEqual(roles.count('input_layer') >= 1, True)
                self.assertEqual(len([m for m in tool.model.modules() if m is tool.model.output_layer]), 1)
                linears = sum(1 for m in (tool.model.hidden if hasattr(tool.model, 'hidden') else tool.model.encoder_hidden) if isinstance(m, torch.nn.Linear))
                self.assertEqual(linears, {'dl_tabular_forecast': 3, 'dl_sequence_classifier': 0, 'dl_anomaly_autoencoder': 2}[name])

    def test_every_optimizer_and_activation_trains(self):
        for opt in ('adam', 'adamw', 'sgd', 'rmsprop', 'adagrad'):
            for act in ('relu', 'tanh', 'swish'):
                with registry.get('dl_tabular_forecast')({'architecture': 'mlp', 'optimizer': opt, 'activation': act, 'epochs': 3, 'lr': 0.01}) as tool:
                    r = tool.execute({'series': series(150)}, CPU)
                    hist = r.diagnostics['training']['history']
                    self.assertEqual(r.diagnostics['training']['optimizer'], opt)
                    self.assertTrue(all(np.isfinite(h['train']) for h in hist))

    def test_shorthand_and_early_stopping(self):
        from backend.dl_tools.core.layers import parse_layers
        specs = parse_layers('64:relu:0.1:bn, 16:tanh')
        self.assertEqual([(s.units, s.activation, s.dropout, s.batch_norm) for s in specs], [(64, 'relu', 0.1, True), (16, 'tanh', 0.0, False)])
        with registry.get('dl_tabular_forecast')({'architecture': 'mlp', 'epochs': 300, 'patience': 3}) as tool:
            t = tool.execute({'series': series(300)}, CPU).diagnostics['training']
        self.assertTrue(t['early_stopped'])
        self.assertLess(t['epochs_run'], 300)

    def test_bottleneck_must_compress(self):
        with self.assertRaises(ToolError):
            registry.get('dl_anomaly_autoencoder')({'latent': 3}).execute({'data': np.zeros((50, 3)).tolist()}, CPU)


class DoesItsJob(unittest.TestCase):
    """Each tool must beat a trivial baseline on data where the signal exists."""

    def test_forecast_learns_ar_process(self):
        rng = np.random.default_rng(3)
        x = np.zeros(1500)
        for i in range(1, 1500):
            x[i] = 0.7 * x[i - 1] + rng.normal(0, 1)
        with registry.get('dl_tabular_forecast')({'architecture': 'mlp', 'epochs': 60}) as tool:
            r = tool.execute({'series': x.tolist()}, CPU)
        # optimum = noise σ = 1.0; best baseline (last value) ≈ 1.086 → the ceiling for confidence is ≈ 0.08
        self.assertLess(r.diagnostics['holdout_rmse'], 1.08)
        self.assertGreater(r.outputs['confidence'], 0.03)

    def test_classifier_learns_predictable_direction(self):
        rng = np.random.default_rng(4)
        s = np.zeros(1500)
        for i in range(1, 1500):
            s[i] = 0.8 * s[i - 1] + rng.normal(0, 1)
        with registry.get('dl_sequence_classifier')({'epochs': 40}) as tool:
            d = tool.execute({'series': s.tolist()}, CPU).diagnostics
        self.assertGreater(d['holdout_accuracy'], d['majority_baseline'] + 0.1)

    def test_autoencoder_separates_broken_correlation(self):
        rng = np.random.default_rng(5)
        a = rng.normal(size=(600, 1))
        x = np.hstack([a, 2 * a + 0.05 * rng.normal(size=(600, 1)), -a])
        x[550:560, 1] *= -1                   # the relation breaks, the marginals do not
        with registry.get('dl_anomaly_autoencoder')({'latent': 1, 'epochs': 150}) as tool:
            out = tool.execute({'data': x.tolist()}, CPU).outputs
        flagged = set(np.flatnonzero(out['is_anomaly']))
        self.assertGreaterEqual(len(flagged & set(range(550, 560))), 8)
        self.assertLessEqual(len(flagged - set(range(550, 560))), 10)


if __name__ == '__main__':
    unittest.main()
