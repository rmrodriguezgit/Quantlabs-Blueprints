"""yfinance Data node: option validation, Yahoo limits and frame normalisation (yfinance mocked)."""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from backend.engine.nodes import yfinance_nodes as yn


def fake_download(ticker, **kw):
    if ticker == 'EMPTY':
        return pd.DataFrame()
    intraday = kw['interval'] in yn.MAX_DAYS
    idx = pd.date_range('2026-11-17 14:30', periods=50, freq='1min', tz='UTC') if intraday else pd.bdate_range('2026-01-02', periods=50)
    close = 100 + np.arange(50.0)
    df = pd.DataFrame({'Open': close - .5, 'High': close + 1, 'Low': close - 1, 'Close': close, 'Volume': np.r_[np.zeros(10), np.full(40, 1000.0)]}, index=idx)
    if not kw['auto_adjust']:
        df['Adj Close'] = close * .98
    if kw['actions']:
        df['Dividends'] = 0.0
        df['Stock Splits'] = 0.0
        df.iloc[20, df.columns.get_loc('Stock Splits')] = 4.0
    return df


class YFinanceDataTests(unittest.TestCase):
    def setUp(self):
        yn._CACHE.clear()
        patcher = mock.patch('yfinance.download', side_effect=fake_download)
        self.dl = patcher.start()
        self.addCleanup(patcher.stop)

    def test_daily_defaults(self):
        frame, rep = yn.yfinance_data({'ticker': 'spy'})
        self.assertEqual(list(frame.columns), ['Open', 'High', 'Low', 'Close', 'Volume'])
        self.assertIsNone(frame.index.tz)
        self.assertEqual(frame.attrs['ticker'], 'SPY')
        kw = self.dl.call_args.kwargs
        self.assertEqual((kw['period'], kw['interval'], kw['auto_adjust'], kw['multi_level_index']), ('1y', '1d', True, False))
        self.assertAlmostEqual(rep['metrics']['retorno_total'], 149 / 100 - 1)

    def test_intraday_timezone_and_prepost_note(self):
        frame, rep = yn.yfinance_data({'ticker': 'QQQ', 'interval': '1m', 'period': '5d', 'prepost': True, 'timezone': 'America/New_York'})
        self.assertIsNone(frame.index.tz)
        self.assertEqual(frame.index[0], pd.Timestamp('2026-11-17 09:30'))     # 14:30 UTC → 09:30 New York
        self.assertTrue(any('pre/post' in n for n in rep['notes']))

    def test_unadjusted_with_actions_and_dates(self):
        frame, rep = yn.yfinance_data({'ticker': 'AAPL', 'range_mode': 'start_end', 'start': '2020-08-20', 'end': '2020-09-05', 'auto_adjust': False, 'actions': True})
        self.assertIn('AdjClose', frame)
        self.assertIn('StockSplits', frame)
        kw = self.dl.call_args.kwargs
        self.assertEqual((kw['start'], kw['end']), ('2020-08-20', '2020-09-05'))
        self.assertNotIn('period', kw)
        self.assertTrue(any('1 splits' in n for n in rep['notes']))

    def test_yahoo_limits_and_bad_options(self):
        bad = [
            {'interval': '1m', 'period': '1mo'},                      # 1m only ~7 days
            {'interval': '5m', 'period': 'max'},
            {'interval': '1h', 'period': '5y'},                       # 1h only 730 days
            {'interval': '15m', 'range_mode': 'start_end', 'start': '2020-01-01', 'end': '2020-01-10'},  # older than 60 days
            {'back_adjust': True},                                    # needs auto_adjust off
            {'prepost': True, 'interval': '1d'},
            {'interval': '7m'}, {'period': '3y'}, {'ticker': 'SPY QQQ'},
            {'range_mode': 'start_end', 'start': '2024-02-01', 'end': '2024-01-01'},
        ]
        for p in bad:
            with self.subTest(p=p), self.assertRaises(ValueError):
                yn.yfinance_data({'ticker': 'SPY', **p})
        self.dl.assert_not_called()                                   # rejected before any download
        yn.validate({'ticker': 'SPY', 'interval': '1h', 'period': '1y'})
        yn.validate({'ticker': 'SPY', 'interval': '5m', 'period': '1mo'})

    def test_unfinished_last_bar_is_dropped(self):
        now = pd.Timestamp.now(tz='UTC').floor('min')
        idx = pd.date_range(end=now, periods=30, freq='1min')

        def live(ticker, **kw):
            c = 100 + np.arange(30.0)
            return pd.DataFrame({'Open': c, 'High': c, 'Low': c, 'Close': c, 'Volume': np.full(30, 10.0)}, index=idx)
        self.dl.side_effect = live
        frame, rep = yn.yfinance_data({'ticker': 'QQQ', 'interval': '1m', 'period': '1d'})
        self.assertEqual(len(frame), 29)
        self.assertTrue(any('aún no había cerrado' in n for n in rep['notes']))
        yn._CACHE.clear()
        frame, _ = yn.yfinance_data({'ticker': 'QQQ', 'interval': '1m', 'period': '1d', 'drop_incomplete': False})
        self.assertEqual(len(frame), 30)

    def test_empty_answer_is_explained(self):
        with self.assertRaisesRegex(ValueError, 'no devolvió datos'):
            yn.yfinance_data({'ticker': 'EMPTY'})

    def test_cache_avoids_second_download(self):
        yn.yfinance_data({'ticker': 'SPY'})
        yn.yfinance_data({'ticker': 'SPY'})
        self.assertEqual(self.dl.call_count, 1)


if __name__ == '__main__':
    unittest.main()
