import unittest
import numpy as np
import pandas as pd
from backend.engine.nodes.transform_nodes import transform, check_fit_boundary
from backend.engine.nodes.curve_nodes import monthly_curve

class TransformTests(unittest.TestCase):
    def setUp(self):
        self.frame=pd.DataFrame({'Close':np.arange(120.,240.),'Volume':np.ones(120)},index=pd.date_range('2020-01-01',periods=120))

    def test_http_schema_accepts_catalog(self):
        from backend.server import Graph
        from backend.engine.dag_runner import PINS
        for kind in PINS:
            with self.subTest(kind=kind):
                Graph.model_validate({'nodes':[{'id':'node','data':{'kind':kind,'params':{}}}],'edges':[]})

    def test_select_resample_coordinate_center(self):
        selected=transform(self.frame,{'columns':'Close'},'select_columns')
        monthly=transform(selected,{'columns':'Close','frequency':'ME','exclude_incomplete':False},'resample')
        pd.testing.assert_frame_equal(monthly,selected.resample('ME').last())
        x=transform(monthly,{},'coordinate')
        result=transform(x,{'fit_fraction':1},'center')
        self.assertAlmostEqual(result.Centered_Close.mean(),0)
        self.assertEqual((result.Coordinate.iloc[0],result.Coordinate.iloc[-1]),(-2,2))
        self.assertNotIn('Centered_Close',x)
        pd.testing.assert_series_equal(result.Close,monthly.Close)

    def test_scaler_fit_only_leading_rows_and_constant(self):
        first=transform(self.frame,{'columns':'Close,Volume','fit_fraction':.5},'scale_columns')
        changed=self.frame.copy();changed.iloc[60:,0]*=100
        second=transform(changed,{'columns':'Close,Volume','fit_fraction':.5},'scale_columns')
        np.testing.assert_allclose(first.Scaled_Close[:60],second.Scaled_Close[:60])
        self.assertAlmostEqual(first.Scaled_Close[:60].std(ddof=0),1)
        self.assertTrue((first.Scaled_Volume==0).all())
        self.assertEqual(first.attrs['transform_audits'][0]['fit_rows'],60)
        check_fit_boundary(first,['Scaled_Close'],self.frame.index[59])
        with self.assertRaisesRegex(ValueError,'held-out'):check_fit_boundary(first,['Scaled_Close'],self.frame.index[58])
        check_fit_boundary(first,['Close'],self.frame.index[1])
        minimum=transform(self.frame,{'columns':'Close','fit_fraction':.5,'method':'minmax'},'scale_columns')
        self.assertAlmostEqual(minimum.Scaled_Close.iloc[59],1)
        self.assertGreater(minimum.Scaled_Close.iloc[-1],1)

    def test_invalid_parameters(self):
        for kind,params in [('select_columns',{'columns':'missing'}),('resample',{'frequency':'invalid'}),('coordinate',{'start':2,'stop':-2}),('coordinate',{'output_column':'Close'}),('center',{'fit_fraction':0}),('scale_columns',{'prefix':''})]:
            with self.subTest(kind=kind),self.assertRaises(ValueError):transform(self.frame,params,kind)

    def test_legacy_monthly_reuses_operations(self):
        daily=pd.DataFrame({'Close':np.arange(1100.)+100},index=pd.date_range('2020-01-01',periods=1100))
        actual=monthly_curve(daily,{'exclude_incomplete_month':False})
        expected=transform(transform(daily,{'exclude_incomplete':False},'resample'),{},'coordinate')
        pd.testing.assert_frame_equal(actual,expected)
