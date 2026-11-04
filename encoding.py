import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder



qual = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

ordinal_maps = {
    'Lot Shape':      {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3},
    'Land Slope':     {'Gtl': 0, 'Mod': 1, 'Sev': 2},
    'Exter Qual':     qual,
    'Exter Cond':     qual,
    'Bsmt Qual':      qual,
    'Bsmt Cond':      qual,
    'Heating QC':     qual,
    'Kitchen Qual':   qual,
    'Fireplace Qu':   qual,
    'Garage Qual':    qual,
    'Garage Cond':    qual,
    'Pool QC':        qual,
    'Bsmt Exposure':  {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
    'BsmtFin Type 1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'BsmtFin Type 2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'Garage Finish':  {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
    'Functional':     {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4,
                       'Min2': 5, 'Min1': 6, 'Typ': 7},
    'Paved Drive':    {'N': 0, 'P': 1, 'Y': 2},
    'Central Air':    {'N': 0, 'Y': 1},
    'Electrical':     {'Mix': 0, 'FuseP': 1, 'FuseF': 2, 'FuseA': 3, 'SBrkr': 4},
    'Fence':          {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4},
    'Utilities':    {'ELO': 0, 'NoSeWa': 1, 'NoSewr': 2, 'AllPub': 3},
}

nominal_cols = [
    'MS SubClass', 'MS Zoning', 'Alley', 'Land Contour', 'Lot Config',
    'Neighborhood', 'Condition 1', 'Condition 2', 'Bldg Type', 'House Style', 'Roof Style',
    'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type', 'Foundation',
    'Garage Type', 'Misc Feature', 'Sale Type', 'Sale Condition', 'Roof Matl', 'Heating', 'Street'
]

# for the target range

price_bins = [0, 100_000, 130_000, 180_000, 250_000, 400_000, np.inf] 
price_labels = ['<100k', '100-130k', '130-180k', '180-250k', '250-400k', '>=400k']
n_classes = 6

target_cols = ['SalePrice', 'Price Class']

def make_target(df):
    return pd.cut(
        df['SalePrice'],
        bins=price_bins,
        labels=False,
        right=False
    ).astype(int)


# def encode_ordinals(df):
#     df = df.copy()

#     for col, mapping in ordinal_maps.items():
#         if col in df.columns:

#             unknown = set(df[col].dropna().unique()) - set(mapping)

#             if unknown:
#                 raise ValueError(f'Unknown values in {col}: {unknown}')

#             df[col] = df[col].map(mapping)

#     return df


# def encode_onehot():
#     preprocessor = ColumnTransformer(
#         transformers=[
#             (
#                 'nominal',
#                 OneHotEncoder(handle_unknown='ignore'),
#                 nominal_cols
#             )
#         ],
#         remainder='passthrough'
#     )

#     return preprocessor

def encode_features(df):
    df = df.copy()

    ordinal_cols = [
        col for col in ordinal_maps if col in df.columns
    ]

    nominal_features = [
        col for col in nominal_cols if col in df.columns
    ]

    if ordinal_cols:
        ordinal_categories = [
            [
                category
                for category, value in sorted(
                    ordinal_maps[col].items(),
                    key=lambda item: item[1]
                )
            ]
            for col in ordinal_cols
        ]

        ordinal_encoder = OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown='use_encoded_value',
            unknown_value=np.nan,
            encoded_missing_value=np.nan
        )

        df[ordinal_cols] = ordinal_encoder.fit_transform(
            df[ordinal_cols]
        )

    if nominal_features:
        onehot_encoder = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        )

        encoded_array = onehot_encoder.fit_transform(
            df[nominal_features].fillna('Unknown').astype(str)
        )

        encoded_names = onehot_encoder.get_feature_names_out(
            nominal_features
        )

        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoded_names,
            index=df.index
        )

        df = df.drop(columns=nominal_features)
        df = pd.concat([df, encoded_df], axis=1)

    return df