import numpy as np
import pandas as pd

none_means_absent = [
    'Alley', 'Fence', 'Misc Feature', 'Pool QC', 'Fireplace Qu', 'Mas Vnr Type',
    'Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond',
    'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2',
]

zero_means_absent = [
    'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF',
    'Bsmt Full Bath', 'Bsmt Half Bath', 'Garage Area', 'Garage Cars',
]

drop_cols = ['Order', 'PID']

low_variance_cols = ['Street', 'Utilities', 'Condition 2', 'Roof Matl', 'Heating']

def load_raw():
    return pd.read_csv('AmesHousing.csv', sep='\t', keep_default_na=False, na_values=[''])

def clean(df, drop_low_variance=True):
    df = df.copy()

    df['MS SubClass'] = df['MS SubClass'].astype(str) # it's a string, not a number

    for col in none_means_absent:
        df[col] = df[col].replace({'NA': 'None', '': 'None'}).fillna('None')
    for col in zero_means_absent:
        df[col] = df[col].fillna(0)

        # --- contradictions in the source data
    
    # finished area of type 2 but no type recorded
    df.loc[(df['BsmtFin SF 2'] > 0) & (df['BsmtFin Type 2'] == 'None'),
           'BsmtFin Type 2'] = 'Rec'

    # veneer type says None but there is veneer area: trust the type
    df.loc[(df['Mas Vnr Type'] == 'None') & (df['Mas Vnr Area'] > 0),
           'Mas Vnr Area'] = 0

    # a basement exists but has no exposure recorded: No is the common value
    has_bsmt = df['Bsmt Qual'] != 'None'
    df.loc[has_bsmt & (df['Bsmt Exposure'] == 'None'), 'Bsmt Exposure'] = 'No'

    # a garage type with no garage anywhere else
    df.loc[(df['Garage Type'] != 'None') & (df['Garage Area'] == 0),
           'Garage Type'] = 'None'

    # one house is remodelled before it was built
    bad_remod = df['Year Remod/Add'] < df['Year Built']
    df.loc[bad_remod, 'Year Remod/Add'] = df.loc[bad_remod, 'Year Built']

    # a real garage with no details: most common values for a detached one
    no_detail = (df['Garage Type'] != 'None') & (df['Garage Finish'] == 'None')
    df.loc[no_detail, ['Garage Finish', 'Garage Qual', 'Garage Cond']] = ['Unf', 'TA', 'TA']

    # one garage is dated 2207, which is a typo
    bad_year = df['Garage Yr Blt'] > df['Yr Sold']
    df.loc[bad_year, 'Garage Yr Blt'] = df.loc[bad_year, 'Year Built']

    # --- missing Electrical value
    df['Electrical'] = df['Electrical'].replace('', np.nan)
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

    # --- what is left is actually missing, flagged not filled
    # Lot Frontage is imputed later, inside the training fold
    df['Lot Frontage Missing'] = df['Lot Frontage'].isna().astype(int)

    # Garage Yr Blt is missing exactly for the houses without a garage
    df['Garage Yr Missing'] = df['Garage Yr Blt'].isna().astype(int)

    # --- columns that should not be used
    drop = list(drop_cols)
    if drop_low_variance:
        drop += low_variance_cols
    df = df.drop(columns=[c for c in drop if c in df.columns])

    return df

def cleaning_report(before, after):
    return pd.DataFrame({
        'rows': [len(before), len(after)],
        'columns': [before.shape[1], after.shape[1]],
        'duplicate rows': [before.duplicated().sum(), after.duplicated().sum()],
        'cells missing': [before.isna().sum().sum(), after.isna().sum().sum()],
        'text columns': [(before.dtypes == object).sum() + (before.dtypes == 'str').sum(),
                         (after.dtypes == object).sum() + (after.dtypes == 'str').sum()],
    }, index=['raw', 'cleaned'])



