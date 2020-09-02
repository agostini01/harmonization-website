import pandas as pd
import numpy as np
from datasets.models import RawNEU


def get_dataframe():
    """Returns a pandas DataFrame with the correct
    format for the generic plotting functions."""

    # First is necessary to pivot the raw NEU dataset so it matches
    # the requested features.

    # This queries the RawNEU dataset and excludes some of the values
    # TODO - Should we drop NaN here?
    df = pd.DataFrame.from_records(
        RawNEU.objects.
        # exclude(Creat_Corr_Result__lt=-1000).
        # exclude(Creat_Corr_Result__isnull=True).
        values()
    )


    # Pivoting the table and reseting index
    numerical_values = 'Result'
    columns_to_indexes = ['PIN_Patient', 'TimePeriod', 'Member_c', 'Outcome']
    categorical_to_columns = ['Analyte']
    indexes_to_columns = ['Member_c', 'TimePeriod', 'Outcome']
    df = pd.pivot_table(df, values=numerical_values,
                        index=columns_to_indexes,
                        columns=categorical_to_columns,
                        aggfunc=np.average)
    df = df.reset_index(level=indexes_to_columns)
    # TODO - Should we drop NaN here?

    # After pivot
    # Analyte     TimePeriod Member_c       BCD  ...      UTMO       UTU       UUR
    # PIN_Patient                                ...
    # A0000M               1        1  1.877245  ...  0.315638  1.095520  0.424221
    # A0000M               3        1  1.917757  ...  0.837639  4.549155  0.067877
    # A0001M               1        1  1.458583  ...  0.514317  1.262910  1.554346
    # A0001M               3        1  1.365789  ...  0.143302  1.692582  0.020716
    # A0002M               1        1  1.547669  ...  0.387643  0.988567  1.081877

    df['CohortType'] = 'NEU'
    df['TimePeriod'] = pd.to_numeric(df['TimePeriod'], errors='coerce')

    return df
