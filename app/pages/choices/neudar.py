from .neu import CAT_NEU_ANALYTES
from .dar import CAT_DAR_ANALYTES

CAT_NEUDAR_ANALYTES = [
    ('Analytes', tuple(set(CAT_NEU_ANALYTES[0][1]) & set(CAT_DAR_ANALYTES[0][1])))]

CAT_NEUDAR_MEMBER_C = (
    ('1', 'mother'),  # maps to maternal
    ('2', 'father'),
    ('3', 'child'),
)

CAT_NEUDAR_TIME_PERIOD = (
    ('9', 'any'),               # all time periods ploted together
    # ('0', 'early enrollment'),  # maps to 12G
    # ('1', 'enrollment'),        # maps to 24G
    # ('3', 'week 36/delivery'),  # maps to 6WP
)

ADDITIONAL_FEATURES = [('Categorical', (
    ('Outcome', 'Outcome'),
    ('Member_c', 'Family Member'),
    # ('TimePeriod', 'Collection Time'),
    ('CohortType', 'Cohort Type'),
)), 
('Outcomes', (
   ('SGA',  'SGA'),
    ('LGA',  'LGA'),
    ('birthWt',  'Birth Weight'),
    ('headCirc', 'headCirc'),
    ('birthLen','birthLen'),
    ('Outcome_weeks',  'Outcome Weeks'),
    ('Outcome',  'Outcome'),

)
)]

NEUDAR_FEATURE_CHOICES = CAT_NEUDAR_ANALYTES + ADDITIONAL_FEATURES
NEUDAR_CATEGORICAL_CHOICES = ADDITIONAL_FEATURES
