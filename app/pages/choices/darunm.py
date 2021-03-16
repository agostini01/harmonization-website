from .dar import CAT_DAR_ANALYTES
from .unm import CAT_UNM_ANALYTES

CAT_DARUNM_ANALYTES = [
    ('Analytes', tuple(set(CAT_UNM_ANALYTES[0][1]) & set(CAT_DAR_ANALYTES[0][1])))]

CAT_DARUNM_MEMBER_C = (
    ('1', 'mother'),  # maps to maternal
    ('2', 'father'),
    ('3', 'child'),
)

CAT_DARUNM_TIME_PERIOD = (
    ('9', 'any'),               # all time periods ploted together
    ('0', 'early enrollment'),  # maps to 12G
    ('1', 'enrollment'),        # maps to 24G
    ('3', 'week 36/delivery'),  # maps to 6WP
)

ADDITIONAL_FEATURES = [('Categorical', (
    ('Outcome', 'Outcome'),
    ('Member_c', 'Family Member'),
    ('TimePeriod', 'Collection Time'),
    ('CohortType', 'Cohort Type'),
)), 
('Outcomes', (
    ('SGA',  'SGA'),
    ('LGA',  'LGA'),
    ('birthWt',  'Birth Weight'),
    ('Outcome_weeks',  'Outcome Weeks'),
    ('Outcome',  'Outcome'),

)
)]

DARUNM_FEATURE_CHOICES = CAT_DARUNM_ANALYTES + ADDITIONAL_FEATURES
DARUNM_CATEGORICAL_CHOICES = ADDITIONAL_FEATURES
