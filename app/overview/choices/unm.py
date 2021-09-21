CAT_UNM_ANALYTES = [('Analytes', (
    ('BCD',  'Cadmium - Blood'),
    ('BHGE', 'Ethyl Mercury - Blood'),
    ('BHGM', 'Methyl Mercury - Blood'),
    ('BMN',  'Manganese - Blood'),
    ('BPB',  'Lead - Blood'),
    ('BSE',  'Selenium - Blood'),
    ('IHG',  'Inorganic Mercury - Blood'),
    ('THG',  'Mercury Total - Blood'),
    ('SCU',  'Copper - Serum'),
    ('SSE',  'Selenium - Serum'),
    ('SZN',  'Zinc - Serum'),
    ('UAS3', 'Arsenous (III) acid - Urine'),
    ('UAS5', 'Arsenic (V) acid - Urine'),
    ('UASB', 'Arsenobetaine - Urine'),
    ('UASC', 'Arsenocholine - Urine'),
    ('UBA',  'Barium - Urine'),
    ('UBE',  'Beryllium - Urine'),
    ('UCD',  'Cadmium - Urine'),
    ('UCO',  'Cobalt - Urine'),
    ('UCS',  'Cesium - Urine'),
    ('UDMA', 'Dimethylarsinic Acid - Urine'),
    ('UHG',  'Mercury - Urine'),
    ('UIO',  'Iodine - Urine'),
    ('UMMA', 'Monomethylarsinic Acid - Urine'),
    ('UMN',  'Manganese - Urine'),
    ('UMO',  'Molybdenum - Urine'),
    ('UPB',  'Lead - Urine'),
    ('UPT',  'Platinum - Urine'),
    ('USB',  'Antimony - Urine'),
    ('USN',  'Tin - Urine'),
    ('USR',  'Strontium - Urine'),
    ('UTAS', 'Arsenic Total - Urine'),
    ('UTL',  'Thallium - Urine'),
    ('UTMO', 'Trimethylarsine - Urine'),
    ('UTU',  'Tungsten -  Urine'),
    ('UUR',  'Uranium - Urine'),


))]

CAT_UNM_MEMBER_C = (
    ('1', 'mother'),
    ('2', 'father'),
    ('3', 'child'),
)

CAT_UNM_TIME_PERIOD = (
    ('9', 'any'),               # all time periods ploted together
    ('1', 'enrollment'),
    ('3', 'week 36/delivery'),
)

ADDITIONAL_FEATURES = [('Categorical', (
    ('Outcome', 'Outcome'),
    ('Member_c', 'Family Member'),
    ('TimePeriod', 'Collection Time'),
    ('CohortType', 'Cohort Type'),
)), 
('Outcomes', (
    ('birthWt',  'Birth Weight'),
    ('Outcome_weeks',  'Outcome Weeks')
)
)]

UNM_FEATURE_CHOICES = CAT_UNM_ANALYTES + ADDITIONAL_FEATURES
#UNM_FEATURE_CHOICES = CAT_UNM_ANALYTES
UNM_CATEGORICAL_CHOICES = ADDITIONAL_FEATURES
