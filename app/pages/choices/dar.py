CAT_DAR_ANALYTES = [('Analytes', (
    # Analyate acronym and name,                    Mapping in the dar DB
    ('UAG', ' Silver - Urine'),                     # Ag in ug/L
    ('UAL', ' Aluminium - Urine'),                  # Al in ug/L
    ('UCR',  'Chromium - Urine'),                   # Cr in ug/L
    ('UCU',  'Copper - Urine'),                     # Cu in ug/L
    ('UFE',  'Iron - Urine'),                       # Fe in ug/L
    ('UNI',  'Niquel - Urine'),                     # Ni in ug/L
    ('UVA',  'Vanadium - Urine'),                   # V in ug/L
    ('UZN',  'Zinc - Urine'),                       # Zn in ug/L
    # ('BCD',  'Cadmium - Blood'),
    # ('BHGE', 'Ethyl Mercury - Blood'),
    # ('BHGM', 'Methyl Mercury - Blood'),
    # ('BMN',  'Manganese - Blood'),
    # ('BPB',  'Lead - Blood'),
    # ('BSE',  'Selenium - Blood'),
    # ('IHG',  'Inorganic Mercury - Blood'),
    # ('THG',  'Mercury Total - Blood'),
    # ('SCU',  'Copper - Serum'),
    # ('SSE',  'Selenium - Serum'),
    # ('SZN',  'Zinc - Serum'),
    ('UAS3', 'Arsenous (III) acid - Urine'),        # As in ug/L
    # ('UAS5', 'Arsenic (V) acid - Urine'),
    ('UASB', 'Arsenobetaine - Urine'),              # AsB in ug/L
    # ('UASC', 'Arsenocholine - Urine'),
    ('UBA',  'Barium - Urine'),                     # Ba in ug/L
    ('UBE',  'Beryllium - Urine'),                  # Be in ug/L
    ('UCD',  'Cadmium - Urine'),                    # Cd in ug/L
    ('UCO',  'Cobalt - Urine'),                     # Co in ug/L
    ('UCS',  'Cesium - Urine'),                     # Cs in ug/L
    ('UDMA', 'Dimethylarsinic Acid - Urine'),       # DMA in ug/L
    ('UHG',  'Mercury - Urine'),                    # Hg in ug/L
    # ('UIO',  'Iodine - Urine'),
    ('UMMA', 'Monomethylarsinic Acid - Urine'),     # MMA in ug/L
    ('UMN',  'Manganese - Urine'),                  # Mn in ug/L
    ('UMO',  'Molybdenum - Urine'),                 # Mo in ug/L
    ('UPB',  'Lead - Urine'),                       # PB in ug/L
    # ('UPT',  'Platinum - Urine'),
    ('USB',  'Antimony - Urine'),                   # Sb in ug/L
    ('USN',  'Tin - Urine'),                        # Sn in ug/L
    ('USR',  'Strontium - Urine'),                  # Sr in ug/L
    ('UTAS', 'Arsenic Total - Urine'),              # iAs in ug/L
    ('UTL',  'Thallium - Urine'),                   # Tl in ug/L
    # ('UTMO', 'Trimethylarsine - Urine')
    ('UTU',  'Tungsten - Urine'),                   # W in ug/L
    ('UUR',  'Uranium - Urine'),                    # U in ug/L

))]

CAT_DAR_MEMBER_C = (
    ('1', 'mother'),  # maps to maternal
    ('2', 'father'),
    ('3', 'child'),
)

# Available at the DAR DB
# CAT_DAR_TIME_PERIOD = (
#     ('12G', 'week 12 gestational'),
#     ('24G', 'week 24 gestational'),
#     ('6WP', 'week 6 portpartun'),
#     ('6MP', 'month 6 postpartum'),
#     ('1YP', 'year 1 postpartum'),
#     ('2YP', 'year 2 postpartum'),
#     ('3YP', 'year 3 postpartum'),
#     ('5YP', 'year 5 postpartum'),
# )

CAT_DAR_TIME_PERIOD = (
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
    ('headCirc', 'headCirc'),
    ('Outcome_weeks',  'Outcome Weeks'),
    ('Outcome',  'Outcome'),

)
)]


DAR_FEATURE_CHOICES = CAT_DAR_ANALYTES + ADDITIONAL_FEATURES
DAR_CATEGORICAL_CHOICES = ADDITIONAL_FEATURES
