CAT_HAR_ANALYTES = [('Analytes', (
    # Analyate acronym and name,
    # DAR only # ('UAG', ' Silver - Urine'),
    # DAR only # ('UAL', ' Aluminium - Urine'),
    # DAR only # ('UCR',  'Chromium - Urine'),
    # DAR only # ('UCU',  'Copper - Urine'),
    # DAR only # ('UFE',  'Iron - Urine'),
    # DAR only # ('UNI',  'Niquel - Urine'),
    # DAR only # ('UVA',  'Vanadium - Urine'),
    # DAR only # ('UZN',  'Zinc - Urine'),
    # UNM only # ('BCD',  'Cadmium - Blood'),
    # UNM only # ('BHGE', 'Ethyl Mercury - Blood'),
    # UNM only # ('BHGM', 'Methyl Mercury - Blood'),
    # UNM only # ('BMN',  'Manganese - Blood'),
    # UNM only # ('BPB',  'Lead - Blood'),
    # UNM only # ('BSE',  'Selenium - Blood'),
    # UNM only # ('IHG',  'Inorganic Mercury - Blood'),
    # UNM only # ('THG',  'Mercury Total - Blood'),
    # UNM only # ('SCU',  'Copper - Serum'),
    # UNM only # ('SSE',  'Selenium - Serum'),
    # UNM only # ('SZN',  'Zinc - Serum'),
    ('UAS3', 'Arsenous (III) acid - Urine'),
    # UNM only # ('UAS5', 'Arsenic (V) acid - Urine'),
    ('UASB', 'Arsenobetaine - Urine'),
    # UNM only # ('UASC', 'Arsenocholine - Urine'),
    ('UBA',  'Barium - Urine'),
    ('UBE',  'Beryllium - Urine'),
    ('UCD',  'Cadmium - Urine'),
    ('UCO',  'Cobalt - Urine'),
    ('UCS',  'Cesium - Urine'),
    ('UDMA', 'Dimethylarsinic Acid - Urine'),
    ('UHG',  'Mercury - Urine'),
    # UNM only # ('UIO',  'Iodine - Urine'),
    ('UMMA', 'Monomethylarsinic Acid - Urine'),
    ('UMN',  'Manganese - Urine'),
    ('UMO',  'Molybdenum - Urine'),
    ('UPB',  'Lead - Urine'),
    # UNM only # ('UPT',  'Platinum - Urine'),
    ('USB',  'Antimony - Urine'),
    ('USN',  'Tin - Urine'),
    ('USR',  'Strontium - Urine'),
    ('UTAS', 'Arsenic Total - Urine'),
    ('UTL',  'Thallium - Urine'),
    # UNM only # ('UTMO', 'Trimethylarsine - Urine')
    ('UTU',  'Tungsten -  Urine'),
    ('UUR',  'Uranium - Urine'),
))]

CAT_HAR_MEMBER_C = (
    ('1', 'mother'),  # maps to maternal
    ('2', 'father'),
    ('3', 'child'),
)

CAT_HAR_TIME_PERIOD = (
    ('0', 'early enrollment'),  # maps to 12G
    ('1', 'enrollment'),        # maps to 24G
    ('3', 'week 63/delivery'),  # maps to 6WP
)

ADDITIONAL_FEATURES = [('Categorical', (
    ('Outcome', 'Outcome'),
    ('Member_c', 'Family Member'),
    ('TimePeriod', 'Collection Time'),
    ('CohortType', 'Cohort Type'),
))]

HAR_FEATURE_CHOICES = CAT_HAR_ANALYTES + ADDITIONAL_FEATURES
HAR_CATEGORICAL_CHOICES = ADDITIONAL_FEATURES
