CAT_DAR_ANALYTES = (
    # Analyate acronym and name,                    Mapping in the dar DB
    ('UAG', ' Silver - Urine'),                     # Ag
    ('UAL', ' Aluminium - Urine'),                  # Al
    ('UCR',  'Chromium - Urine'),                   # Cr
    ('UCU',  'Copper - Urine'),                     # Cu
    ('UFE',  'Iron - Urine'),                       # Fe
    ('UNI',  'Niquel - Urine'),                     # Ni
    ('UVA',  'Vanadium - Urine'),                   # V
    ('UZN',  'Zinc - Urine'),                       # Zn
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
    ('UAS3', 'Arsenous (III) acid - Urine'),        # As
    # ('UAS5', 'Arsenic (V) acid - Urine'),
    ('UASB', 'Arsenobetaine - Urine'),              # AsB
    # ('UASC', 'Arsenocholine - Urine'),
    ('UBA',  'Barium - Urine'),                     # Ba
    ('UBE',  'Beryllium - Urine'),                  # Be
    ('UCD',  'Cadmium - Urine'),                    # Cd
    ('UCO',  'Cobalt - Urine'),                     # Co
    ('UCS',  'Cesium - Urine'),                     # Cs
    ('UDMA', 'Dimethylarsinic Acid - Urine'),       # DMA
    ('UHG',  'Mercury - Urine'),                    # Hg
    # ('UIO',  'Iodine - Urine'),
    ('UMMA', 'Monomethylarsinic Acid - Urine'),     # MMA
    ('UMN',  'Manganese - Urine'),                  # Mn
    ('UMO',  'Molybdenum - Urine'),                 # Mo
    ('UPB',  'Lead - Urine'),                       # PB
    # ('UPT',  'Platinum - Urine'),
    ('USB',  'Antimony - Urine'),                   # Sb
    ('USN',  'Tin - Urine'),                        # Sn
    ('USR',  'Strontium - Urine'),                  # Sr
    ('UTAS', 'Arsenic Total - Urine'),              # iAs
    ('UTL',  'Thallium - Urine'),                   # Tl
    # ('UTMO', 'Trimethylarsine - Urine')
    ('UTU',  'Tungsten -  Urine'),                  # W
    ('UUR',  'Uranium - Urine'),                    # U
)

CAT_DAR_MEMBER_C = (
    ('1', 'mother'), # maps to maternal
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
    ('0', 'early enrollment'),  # maps to 12G
    ('1', 'enrollment'),        # maps to 24G
    ('3', 'week 63/delivery'),  # maps to 6WP
)

ADDITIONAL_FEATURES = (
    ('Outcome', 'Outcome'),
    ('Member_c', 'Family Member'),
    ('TimePeriod', 'Collection Time'),
    ('CohortType', 'Cohort Type'),
)

DAR_FEATURE_CHOICES = CAT_DAR_ANALYTES + ADDITIONAL_FEATURES
DAR_CATEGORICAL_CHOICES = ADDITIONAL_FEATURES
