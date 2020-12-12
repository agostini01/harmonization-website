CAT_NEU_ANALYTES = [('Analytes', (
    ('USB', 'Antimony - Urine'),
    ('UTAS', 'Arsenic Total - Urine'), #modified just for poster - change back later/check if it's actually total 
    ('UBA', 'Barium - Urine'),
    ('UBE', 'Beryllium - Urine'),
    ('UCD', 'Cadmium - Urine'),
    ('UCS', 'Cesium - Urine'),
    ('UCR', 'Chromium - Urine'),
    ('UCO', 'Cobalt - Urine'),
    ('UCU', 'Copper - Urine'),
    ('UPB', 'Lead - Urine'),
    ('UMN', 'Manganese - Urine'),
    ('UHG', 'Mercury - Urine'),
    ('UMO', 'Molybdenum - Urine'),
    ('UNI', 'Nickel - Urine'),
    ('UPT', 'Platinum - Urine'),
    ('USE', 'Selenium - Urine'),
    ('UTL', 'Thallium - Urine'),
    ('USN', 'Tin - Urine'),
    ('UTU', 'Tungsten - Urine'),
    ('UUR', 'Uranium - Urine'),
    ('UVA', 'Vanadium - Urine'),
    ('UZN', 'Zinc - Urine'),
    # Blood
    # ('BSB', 'Antimony - Blood'   ),
    # ('BTAS','Arsenic - Blood'    ),
    # ('BAL', 'Aluminum - Blood'   ),
    # ('BBE', 'Beryllium - Blood'  ),
    # ('BBA', 'Barium - Blood'     ),
    # ('BCD', 'Cadmium - Blood'    ),
    # ('BCS', 'Cesium - Blood'     ),
    # ('BCO', 'Cobalt - Blood'     ),
    # ('BCU', 'Copper - Blood'     ),
    # ('BCR', 'Chromium - Blood'   ),
    # ('BFE', 'Iron - Blood'       ),
    # ('BPB', 'Lead - Blood'       ),
    # ('BPB208','Lead (208) - Blood'),
    # ('BMB', 'Manganese - Blood'  ),
    # ('BHG', 'Mercury - Blood'    ),
    # ('BMO', 'Molybdenum - Blood' ),
    # ('BNI', 'Nickel - Blood'     ),
    # ('BPT', 'Platinum - Blood'   ),
    # ('BTL', 'Thallium - Blood'   ),
    # ('BTU', 'Tungsten - Blood'   ),
    # ('BUR', 'Uranium - Blood'    ),
    # ('BVA', 'Vanadium - Blood'   ),
    # ('BSE', 'Selenium - Blood'),
    # ('BSEG1124', 'Selenium+G1124 - Blood'),
    # ('BSN', 'Tin - Blood'        ),
    # ('BZN', 'Zinc - Blood'       ),
))]

CAT_NEU_MEMBER_C = (
    ('1', 'mother'),
    ('2', 'father'),
    ('3', 'child'),
)

CAT_NEU_TIME_PERIOD = (
    ('9', 'any'),               # all time periods ploted together
    ('1', '16-20 weeks'),
    ('2', '22-26 weeks'),
    ('3', '24-28 weeks')
)

ADDITIONAL_FEATURES = [('Categorical', (
    ('Outcome', 'Outcome'),
    ('Member_c', 'Family Member'),
    ('TimePeriod', 'Collection Time'),
    ('CohortType', 'Cohort Type'),
))]

NEU_FEATURE_CHOICES = CAT_NEU_ANALYTES + ADDITIONAL_FEATURES
NEU_CATEGORICAL_CHOICES = ADDITIONAL_FEATURES
