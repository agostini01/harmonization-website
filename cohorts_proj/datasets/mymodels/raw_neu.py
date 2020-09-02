from django.db import models

CAT_NEU_ANALYTES = (
    # Urine
    ('USB', 'Antimony - Urine'),
    ('UTAS', 'Arsenic - Urine'),
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
    ('BSB', 'Antimony - Blood'),
    ('BTAS', 'Arsenic - Blood'),
    ('BAL', 'Aluminum - Blood'),
    ('BBE', 'Beryllium - Blood'),
    ('BBA', 'Barium - Blood'),
    ('BCD', 'Cadmium - Blood'),
    ('BCS', 'Cesium - Blood'),
    ('BCO', 'Cobalt - Blood'),
    ('BCU', 'Copper - Blood'),
    ('BCR', 'Chromium - Blood'),
    ('BFE', 'Iron - Blood'),
    ('BPB', 'Lead - Blood'),
    ('BPB2', 'Lead (208) - Blood'),
    ('BMB', 'Manganese - Blood'),
    ('BHG', 'Mercury - Blood'),
    ('BMO', 'Molybdenum - Blood'),
    ('BNI', 'Nickel - Blood'),
    ('BPT', 'Platinum - Blood'),
    ('BTL', 'Thallium - Blood'),
    ('BTU', 'Tungsten - Blood'),
    ('BUR', 'Uranium - Blood'),
    ('BVA', 'Vanadium - Blood'),
    ('BSE', 'Selenium - Blood'),
    ('BSEG', 'Selenium+G1124 - Blood'),
    ('BSN', 'Tin - Blood'),
    ('BZN', 'Zinc - Blood'),
    #     'triclosan - Urine',
    #     'Butyl Paraben - Urine',
    #     'bisphenol F - Urine',
    #     'bisphenol S - Urine',
    #     '2 4-dichlorophenol - Urine',
    #     'bisphenol A - Urine',
    #     'Ethyl Paraben - Urine',
    #     '2 5-dichlorophenol - Urine',
    #     'Propyl Paraben - Urine',
    #     'triclocarban - Urine',
    #     'Methyl Paraben - Urine',
    #     'benzophenone-3 - Urine',
    #     'Cyclohexane-1 2-dicarboxylic acid monohydroxy isononyl ester - Urine',
    #     'cyclohexane-1 2-dicarboxylic acid monocarboxyisooctyl ester - Urine',
    #     'Mono-isononyl phthalate - Urine',
    #     'Mono carboxyisononyl phthalate - Urine',
    #     'Mono-3-carboxypropyl phthalate - Urine',
    #     'mono-2-ethyl-5-hydrohexyl terephthalate - Urine',
    #     'Monooxononyl phthalate - Urine',
    #     'Mono-hydroxybutyl phthalate - Urine',
    #     'Monobenzyl phthalate - Urine',
    #     'Mono-2-ethylhexyl phthalate - Urine',
    #     'Mono-hydroxyisobutyl phthalate - Urine',
    #     'Mono carboxyisooctyl phthalate - Urine',
    #     'mono-2-ethyl-5-carboxypentyl terephthalate - Urine',
    #     'Mono-2-ethyl-5-oxohexyl phthalate - Urine',
    #     'Mono-2-ethyl-5-hydroxyhexyl phthalate - Urine',
    #     'Mono-2-ethyl-5-carboxypentyl phthalate - Urine',
    #     'Mono-isobutyl phthalate - Urine',
    #     'Mono-n-butyl phthalate - Urine',
    #     'Monoethyl phthalate - Urine',
    #     'Mono carboxy isononyl phthalate - Urine',
    #     'Mono carboxy isooctyl phthalate - Urine',
)

CAT_NEU_MEMBER_C = (
    ('1', 'mother'),
    ('2', 'father'),
    ('3', 'child'),
)

CAT_NEU_TIME_PERIOD = (
    ('1', '16-20 weeks'),
    ('2', '22-26 weeks'),
    ('3', '24-28 weeks')
)

# TODO Invert the outcome results
# NEU provides "Is Preterm?" instead of "Is Term?"
# To conform to other datasets, we should invert the 0/1 order
CAT_NEU_OUTCOME = (
    ('0', 'term'),
    ('1', 'preterm'),
)


class RawNEU(models.Model):

    # PIN_Patient: unique identifier
    # TODO: Maybe update to UUIDField
    # TODO: Maybe create a ManyToManyField for UUID
    PIN_Patient = models.CharField(max_length=100)

    # Member_c – categorical variable: 1 = mother; 2 = father; 3 = child
    Member_c = models.CharField(max_length=1, choices=CAT_NEU_MEMBER_C)

    # TimePeriod – categorical: 1, 2 , 3
    TimePeriod = models.CharField(max_length=1, choices=CAT_NEU_TIME_PERIOD)

    # Analyte – categorical:
    Analyte = models.CharField(max_length=4, choices=CAT_NEU_ANALYTES)

    # Result: value, corrected for detection levels, in ug/L
    Result = models.FloatField()

    # LOD : value, Limit of Detection
    LOD = models.FloatField()

    # Creat_Corr_Result: value, corrected for creatinine (urine only), ug/L
    # Creat_Corr_Result = models.FloatField()

    # Outcome – categorical variable: 1 = preterm birth; 0 = term
    Outcome = models.CharField(
        max_length=1, choices=CAT_NEU_OUTCOME, blank=True)
