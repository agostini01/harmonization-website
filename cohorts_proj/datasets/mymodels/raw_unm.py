from django.db import models

CAT_UNM_ANALYTES = (
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
)

CAT_UNM_MEMBER_C = (
    ('1', 'mother'),
    ('2', 'father'),
    ('3', 'child'),
)

CAT_UNM_TIME_PERIOD = (
    ('1', 'enrollment'),
    ('3', 'week 36/delivery'),
)

# TODO Invert the outcome results
# UNM provides "Is Preterm?" instead of "Is Term?"
# To conform to other datasets, we should invert the 0/1 order
CAT_UNM_OUTCOME = (
    ('0', 'term'),
    ('1', 'preterm'),
)


class RawUNM(models.Model):

    # PIN_Patient: unique identifier
    # TODO: Maybe update to UUIDField
    # TODO: Maybe create a ManyToManyField for UUID
    PIN_Patient = models.CharField(max_length=100)

    # Member_c – categorical variable: 1 = mother; 2 = father; 3 = child
    Member_c = models.CharField(max_length=1, choices=CAT_UNM_MEMBER_C)

    # TimePeriod – categorical: 1 = enrollment; 3 = week 63/delivery
    TimePeriod = models.CharField(max_length=1, choices=CAT_UNM_TIME_PERIOD)

    # Analyte – categorical:
    Analyte = models.CharField(max_length=4, choices=CAT_UNM_ANALYTES)

    # Result: value, corrected for detection levels, in ug/L
    Result = models.FloatField()

    # Creat_Corr_Result: value, corrected for creatinine (urine only), ug/L
    Creat_Corr_Result = models.FloatField()
    
    # Outcome – categorical variable: 1 = preterm birth; 0 = term
    Outcome = models.CharField(max_length=1, choices=CAT_UNM_OUTCOME, blank=True)
