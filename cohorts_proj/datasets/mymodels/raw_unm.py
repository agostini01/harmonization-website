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

CAT_UNM_ETHNICITY = [
    ('1', 'Puerto Rican'), 
    ('2', 'Cuban or Cuban-American') ,
    ('3', 'Dominican'),
    ('4', 'Mexican'), 
    ('5', 'Mexican-American' ),
    ('6', 'Central or South American' ),
    ('97', 'Other'),
    ('888', 'Refused'),
    ('999', 'Don"t Know')
]

CAT_UNM_RACE = [
    ('1', 'White'), 
    ('2', 'Black or African American'), 
    ('3', 'American Indian or Alaska Native'), 
    ('4', 'Asian'), 
    ('5', 'Native to Hawaii or Other Pacific Islands'),
    ('6', 'More than one race'),
    ('97', 'Some other race'),
    ('888', 'Refused'),
    ('999', 'Don"t know')
]

CAT_UNM_EDUCATION = [
    ('1', 'Less than 11th grade' ),
    ('2', 'High school graduate or equivalent' ),
    ('3', 'Junior college graduate') ,
    ('4', 'College graduate'),
    ('5', 'Any post-graduate schooling')]

CAT_UNM_INCOME = [
    ('1', '0 - 4999'),
    ('2', '5000 - 9999'),
    ('3', '10000, 19999'), 
    ('4', '20000 - 39999'),
    ('5', '40000 - 69999'),
    ('6', '70000+')
]

CAT_UNM_SMOKING = [
    ('0', 'never smoked'),
    ('1', 'past smoker'),
    ('2', 'current smoker'), 
    ('3', 'smoke during pregnancy')
]

CAT_UNM_COMPLICATIONS = [
    ('0', 'none'), 
    ('1', 'complications present')
]
CAT_UNM_FOLIC = [
    ('0', 'none'), 
    ('1', 'complications present')
]

CAT_UNM_FISH = [
    ('0', 'no'), 
    ('1', 'yes')
]

CAT_UNM_SEX = [
    ('1','Female'),
    ('2', 'Male'),
    ('3', 'Ambiguos Genitalia')
]

class RawUNM(models.Model):

    # PIN_Patient: unique identifier
    # TODO: Maybe update to UUIDField
    # TODO: Maybe create a ManyToManyField for UUID
    PIN_Patient = models.CharField(max_length=100)

    # Member_c – categorical variable: 1 = mother; 2 = father; 3 = child
    Member_c = models.CharField(max_length=1, choices=CAT_UNM_MEMBER_C)

    # TimePeriod – categorical: 1 = enrollment; 3 = week 36/delivery
    TimePeriod = models.CharField(max_length=1, choices=CAT_UNM_TIME_PERIOD)

    # Analyte – categorical:
    Analyte = models.CharField(max_length=4, choices=CAT_UNM_ANALYTES)

    # Result: value, corrected for detection levels, in ug/L
    Result = models.FloatField(blank = True, null=True,default = -9.0)

    # Creat_Corr_Result: value, corrected for creatinine (urine only), ug/L
    Creat_Corr_Result = models.CharField(max_length=22, blank=True, null=True,default = '-9')

    creatininemgdl = models.CharField(max_length=22, blank=True, null=True,default = '-9')
    
    # Outcome – categorical variable: 1 = preterm birth; 0 = term
    Outcome = models.CharField(max_length=1, choices=CAT_UNM_OUTCOME, blank=True)

    Outcome_weeks = models.FloatField(blank = True, null=True,default = -9.0)

    age = models.FloatField(blank=True, null = True, default = -9)	
    
    ethnicity = models.CharField(max_length=4, choices=CAT_UNM_ETHNICITY, blank=True, null=True,default = '-9')
    
    race = models.CharField(max_length=4, choices=CAT_UNM_RACE, blank=True, null=True,default = '-9')
    
    education = models.CharField(max_length=4, choices=CAT_UNM_EDUCATION, blank=True, null=True,default = '-9')
    
    BMI = models.FloatField(blank = True, null=True,default = -9.0)	
    
    income = models.CharField(max_length=4, choices=CAT_UNM_INCOME, blank=True, null=True,default = '-9')
    
    smoking = models.CharField(max_length=4, choices=CAT_UNM_SMOKING, blank=True, null=True,default = '-9')
    
    parity = models.IntegerField(max_length=4, blank=True, null=True,default = -9)
    
    preg_complications	= models.CharField(max_length=4, choices=CAT_UNM_COMPLICATIONS, blank=True, null=True,default = '-9')
    
    folic_acid_supp	= models.CharField(max_length=4, choices=CAT_UNM_FOLIC, blank=True, default = '-9')
    
    fish = models.FloatField(blank = True, null = True, default = -9.0)	
    
    babySex	= models.CharField(max_length=4, choices=CAT_UNM_SEX, blank=True, default = '-9')
    
    birthWt = models.FloatField(blank = True, null = True, default = -9.0)
    
    birthLen = models.FloatField(blank = True, null = True, default = -9.0)

    WeightZScore = models.FloatField(blank = True, null = True,  default = -9.0)
    
    WeightCentile = models.FloatField(blank = True,null = True, default = -9.0)	
    
    LGA	= models.FloatField(blank = True, null = True, default = -9.0)
    
    SGA= models.FloatField(blank = True,null = True, default = -9.0)

    headCirc = models.FloatField(blank = True,null = True, default = -9.0)

    gestAge_collection = models.FloatField(blank = True,null = True, default = -9.0)

    birth_year = models.IntegerField(blank=True, default = -9)
    
    birth_month = models.IntegerField(blank=True, default = -9)



