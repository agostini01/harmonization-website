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

CAT_NEU_ETHNICITY = [
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

CAT_NEU_RACE = [
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

#CAT_UNM_EDUCATION = [
#    ('1', 'Less than 11th grade' ),
#    ('2', 'High school graduate or equivalent' ),
 #   ('3', 'Junior college graduate') ,
 #   ('4', 'College graduate'),
 #   ('5', 'Any post-graduate schooling')]
#
#CAT_UNM_INCOME = [
#    ('1', '0 - 4999'),
#    ('2', '5000 - 9999'),
#    ('3', '10000, 19999'), 
#    ('4', '20000 - 39999'),
#   ('5', '40000 - 69999'),
#   ('6', '70000+')
#]

CAT_NEU_SMOKING = [
    ('0', 'never smoked'),
    ('1', 'past smoker'),
    ('2', 'current smoker'), 
    ('3', 'smoke during pregnancy')
]

CAT_NEU_COMPLICATIONS = [
    ('0', 'none'), 
    ('1', 'complications present')
]
CAT_NEU_FOLIC = [
    ('0', 'no'), 
    ('1', 'yes')
]

CAT_NEU_SEX = [
    ('1','Female'),
    ('2', 'Male'),
    ('3', 'Ambiguos Genitalia')
]

class RawNEU(models.Model):

    # PIN_Patient: unique identifier
    # TODO: Maybe update to UUIDField
    # TODO: Maybe create a ManyToManyField for UUID
    PIN_Patient = models.CharField(max_length=500)

    # Member_c – categorical variable: 1 = mother; 2 = father; 3 = child
    Member_c = models.CharField(max_length=1, choices=CAT_NEU_MEMBER_C)

    # TimePeriod – categorical: 1, 2 , 3
    TimePeriod = models.CharField(max_length=1, choices=CAT_NEU_TIME_PERIOD)

    # Analyte – categorical:
    Analyte = models.CharField(max_length=50, choices=CAT_NEU_ANALYTES)

    # Result: value, corrected for detection levels, in ug/L
    Result = models.FloatField()

    # LOD : value, Limit of Detection
    LOD = models.FloatField()

    # Creat_Corr_Result: value, corrected for creatinine (urine only), ug/L
    # Creat_Corr_Result = models.FloatField()

    # Outcome – categorical variable: 1 = preterm birth; 0 = term
    Outcome = models.CharField(
        max_length=1, choices=CAT_NEU_OUTCOME, blank=True)

    Outcome_weeks = models.FloatField(null=True, blank = True, default = -9.0)

    age = models.IntegerField(null=True,blank=True, default = -9)	
    
    ethnicity = models.CharField(max_length=50, choices=CAT_NEU_ETHNICITY, blank=True, null=True,default = '-9')
    
    race = models.CharField(max_length=50, choices=CAT_NEU_RACE, blank=True, null=True,default = '-9')
    
    ed = models.CharField(max_length=50,  blank=True, null=True,default = '-9')
    
    BMI = models.FloatField(blank = True,null=True, default = -9.0)	
    
    fvinc = models.CharField(max_length=50, blank=True, null=True,default = '-9')
    
    smoking = models.CharField(max_length=50, choices=CAT_NEU_SMOKING, blank=True,null=True, default = '-9')
    
    pregnum = models.IntegerField(max_length=50, blank=True, null=True,default = -9)
    
    preg_complications	= models.CharField(max_length=50, choices=CAT_NEU_COMPLICATIONS, blank=True, null=True,default = '-9')
    
    folic_acid_supp	= models.CharField(max_length=50, choices=CAT_NEU_FOLIC, blank=True, null=True,default = '-9')
    
    fish = models.FloatField(blank = True, null=True,default = -9.0)
    
    babySex	= models.CharField(max_length=50, choices=CAT_NEU_SEX, blank=True, null=True,default = '-9')
    
    birthWt = models.FloatField(blank = True, null=True,default = -9.0)
    
    birthLen = models.FloatField(blank = True, null=True,default = -9.0)

    headCirc = models.FloatField(blank = True, null=True,default = -9.0)

    fvdate = models.CharField(max_length=50, blank=True, null=True,default = '-9')

    svdate = models.CharField(max_length=50, blank=True, null=True,default = '-9')

    tvdate = models.CharField(max_length=50, blank=True,null=True, default = '-9')

    SPECIFICGRAVITY_V1 = models.FloatField(blank = True, null=True,default = -9.0)

    SPECIFICGRAVITY_V2 = models.FloatField(blank = True, null=True,default = -9.0)

    SPECIFICGRAVITY_V3 = models.FloatField(blank = True, null=True,default = -9.0)

    WeightZScore = models.FloatField(blank = True, null=True,default = -9.0)
    
    WeightCentile = models.FloatField(blank = True, null=True,default = -9.0)	
    
    LGA	= models.FloatField(blank = True, null=True,default = -9.0)
    
    SGA= models.FloatField(blank = True, null=True,default = -9.0)

    PPDATEDEL = models.CharField(blank = True, null=True,max_length = 50, default = '-9.0')

    ga_collection = models.FloatField(blank = True, null=True,default = -9.0)	

    fish_pu_v1 = models.FloatField(blank = True, null=True,default = -9.0)
    fish_pu_v2 = models.FloatField(blank = True, null=True,default = -9.0)	
    fish_pu_v3 = models.FloatField(blank = True,null=True, default = -9.0)	
 
    


