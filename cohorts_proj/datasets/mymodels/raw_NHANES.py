from django.db import models

CAT_NHANES_ANALYTES = (
    ("UTAS", "Urinary Total Arsenic"),
    ("UALB_mg", "Urinary Albumin in mg/L"),
    ("UALB_ug ", "Urinary Albumin in ug/L"),
    ("UCRT_mg", "Urinary Creatinine in mg/dL"),
    ("UCRT_umol", "Urinary Creatinine in umol/L"),
    ("UCR", "Urinary Chromium in ug/L"),
    ("I", "Urinary Iodine in ug/L"),
    ("UHG", "Urinary Mercury in ug/L"),
    ("UBA", "Urinary Barium in ug/L"),
    ("UCD", "Urinary Cadmium in ug/L"),
    ("UCO", "Urinary Cobalt in ug/L"),
    ("UCS", "Urianry Cesium in ug/L"),
    ("UMO","Urinary Molybdenum in ug/L"),
    ("UMN","Urinary Manganese in ug/L"),
    ("UPB","Urinary Lead in ug/L"),
    ("USB", "Urianry Antimony in ug/L"),
    ("USN","Urinary Tin in ug/L"),
    ("UTL", "Urinary Thallium in ug/L"),
    ("UTU", "Urinary Tungsten in ug/L"),
    ("UNI", "Urinary Nickel in ug/L"))

   
CAT_NHANES_PREGNANT = (
    (1, 'pregnant'),
    (2, 'not pregnant'),
    (3, 'could not be determined'),
)

CAT_NHANES_MARITAL = (
    (1, 'married'),
    (2, 'widowed'),
    (3, 'divorced'),
    (4, 'seperated'),
    (5, 'never married'),
    (6, 'living with partner'),
    (7, 'refused'),
    (99, 'do not know')
)

##Number of kids in the household 5 years old or less
CAT_NHANES_CHILD_A = (
    (1, '1 kid'),
    (2, '2 kids'),
    (3, '3 or more kids')
)

##Number of kids in the household 6-17 years old 
CAT_NHANES_CHILD_B = (
    (1, '1 kid'),
    (2, '2 kids'),
    (3, '3 or more kids')
)


##Annual Household Income
CAT_NHANES_H_INC = (
    (1, '$0 to $4,999'),
    (2, '$5,000 to $9,999'),
    (3, '$10,000 to $14,999'),
    (4, '$15,000 to $19,999'),
    (5, '$20,000 to $24,999'),
    (6, '$25,000 to $34,999'),
    (7, '$35,000 to $44,999 '),
    (8, '$45,000 to $54,999'),
    (9, '	$55,000 to $64,999 '),
    (10, '$$65,000 to $74,999'),

    (12, '$20,000 and Over'),
    (13, 'Under $20,000'),
    (14, '$75,000 to $99,999'),
    (15, '$100,000 and Over '),
    (77, 'refused'))

##Annual Family Income
CAT_NHANES_F_INC = (
    (1, '$0 to $4,999'),
    (2, '$5,000 to $9,999'),
    (3, '$10,000 to $14,999'),
    (4, '$15,000 to $19,999'),
    (5, '$20,000 to $24,999'),
    (6, '$25,000 to $34,999'),
    (7, '$35,000 to $44,999 '),
    (8, '$45,000 to $54,999'),
    (9, '	$55,000 to $64,999 '),
    (10, '$$65,000 to $74,999'),

    (12, '$20,000 and Over'),
    (13, 'Under $20,000'),
    (14, '$75,000 to $99,999'),
    (15, '$100,000 and Over '),
    (77, 'refused'))



##Education level for Adults 20 years old and Over 
CAT_NHANES_EDU = (
    (1, 'Less than 9th grade'),
    (2, '9-11th grade (Includes 12th grade with no diploma)'),
    (3, 'High school graduate/GED or equivalent'),
    (4, 'Some college or AA degree'),
    (5, 'College graduate or above'),

    (7, 'refused'),
    (9, 'do not know'))


##Race/Hispanic origin w/ NH (Non-Hispanic) Asian
CAT_NHANES_RAC = (
    (1, 'Mexican American'),
    (2, 'Other Hispanic'),
    (3, 'Non-Hispanic White'),
    (4, 'Non-Hispanic Black'),
    (6, 'Non-Hispanic Asian'),
    (7, 'Other Race - Including Multi-Racial'))

class RawNHANES_BIO(models.Model):

   
    Participant = models.IntegerField(max_length=100)
    Age = models.IntegerField(max_length=6)
    Time_Period = models.CharField(max_length=30)
    ##1 means pregnant, 2 means not pregant, 3 means not determined
    Pregnant = models.IntegerField(max_length=1, choices=CAT_NHANES_PREGNANT, null=True)
    Marital = models.IntegerField(max_length=3, choices=CAT_NHANES_MARITAL, null=True)
    Child_A = models.IntegerField(max_length=3, choices=CAT_NHANES_CHILD_A, null=True)
    Child_B = models.IntegerField(max_length=3, choices=CAT_NHANES_CHILD_B, null=True)
    H_Inc = models.IntegerField(max_length=3, choices=CAT_NHANES_H_INC, null=True)
    F_Inc = models.IntegerField(max_length=3, choices=CAT_NHANES_F_INC, null=True)
    Edu = models.IntegerField(max_length=3, choices=CAT_NHANES_EDU, null=True)
    Rac = models.IntegerField(max_length=3, choices=CAT_NHANES_RAC, null=True)
    Blod = models.FloatField(max_length=20, null=True)
    Analyte_Value = models.FloatField(max_length=15, null=True) 
    Analyte = models.CharField(max_length=10, choices=CAT_NHANES_ANALYTES, null=True) 
    


NHANES_DD_COL_NAME = (
    ("UTAS", "Urinary Total Arsenic"),
    ("Alb", "Urinary Albumin in mg/L"),
    ("Cr_mg", "Urinary Creatinine in mg/dL"),
    ("Cr_umol", "Urinary Creatinine in umol/L"),
    ("Chrom", "Urinary Chromium in ug/L"),
    ("I", "Urinary Iodine in ug/L"),
    ("Hg", "Urinary Mercury in ug/L"),
    ("Ba", "Urinary Barium in ug/L"),
    ("Cd", "Urinary Cadmium in ug/L"),
    ("Co", "Urinary Cobalt in ug/L"),
    ("Cs", "Urianry Cesium in ug/L"),
    ("Mo","Urinary Molybdenum in ug/L"),
    ("Mn","Urinary Manganese in ug/L"),
    ("Pb","Urinary Lead in ug/L"),
    ("Sb", "Urianry Antimony in ug/L"),
    ("Sn","Urinary Tin in ug/L"),
    ("TI", "Urinary Thallium in ug/L"),
    ("W", "Urinary Tungsten in ug/L"),
    ("Ni", "Urinary Nickel in ug/L"),
    ('Participant', 'Particiapnt ID Number'),
    ('Time_Period', 'Time period that data was taken from'),
    ('UTAS_wt', 'UTAS Sample Weight'),
    ('UTAS_Blod', 'UTAS Below Limit of Detection Value '),
    ('Alb_Blod', 'Albumin Below Limit of Detection Value'),
    ('Cr_Blod', 'Creatinine Below Limit of Detection Value for Creatinine'),
    ('Alb_to_Cr', ),
    ('Chrom_wt', 'Chromium Sample Weight'),
    ('Chrom_Blod', 'Chromium Below Limit of Detection Value'),
    ('I_Blod', 'Iodine  Below Limit of Detection Value'),
    ('Hg_Blod', 'Mercury Below Limit of Detection Value'),
    ('Ba_Blod', 'Barium Below Limit of Detection Value'),
    ('Cd_Blod', 'Cadmium Below Limit of Detection Value'),
    ('Co_Blod',  'Cobalt Below Limit of Detection Value'),
    ('Cs_Blod', 'Cesium Below Limit of Detection Value'),
    ('Mo_Blod',  'Molybdenum Below Limit of Detection Value'),
    ('Mn_Blod', 'Manganese Below Limit of Detection Value'),
    ('Pb_Blod',  'Lead Below Limit of Detection Value'),
    ('Sb_Blod',  'Antimony Below Limit of Detection Value'),
    ('Sn_Blod',  'Tin Below Limit of Detection Value'),
    ('TI_Blod', 'Thallium'),
    ('W_Blod', 'Tungsten Below Limit of Detection Value'),
    ('Ni_Blod', 'Nickel Below Limit of Detection Value'),
    ('Pregnant', 'Pregnancy Status'),
    ('Age', 'Age in Years'),
    ('Marital', 'Marital Status'),
    ('Yng_Kids', 'Number of kids in the household 5 years old or less'),
    ('Old_Kids', 'Number of kids in the household 6-17 years old'),
    ('H_Inc', 'Annual Household Income'),
    ('F_Inc', 'Annual Family Income'),
    ('Edu', 'Education Level for Adults 20 Years Old and Over'),
    ('Rac', 'Race/Hispanic origin w/ NH (Non-Hispanic) Asian')
    )

NHANES_DD_TYPE= (
    ('int64', 'integer 64 bits'),
    ('float64', 'float 64 bits'),
    ('str', 'string'))


class RawNHANES_DD(models.Model):

    cohort = models.CharField(max_length=1000)
    var_name = models.CharField(max_length=1000)
    ##Whatis from name
    form_name = models.CharField(max_length=1000, null = True, blank=True)
    ##what is section name
    section_name = models.CharField(max_length=1000, null = True, blank=True)
    ##what
    field_type = models.CharField(max_length=1000, null = True, blank=True)
    field_label = models.CharField(max_length =1000, null = True, blank=True)
    field_choices = models.CharField(max_length= 1000, null = True, blank=True)
    field_min = models.CharField(max_length= 1000, null = True, blank=True)
    field_max = models.CharField(max_length= 1000, null = True, blank=True)

#    Col = models.CharField(max_length=40, choices=NHANES_DD_COL, null=True)
#    Type = models.CharField(max_length=30, choices=NHANES_DD_TYPE, null=True)
#    Target = models.CharField(max_length=500, null=True)



NHANES_LLOD_ANALYTE = (
    ('UTAS', 'Urinary Total Arsenic'),
    ('UALB', 'Albumin'), 
    ('UCRT', 'Creatinine'),
    ('UCR ', 'Chromium'),
    ('UI',  'Iodine'),
    ('UHG', 'Mercury'),
    ('UBA', 'Barium'),
    ('UCD', 'Cadmium'),
    ('UCS', 'Cesium'),
    ('UCO', 'Cobalt'),
    ('UMN', 'Manganese'),
    ('UMO', 'Molbdenum'),
    ('UPB', 'Lead'),
    ('USB', 'Antimony'),
    ('UTL', 'Thallium'),
    ('USN', 'Tin'),
    ('UTU', 'Tungsten'),
    ('UNI', 'Nickel')
)

NHANES_LLOD_UNITS = (
    ('ug/L', 'micrograms per Liter'),
    ('mg/dL','milligrams per deciliter')
)
class RawNHANES_LLOD(models.Model):

    Analyte = models.CharField(max_length=100, choices=NHANES_LLOD_ANALYTE, null=True)
    Value = models.FloatField(max_length=6, null=True)
    Units = models.CharField(max_length=30, choices=NHANES_LLOD_UNITS, null=True)
    Time_Period = models.CharField(max_length=30, null=True)
