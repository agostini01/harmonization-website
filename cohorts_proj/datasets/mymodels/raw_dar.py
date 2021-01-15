from django.db import models
import numpy as np

# Each row of raw Dartmouth datset has the following fields
# unq_id
# assay
# lab
# participant_type
# time_period
# batch
# squid
# sample_gestage_days
# iAs, iAs_IDL, iAs_BDL
# AsB, AsB_IDL, AsB_BDL
# DMA, DMA_IDL, DMA_BDL
# MMA, MMA_IDL, MMA_BDL
# Ba, Ba_IDL, Ba_BDL
# Cs, Cs_IDL, Cs_BDL
# Sr, Sr_IDL, Sr_BDL
# W,  W_IDL,  W_BDL
# Ag, Ag_IDL, Ag_BDL
# Al, Al_IDL, Al_BDL
# As, As_IDL, As_BDL
# Be, Be_IDL, Be_BDL
# Cd, Cd_IDL, Cd_BDL
# Co, Co_IDL, Co_BDL
# Cr, Cr_IDL, Cr_BDL
# Cu, Cu_IDL, Cu_BDL
# Fe, Fe_IDL, Fe_BDL
# Hg, Hg_IDL, Hg_BDL
# Mn, Mn_IDL, Mn_BDL
# Mo, Mo_IDL, Mo_BDL
# Ni, Ni_IDL, Ni_BDL
# Pb, Pb_IDL, Pb_BDL
# Sb, Sb_IDL, Sb_BDL
# Se, Se_IDL, Se_BDL
# Sn, Sn_IDL, Sn_BDL
# Tl, Tl_IDL, Tl_BDL
# U,  U_IDL,  U_BDL
# W,  W_IDL,  W_BDL
# Zn, Zn_IDL, Zn_BDL
# V,  V_IDL,  V_BDL


CAT_DAR_MEMBER_C = (
    ('maternal', 'mother'),
    ('child', 'child'),
)

# From email
# 6 weeks postpartum, 6 months, 12 months, 24 months, 3 years, 5 years
# TODO verify accronyms on the left hand side
CAT_DAR_TIME_PERIOD = (
    ('12G', 'week 12 gestational'),
    ('24G', 'week 24 gestational'),
    ('6WP', 'week 6 portpartun'),
    ('6MP', 'month 6 postpartum'),
    ('1YP', 'year 1 postpartum'),
    ('2YP', 'year 2 postpartum'),
    ('3YP', 'year 3 postpartum'),
    ('5YP', 'year 5 postpartum'),
)

CAT_DAR_BDL = (
    ('1', 'below detection level'),
    ('0', 'above detection level'),
    ('nan', 'invalid'),
)

CAT_DAR_OUTCOME = (
    ('0', 'term'),
    ('1', 'preterm'),
    ('nan', 'invalid'),
)


CAT_DAR_ETHNICITY = [
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

CAT_DAR_RACE = [
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

CAT_DAR_EDUCATION = [
    ('1', 'Less than 11th grade' ),
    ('2', 'High school graduate or equivalent' ),
    ('3', 'Junior college graduate') ,
    ('4', 'College graduate'),
    ('5', 'Any post-graduate schooling')]

CAT_DAR_INCOME = [
    ('1', '0 - 4999'),
    ('2', '5000 - 9999'),
    ('3', '10000, 19999'), 
    ('4', '20000 - 39999'),
    ('5', '40000 - 69999'),
    ('6', '70000+')
]

CAT_DAR_SMOKING = [
    ('0', 'never smoked'),
    ('1', 'past smoker'),
    ('2', 'current smoker'), 
    ('3', 'smoke during pregnancy')
]

CAT_DAR_COMPLICATIONS = [
    ('0', 'none'), 
    ('1', 'complications present')
]
CAT_DAR_FOLIC = [
    ('0', 'none'), 
    ('1', 'complications present')
]

CAT_DAR_SEX = [
    ('1','Female'),
    ('2', 'Male'),
    ('3', 'Ambiguos Genitalia')
]

class RawDAR(models.Model):

    # PIN_Patient: unique identifier
    # TODO: Maybe update to UUIDField
    # TODO: Maybe create a ManyToManyField for UUID
    unq_id = models.CharField(max_length=1000)

    assay = models.CharField(max_length=1000)
    # lab = models.CharField(max_length=1000)

    # participant_type – categorical variable: maternal = mother; child = child
    participant_type = models.CharField(
        max_length=15, choices=CAT_DAR_MEMBER_C)

    # time_period – categorical: 12G, 24G, 6WP, 6MP, 1YP, 2YP, 3YP, 5YP
    time_period = models.CharField(max_length=100, choices=CAT_DAR_TIME_PERIOD)

    # squid - unique identifier: Sample identifier
    # squid = models.CharField(max_length=1000)

    # sample_gestage_days - numeric: days of gestation
    sample_gestage_days = models.FloatField(blank = True, default = -9.0)

    # Outcome – categorical variable: 1 = preterm birth; 0 = term
    preterm = models.CharField(max_length=100, choices=CAT_DAR_OUTCOME, blank=True)

    #Outcome_weeks = models.FloatField(blank = True, default = -9.0)

    # List of analytes, Index of detection level, Above/Below IDL
    # Floating values unit: 1ppb = 1ug/L

    age = models.IntegerField(blank=True, default = -9)	
    
    ethnicity = models.CharField(max_length=100, choices=CAT_DAR_ETHNICITY, blank=True, default = '-9')
    
    race = models.CharField(max_length=100, choices=CAT_DAR_RACE, blank=True, default = '-9')
    
    education = models.CharField(max_length=100,  blank=True, default = '-9')
    
    BMI = models.FloatField(blank = True, default = -9.0)	
    
    smoking = models.CharField(max_length=100, choices=CAT_DAR_SMOKING, blank=True, default = '-9')
    
    parity = models.IntegerField(blank=True, default = -9)
    
    preg_complications	= models.CharField(max_length=100, choices=CAT_DAR_COMPLICATIONS, blank=True, default = '-9')
    
    folic_acid_supp	= models.CharField(max_length=100, choices=CAT_DAR_FOLIC, blank=True, default = '-9')
    
    babySex	= models.CharField(max_length=100, choices=CAT_DAR_SEX, blank=True, default = '-9')
    
    birthWt = models.FloatField(blank = True, default = -9.0)
    
    birthLen = models.FloatField(blank = True, default = -9.0)

    headCirc = models.FloatField(blank = True, default = -9.0)

    ponderal = models.FloatField(blank = True, default = -9.0)

    PNFFQTUNA = models.CharField(max_length=100, blank=True, default = '-9')

    PNFFQFR_FISH_KIDS = models.CharField(max_length=100, blank=True, default = '-9')

    PNFFQSHRIMP_CKD = models.CharField(max_length=100, blank=True, default = '-9')

    PNFFQDK_FISH = models.CharField(max_length=100, blank=True, default = '-9')

    PNFFQOTH_FISH = models.CharField(max_length=100, blank=True, default = '-9')

    mfsp_6 = models.CharField(max_length=100, blank=True, default = '-9')

    fish = models.FloatField(blank = True, default = -9.0)
    
    TOTALFISH_SERV = models.FloatField(blank = True, default = -9.0)

    folic_acid = models.FloatField(blank = True, default = -9.0)

    income5y = models.CharField(max_length=100, blank=True, default = '-9')

    urine_batchno_bulk = models.FloatField(blank = True, default = -9.0)

    urine_batchno_spec = models.FloatField(blank = True, default = -9.0)

    collect_age_days = models.FloatField(blank = True, default = -9.0)

    collect_age_src = models.CharField(max_length=100, blank=True, default = '-9')

    collection_season = models.CharField(max_length=100, blank=True, default = '-9')

    pH = models.FloatField(blank = True, default = -9.0)

    TotAs_noAsB = models.FloatField(blank = True, default = -9.0)

    PropMMAtoiAs = models.FloatField(blank = True, default = -9.0)
    
    PropDMAtoMMA = models.FloatField(blank = True, default = -9.0)

    urine_specific_gravity = models.FloatField(blank=True, null=True, default = -9.0)
    
    iAs = models.FloatField(blank=True, null=True, default = -9.0)
    iAs_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    iAs_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    iAs_N = models.FloatField(blank=True, null=True, default = -9.0)

    AsB = models.FloatField(blank=True, null=True, default = -9.0)
    AsB_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    AsB_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    AsB_N = models.FloatField(blank=True, null=True, default = -9.0)

    AsIII  = models.FloatField(blank=True, null=True, default = -9.0)
    AsIII_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    AsIII_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    AsIII_N = models.FloatField(blank=True, null=True, default = -9.0)

    AsV  = models.FloatField(blank=True, null=True, default = -9.0)
    AsV_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    AsV_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    AsV_N = models.FloatField(blank=True, null=True, default = -9.0)

    DMA = models.FloatField(blank=True, null=True, default = -9.0)
    DMA_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    DMA_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    DMA_N = models.FloatField(blank=True, null=True, default = -9.0)

    MMA = models.FloatField(blank=True, null=True, default = -9.0)
    MMA_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    MMA_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    MMA_N = models.FloatField(blank=True, null=True, default = -9.0)

    Ba = models.FloatField(blank=True, null=True, default = -9.0)
    Ba_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Ba_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Ba_N = models.FloatField(blank=True, null=True, default = -9.0)

    Cs = models.FloatField(blank=True, null=True, default = -9.0)
    Cs_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Cs_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Cs_N = models.FloatField(blank=True, null=True, default = -9.0)

    W = models.FloatField(blank=True, null=True, default = -9.0)
    W_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    W_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    W_N = models.FloatField(blank=True, null=True, default = -9.0)

    Ag = models.FloatField(blank=True, null=True, default = -9.0)
    Ag_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Ag_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Ag_N = models.FloatField(blank=True, null=True, default = -9.0)

    Al = models.FloatField(blank=True, null=True, default = -9.0)
    Al_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Al_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Al_N = models.FloatField(blank=True, null=True, default = -9.0)

    As = models.FloatField(blank=True, null=True, default = -9.0)
    As_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    As_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    As_N = models.FloatField(blank=True, null=True, default = -9.0)

    Be = models.FloatField(blank=True, null=True, default = -9.0)
    Be_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Be_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Be_N = models.FloatField(blank=True, null=True, default = -9.0)

    Ca = models.FloatField(blank=True, null=True, default = -9.0)
    Ca_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Ca_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Ca_N = models.FloatField(blank=True, null=True, default = -9.0)

    Cd = models.FloatField(blank=True, null=True, default = -9.0)
    Cd_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Cd_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Cd_N = models.FloatField(blank=True, null=True, default = -9.0)

    Co = models.FloatField(blank=True, null=True, default = -9.0)
    Co_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Co_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Co_N = models.FloatField(blank=True, null=True, default = -9.0)

    Cr = models.FloatField(blank=True, null=True, default = -9.0)
    Cr_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Cr_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Cr_N = models.FloatField(blank=True, null=True, default = -9.0)

    Cs = models.FloatField(blank=True, null=True, default = -9.0)
    Cs_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Cs_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Cs_N = models.FloatField(blank=True, null=True, default = -9.0)

    Cu = models.FloatField(blank=True, null=True, default = -9.0)
    Cu_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Cu_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Cu_N = models.FloatField(blank=True, null=True, default = -9.0)

    Fe = models.FloatField(blank=True, null=True, default = -9.0)
    Fe_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Fe_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Fe_N = models.FloatField(blank=True, null=True, default = -9.0)

    Hg = models.FloatField(blank=True, null=True, default = -9.0)
    Hg_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Hg_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Hg_N = models.FloatField(blank=True, null=True, default = -9.0)

    K = models.FloatField(blank=True, null=True, default = -9.0)
    K_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    K_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    K_N = models.FloatField(blank=True, null=True, default = -9.0)

    Mg = models.FloatField(blank=True, null=True, default = -9.0)
    Mg_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Mg_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Mg_N = models.FloatField(blank=True, null=True, default = -9.0)

    Mn = models.FloatField(blank=True, null=True, default = -9.0)
    Mn_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Mn_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Mn_N = models.FloatField(blank=True, null=True, default = -9.0)

    Mo = models.FloatField(blank=True, null=True, default = -9.0)
    Mo_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Mo_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Mo_N = models.FloatField(blank=True, null=True, default = -9.0)

    Ni = models.FloatField(blank=True, null=True, default = -9.0)
    Ni_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Ni_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Ni_N = models.FloatField(blank=True, null=True, default = -9.0)

    P = models.FloatField(blank=True, null=True, default = -9.0)
    P_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    P_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    P_N = models.FloatField(blank=True, null=True, default = -9.0)

    Pb = models.FloatField(blank=True, null=True, default = -9.0)
    Pb_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Pb_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Pb_N = models.FloatField(blank=True, null=True, default = -9.0)

    Sb = models.FloatField(blank=True, null=True, default = -9.0)
    Sb_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Sb_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Sb_N = models.FloatField(blank=True, null=True, default = -9.0)

    Se = models.FloatField(blank=True, null=True, default = -9.0)
    Se_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Se_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Se_N = models.FloatField(blank=True, null=True, default = -9.0)

    Si = models.FloatField(blank=True, null=True, default = -9.0)
    Si_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Si_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Si_N = models.FloatField(blank=True, null=True, default = -9.0)

    Sn = models.FloatField(blank=True, null=True, default = -9.0)
    Sn_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Sn_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Sn_N = models.FloatField(blank=True, null=True, default = -9.0)

    Sr = models.FloatField(blank=True, null=True, default = -9.0)
    Sr_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Sr_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Sr_N = models.FloatField(blank=True, null=True, default = -9.0)

    Tl = models.FloatField(blank=True, null=True, default = -9.0)
    Tl_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Tl_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Tl_N = models.FloatField(blank=True, null=True, default = -9.0)

    U = models.FloatField(blank=True, null=True, default = -9.0)
    U_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    U_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    U_N = models.FloatField(blank=True, null=True, default = -9.0)

    W = models.FloatField(blank=True, null=True, default = -9.0)
    W_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    W_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    W_N = models.FloatField(blank=True, null=True, default = -9.0)

    Zn = models.FloatField(blank=True, null=True, default = -9.0)
    Zn_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    Zn_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    Zn_N = models.FloatField(blank=True, null=True, default = -9.0)

    V = models.FloatField(blank=True, null=True, default = -9.0)
    V_IDL = models.FloatField(blank=True, null=True, default = -9.0)
    V_BDL = models.CharField(max_length=100, choices=CAT_DAR_BDL, default = -9.0)
    V_N = models.FloatField(blank=True, null=True, default = -9.0)