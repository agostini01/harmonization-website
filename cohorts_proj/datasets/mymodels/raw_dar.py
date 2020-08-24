from django.db import models

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

class RawDAR(models.Model):

    # PIN_Patient: unique identifier
    # TODO: Maybe update to UUIDField
    # TODO: Maybe create a ManyToManyField for UUID
    unq_id = models.CharField(max_length=100)

    assay = models.CharField(max_length=100)
    # lab = models.CharField(max_length=100)

    # participant_type – categorical variable: maternal = mother; child = child
    participant_type = models.CharField(
        max_length=15, choices=CAT_DAR_MEMBER_C)

    # time_period – categorical: 12G, 24G, 6WP, 6MP, 1YP, 2YP, 3YP, 5YP
    time_period = models.CharField(max_length=3, choices=CAT_DAR_TIME_PERIOD)
    
    # batch - numeric: batch number
    batch = models.IntegerField()

    # squid - unique identifier: Sample identifier
    # squid = models.CharField(max_length=100)

    # sample_gestage_days - numeric: days of gestation
    sample_gestage_days = models.IntegerField()

    # Outcome – categorical variable: 1 = preterm birth; 0 = term
    preterm = models.CharField(max_length=3, choices=CAT_DAR_OUTCOME, blank=True)

    # List of analytes, Index of detection level, Above/Below IDL
    # Floating values unit: 1ppb = 1ug/L

    urine_specific_gravity = models.FloatField(blank=True, null=True)
    
    iAs = models.FloatField(blank=True, null=True)
    iAs_IDL = models.FloatField(blank=True, null=True)
    iAs_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    AsB = models.FloatField(blank=True, null=True)
    AsB_IDL = models.FloatField(blank=True, null=True)
    AsB_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    DMA = models.FloatField(blank=True, null=True)
    DMA_IDL = models.FloatField(blank=True, null=True)
    DMA_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    MMA = models.FloatField(blank=True, null=True)
    MMA_IDL = models.FloatField(blank=True, null=True)
    MMA_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Ba = models.FloatField(blank=True, null=True)
    Ba_IDL = models.FloatField(blank=True, null=True)
    Ba_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Cs = models.FloatField(blank=True, null=True)
    Cs_IDL = models.FloatField(blank=True, null=True)
    Cs_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Sr = models.FloatField(blank=True, null=True)
    Sr_IDL = models.FloatField(blank=True, null=True)
    Sr_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    W = models.FloatField(blank=True, null=True)
    W_IDL = models.FloatField(blank=True, null=True)
    W_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Ag = models.FloatField(blank=True, null=True)
    Ag_IDL = models.FloatField(blank=True, null=True)
    Ag_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Al = models.FloatField(blank=True, null=True)
    Al_IDL = models.FloatField(blank=True, null=True)
    Al_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    As = models.FloatField(blank=True, null=True)
    As_IDL = models.FloatField(blank=True, null=True)
    As_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Be = models.FloatField(blank=True, null=True)
    Be_IDL = models.FloatField(blank=True, null=True)
    Be_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Cd = models.FloatField(blank=True, null=True)
    Cd_IDL = models.FloatField(blank=True, null=True)
    Cd_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Co = models.FloatField(blank=True, null=True)
    Co_IDL = models.FloatField(blank=True, null=True)
    Co_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Cr = models.FloatField(blank=True, null=True)
    Cr_IDL = models.FloatField(blank=True, null=True)
    Cr_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Cu = models.FloatField(blank=True, null=True)
    Cu_IDL = models.FloatField(blank=True, null=True)
    Cu_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Fe = models.FloatField(blank=True, null=True)
    Fe_IDL = models.FloatField(blank=True, null=True)
    Fe_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Hg = models.FloatField(blank=True, null=True)
    Hg_IDL = models.FloatField(blank=True, null=True)
    Hg_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Mn = models.FloatField(blank=True, null=True)
    Mn_IDL = models.FloatField(blank=True, null=True)
    Mn_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Mo = models.FloatField(blank=True, null=True)
    Mo_IDL = models.FloatField(blank=True, null=True)
    Mo_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Ni = models.FloatField(blank=True, null=True)
    Ni_IDL = models.FloatField(blank=True, null=True)
    Ni_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Pb = models.FloatField(blank=True, null=True)
    Pb_IDL = models.FloatField(blank=True, null=True)
    Pb_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Sb = models.FloatField(blank=True, null=True)
    Sb_IDL = models.FloatField(blank=True, null=True)
    Sb_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Se = models.FloatField(blank=True, null=True)
    Se_IDL = models.FloatField(blank=True, null=True)
    Se_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Sn = models.FloatField(blank=True, null=True)
    Sn_IDL = models.FloatField(blank=True, null=True)
    Sn_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Tl = models.FloatField(blank=True, null=True)
    Tl_IDL = models.FloatField(blank=True, null=True)
    Tl_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    U = models.FloatField(blank=True, null=True)
    U_IDL = models.FloatField(blank=True, null=True)
    U_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    W = models.FloatField(blank=True, null=True)
    W_IDL = models.FloatField(blank=True, null=True)
    W_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Zn = models.FloatField(blank=True, null=True)
    Zn_IDL = models.FloatField(blank=True, null=True)
    Zn_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    V = models.FloatField(blank=True, null=True)
    V_IDL = models.FloatField(blank=True, null=True)
    V_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)
