from django.db import models

# Each row of raw Dartmounth datset has the following fields
# unq_id
# assay
# lab
# participant_type
# time_period
# batch
# squid
# sample_gestage_days
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


class RawDAR(models.Model):

    # PIN_Patient: unique identifier
    # TODO: Maybe update to UUIDField
    # TODO: Maybe create a ManyToManyField for UUID
    unq_id = models.CharField(max_length=100)

    assay = models.CharField(max_length=100)
    lab = models.CharField(max_length=100)

    # participant_type – categorical variable: maternal = mother; child = child
    participant_type = models.CharField(
        max_length=15, choices=CAT_DAR_MEMBER_C)

    # time_period – categorical: 12G, 24G, 6WP, 6MP, 1YP, 2YP, 3YP, 5YP
    time_period = models.CharField(max_length=3, choices=CAT_DAR_TIME_PERIOD)

    # batch - numeric: batch number
    batch = models.IntegerField()

    # squid - unique identifier: Sample identifier
    squid = models.CharField(max_length=100)

    # sample_gestage_days - numeric: days of gestation
    sample_gestage_days = models.IntegerField()

    # List of analytes, Index of detection level, Above/Below IDL

    Ag = models.FloatField()
    Ag_IDL = models.FloatField()
    Ag_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Al = models.FloatField()
    Al_IDL = models.FloatField()
    Al_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    As = models.FloatField()
    As_IDL = models.FloatField()
    As_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Be = models.FloatField()
    Be_IDL = models.FloatField()
    Be_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Cd = models.FloatField()
    Cd_IDL = models.FloatField()
    Cd_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Co = models.FloatField()
    Co_IDL = models.FloatField()
    Co_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Cr = models.FloatField()
    Cr_IDL = models.FloatField()
    Cr_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Cu = models.FloatField()
    Cu_IDL = models.FloatField()
    Cu_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Fe = models.FloatField()
    Fe_IDL = models.FloatField()
    Fe_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Hg = models.FloatField()
    Hg_IDL = models.FloatField()
    Hg_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Mn = models.FloatField()
    Mn_IDL = models.FloatField()
    Mn_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Mo = models.FloatField()
    Mo_IDL = models.FloatField()
    Mo_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Ni = models.FloatField()
    Ni_IDL = models.FloatField()
    Ni_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Pb = models.FloatField()
    Pb_IDL = models.FloatField()
    Pb_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Sb = models.FloatField()
    Sb_IDL = models.FloatField()
    Sb_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Se = models.FloatField()
    Se_IDL = models.FloatField()
    Se_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Sn = models.FloatField()
    Sn_IDL = models.FloatField()
    Sn_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Tl = models.FloatField()
    Tl_IDL = models.FloatField()
    Tl_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    U = models.FloatField()
    U_IDL = models.FloatField()
    U_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    W = models.FloatField()
    W_IDL = models.FloatField()
    W_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    Zn = models.FloatField()
    Zn_IDL = models.FloatField()
    Zn_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)

    V = models.FloatField()
    V_IDL = models.FloatField()
    V_BDL = models.CharField(max_length=3, choices=CAT_DAR_BDL)
