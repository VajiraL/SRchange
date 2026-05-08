# Time Periods:
## for climate
Historical (one 30-year climatology):
1901-1930
1931-1960
1961-1990
1991-2020

Future (here we have alternative 30-year climatologies under SSP-RCPs)
2041-2070
2071-2099

## for LU
We represent LU as the most frequent LU class within the last decade of each period.
Intact: An idealized LU with no human modification (No Hilda+ LU classes, Use "Wild Anthromes")
Pre-industrial: Around 1850 ((No Hilda+ LU classes, use Anthromes))

1901-1930: mode of 1921-1930
1931-1960: mode of 1941-1960
1961-1990: mode of 1981-1990
1991-2020: mode of 2011-2020

2041-2070: mode of 2061-2070
2071-2099: mode of 2090-2099

# Future SSP-RCPs:
We have climatologies for:
SSP1-RCP1.9: 'ssp119'
SSP1-RCP2.6: 'ssp126'
SSP2-RCP4.5: 'ssp245'
SSP3-RCP7.0: 'ssp370'
SSP4-RCP3.4: 'ssp434'
SSP4-RCP6.0: 'ssp460'
SSP5-RCP8.5: 'ssp585'

We have LU only for the following. So we only use these in this study:
SSP1-RCP2.6: 'ssp126'
SSP2-RCP4.5: 'ssp245'
SSP3-RCP7.0: 'ssp370'
SSP4-RCP6.0: 'ssp460'

# Training scenario
ML-models of SR (Root zone storage capacity) are trained on present SR using present climate (1991-2020) and LU (1991-2020).

# prediction
## key prediction scenarios

|                  |                                 |                             |                             |                             |                             |                                    |                                           |                                           |                                           |                                           |                                           |                                           |                                           |                                           |
| ---------------- | ------------------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- | ---------------------------------- | ----------------------------------------- | ----------------------------------------- | ----------------------------------------- | ----------------------------------------- | ----------------------------------------- | ----------------------------------------- | ----------------------------------------- | ----------------------------------------- |
| Climate          | Intact                          | PI                          | 1901_1930                   | 1931_1960                   | 1961_1990                   | 1991_2020                          | 2041_2070_ssp126                          | 2041_2070_ssp245                          | 2041_2070_ssp370                          | 2041_2070_ssp460                          | 2071_2099_ssp126                          | 2071_2099_ssp245                          | 2071_2099_ssp370                          | 2071_2099_ssp460                          |
| 1901_1930        | 1901_1930_Clim_Intact_LU        | 1901_1930_Clim_PI_LU        | 1901_1930_Clim_1901_1930_LU |                             |                             | 1901_1930_Clim_1991_2020_LU        |                                           |                                           |                                           |                                           |                                           |                                           |                                           |                                           |
| 1931_1960        | 1931_1960_Clim_Intact_LU        | 1931_1960_Clim_PI_LU        |                             | 1931_1960_Clim_1931_1960_LU |                             | 1931_1960_Clim_1991_2020_LU        |                                           |                                           |                                           |                                           |                                           |                                           |                                           |                                           |
| 1961_1990        | 1961_1990_Clim_Intact_LU        | 1961_1990_Clim_PI_LU        |                             |                             | 1961_1990_Clim_1961_1990_LU | 1961_1990_Clim_1991_2020_LU        |                                           |                                           |                                           |                                           |                                           |                                           |                                           |                                           |
| 1991_2020        | 1991_2020_Clim_Intact_LU        | 1991_2020_Clim_PI_LU        |                             |                             |                             | 1991_2020_Clim_1991_2020_LU        |                                           |                                           |                                           |                                           |                                           |                                           |                                           |                                           |
| 2041_2070_ssp126 | 2041_2070_ssp126_Clim_Intact_LU | 2041_2070_ssp126_Clim_PI_LU |                             |                             |                             | 2041_2070_ssp126_Clim_1991_2020_LU | 2041_2070_ssp126_Clim_2041_2070_ssp126_LU |                                           |                                           |                                           |                                           |                                           |                                           |                                           |
| 2041_2070_ssp245 | 2041_2070_ssp245_Clim_Intact_LU | 2041_2070_ssp245_Clim_PI_LU |                             |                             |                             | 2041_2070_ssp245_Clim_1991_2020_LU |                                           | 2041_2070_ssp245_Clim_2041_2070_ssp245_LU |                                           |                                           |                                           |                                           |                                           |                                           |
| 2041_2070_ssp370 | 2041_2070_ssp370_Clim_Intact_LU | 2041_2070_ssp370_Clim_PI_LU |                             |                             |                             | 2041_2070_ssp370_Clim_1991_2020_LU |                                           |                                           | 2041_2070_ssp370_Clim_2041_2070_ssp370_LU |                                           |                                           |                                           |                                           |                                           |
| 2041_2070_ssp460 | 2041_2070_ssp460_Clim_Intact_LU | 2041_2070_ssp460_Clim_PI_LU |                             |                             |                             | 2041_2070_ssp460_Clim_1991_2020_LU |                                           |                                           |                                           | 2041_2070_ssp460_Clim_2041_2070_ssp460_LU |                                           |                                           |                                           |                                           |
| 2071_2099_ssp126 | 2071_2099_ssp126_Clim_Intact_LU | 2071_2099_ssp126_Clim_PI_LU |                             |                             |                             | 2071_2099_ssp126_Clim_1991_2020_LU |                                           |                                           |                                           |                                           | 2071_2099_ssp126_Clim_2071_2099_ssp126_LU |                                           |                                           |                                           |
| 2071_2099_ssp245 | 2071_2099_ssp245_Clim_Intact_LU | 2071_2099_ssp245_Clim_PI_LU |                             |                             |                             | 2071_2099_ssp245_Clim_1991_2020_LU |                                           |                                           |                                           |                                           |                                           | 2071_2099_ssp245_Clim_2071_2099_ssp245_LU |                                           |                                           |
| 2071_2099_ssp370 | 2071_2099_ssp370_Clim_Intact_LU | 2071_2099_ssp370_Clim_PI_LU |                             |                             |                             | 2071_2099_ssp370_Clim_1991_2020_LU |                                           |                                           |                                           |                                           |                                           |                                           | 2071_2099_ssp370_Clim_2071_2099_ssp370_LU |                                           |
| 2071_2099_ssp460 | 2071_2099_ssp460_Clim_Intact_LU | 2071_2099_ssp460_Clim_PI_LU |                             |                             |                             | 2071_2099_ssp460_Clim_1991_2020_LU |                                           |                                           |                                           |                                           |                                           |                                           |                                           | 2071_2099_ssp460_Clim_2071_2099_ssp460_LU |

