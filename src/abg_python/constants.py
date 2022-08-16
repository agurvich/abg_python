## define some constants
HYDROGENMASS = 1.67e-24  # g
cm_per_kpc =3.08567758e21 #cm/kpc
Gcgs = 6.674e-8 #cm3/(g s^2)
SOLAR_MASS_g = 1.988435e33 #1.989e33 ## g
seconds_per_year = 31557600 # s/yr
#cm3/(g s^2) * kpc/cm * (km/cm)^2 * g/msun * msun/mcode= kpc/mcode (km/s)^2  
#Gcgs/cm_per_kpc*SOLAR_MASS_g/1e10*1e10 = 
Gcode = 43099.305194805194
## km/s -> kpc/gyr
kms_to_kpcgyr = 1/(cm_per_kpc/1e5)*seconds_per_year*1e9

## code density -> cm^-3
# Code mass -> g , (code length)^-3 -> cm^-3 , g -> nH
DENSITYFACT = 1e10*SOLAR_MASS_g*cm_per_kpc**-3/HYDROGENMASS 
