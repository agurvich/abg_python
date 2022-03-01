## define some constants
# Code mass -> g , (code length)^-3 -> cm^-3 , g -> nH
DENSITYFACT = 2e43*(3.086e21)**-3/(1.67e-24)
HYDROGENMASS = 1.67e-24  # g
cm_per_kpc = 3.08e21 # cm/kpc
Gcgs = 6.674e-8 #cm3/(g s^2)
SOLAR_MASS_g = 1.989e33 ## g
seconds_per_year = 3.15e7
#cm3/(g s^2) * kpc/cm * (km/cm)^2 * g/msun * msun/mcode= kpc/mcode (km/s)^2  
#Gcgs/cm_per_kpc*SOLAR_MASS_g/1e10*1e10 = 
Gcode = 43099.305194805194
## km/s -> kpc/gyr
kms_to_kpcgyr = 1/(cm_per_kpc/1e5)*seconds_per_year*1e9