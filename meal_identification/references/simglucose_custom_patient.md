# Parameter Data Dictionary

## Quest
- CR: carbs ratio (CR)
- CF: correction factor (CF)
- TDI: Total Daily Insulin
- Age: Age of the patient

## Pump Parameters
- min_bolus: the minimum amount of bolus a pump can administer
- max_bolus: the maximum amount of bolus a pump can administer
- inc_bolus: the increment amount of bolus
- min_basal: the minimum amount of basal insulin the pump can administer
- max_basal: the maximum amount of basal insulin the pump can administer
- inc_basal: the increment amount of basal
- sample_time: how often the pump receives a reading.

## Sensor Parameters
### Johnson_S_U
These parameters add noise to the CGM readings
- johnson_SU (xi + lam * np.sinh((x - gamma) / delta)): https://www.jstor.org/stable/2332539?origin=crossref&seq=4
- PACF: Partial Autocorrelation Function
- gamma: (affect only the expected value (or other central measure) of the distribution of) parameter in the johnson_SU function.
- lambda: (distribution shape) parameter in the johnson_SU function
- delta: (The scale factor) parameter in the johnson_SU function
- xi: (distribution shape) parameter in the johnson_SU function

### CGM Sensor
- sample_time: how often the sensor samples (provides a cgm reading)
- min: minimum value of the sensor
- max: maximum value of the sensor

## Patient Parameters
- i: does not seem to be used anywhere obvious
- x0_1:
- BW: Body Weight
-
-
