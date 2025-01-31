import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


def calculate_hill_slope_ic50(concentrations, activities, sample_num):
    """
    Calculate the Hill Slope, IC50, and AC50 from given concentration-response data.

    Parameters:
    concentrations (array-like): A list or numpy array of compound concentrations (μM).
    activities (array-like): A list or numpy array of corresponding FP-Activity values.

    Returns:
    dict: A dictionary containing the Hill Slope, EC50, IC50, and AC50 values.
    """
    print(sample_num)
    # Define the Hill equation
    def hill_equation(concentration, Emax, EC50, hill_slope):
        return Emax * (concentration ** hill_slope) / (EC50 ** hill_slope + concentration ** hill_slope)

    # Automatically determine initial parameters p0
    def auto_p0(concentrations, activities):
        Emax_init = min(activities)  # Maximum inhibition effect
        EC50_init = np.median(concentrations)  # Median concentration as starting EC50
        Hill_Slope_init = 1  # Default Hill Slope
        return [Emax_init, EC50_init, Hill_Slope_init]

    # Fit the data using the Hill equation
    try:
        p0 = auto_p0(concentrations, activities)
        popt, _ = curve_fit(hill_equation, concentrations, activities, p0=p0)
        Emax_fit, EC50_fit, hill_slope_fit = popt
    except RuntimeError:
        print("Curve fitting failed. Check data quality.")
        return np.nan, np.nan, np.nan, np.nan
        #return {"Error": "Curve fitting failed. Check data quality."}

    # Compute IC50 using interpolation
    min_activity = min(activities)
    max_activity = max(activities)
    target_activity = (max_activity + min_activity) / 2  # 50% inhibition level

    try:
        interp_func = interp1d(activities, concentrations, kind='linear', fill_value="extrapolate")
        IC50_fit = interp_func(target_activity)
    except ValueError:
        IC50_fit = np.nan  # Assign NaN if interpolation fails

    # Compute AC50 (alternative method, based on midpoint response)
    try:
        ac50_func = interp1d(activities, concentrations, kind='linear', fill_value="extrapolate")
        AC50_fit = ac50_func(0)  # Midpoint activity (50% effect level)
    except ValueError:
        AC50_fit = np.nan

    # Generate fitted curve
    concentration_range = np.logspace(np.log10(concentrations.min()), np.log10(concentrations.max()), 100)
    fitted_activities = hill_equation(concentration_range, *popt)

    # Plot the concentration-response curve
    plt.figure(figsize=(8, 6))
    plt.plot(concentrations, activities, 'o', label='Observed Data', markersize=8)
    plt.plot(concentration_range, fitted_activities, '-', label='Fitted Hill Curve')
    plt.xscale('log')
    plt.xlabel('Concentration (μM)')
    plt.ylabel('FP-Activity')
    plt.title('Concentration-Response Curve (Hill Fit)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'hillslope\\{sample_num}.png')

    print(hill_slope_fit)
    print(AC50_fit)
    print(EC50_fit)
    print(IC50_fit)
    # Return results as a dictionary
    return hill_slope_fit, EC50_fit, IC50_fit, AC50_fit