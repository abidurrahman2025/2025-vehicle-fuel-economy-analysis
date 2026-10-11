# 2025 Vehicle Fuel Economy: Turbocharging and Fuel Type Effects

## Hypothesis Establishment and Research Framework

I used the 2025 fuel economy dataset from US Department of Energy. I will investigate two primary research questions: Do turbocharged engines deliver better fuel economy than naturally aspirated engines of similar displacement in the 2025 model year? And do vehicles requiring premium fuel demonstrate different fuel economy patterns compared to those using regular fuel in the 2025 model year?

Based on the research questions, I established two primary hypotheses for this analysis:

## Hypothesis 1:

**Null Hypothesis (H₀)**

There is no significant difference in combined fuel economy between turbocharged and naturally aspirated engines of similar displacement.

**Alternative Hypothesis (H₁)**

Turbocharged engines have significantly better combined fuel economy than naturally aspirated engines of similar displacement.

## Hypothesis 2:

**Null Hypothesis (H₀)**

There is no significant difference in fuel economy patterns between vehicles requiring premium fuel and those using regular fuel.

**Alternative Hypothesis (H₁)**

Vehicles requiring premium fuel show different fuel economy patterns compared to vehicles using regular fuel.

These hypotheses were tested using comprehensive statistical methods including t-tests, ANOVA, correlation analysis, and multiple regression modeling to control for potential confounding factors.

## Research Methodology and Analysis

*Note: You can click on the images to view them on a new page.*

The analysis began with loading and preparing the dataset from the 2025 Fuel Economy Guide:

```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import OLSInfluence
import requests, zipfile, io
import warnings
warnings.filterwarnings('ignore')

# Load the 2025 Fuel Economy dataset directly from the DOE website
url = "https://www.fueleconomy.gov/feg/epadata/25data.zip"
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    csv_name = [n for n in z.namelist() if n.lower().endswith('.csv')][0]
    df = pd.read_csv(z.open(csv_name), low_memory=False)

print("Dataset shape:", df.shape)
df.head()
```

```python
# Select and clean relevant columns for analysis - only 2025 model year
analysis_data = df[df['Model year'] == 2025].copy()

# Extract relevant columns and convert to proper numeric types
analysis_data['Eng_Disp1'] = pd.to_numeric(analysis_data['Eng Displ'], errors='coerce')
analysis_data['Comb_FE']   = pd.to_numeric(analysis_data['Comb FE (Guide) - Conventional Fuel'], errors='coerce')
analysis_data['City_FE']   = pd.to_numeric(analysis_data['City FE (Guide) - Conventional Fuel'], errors='coerce')
analysis_data['Hwy_FE']    = pd.to_numeric(analysis_data['Hwy FE (Guide) - Conventional Fuel'], errors='coerce')

# Create categorical variables for analysis
analysis_data['is_turbocharged'] = np.where(
    analysis_data['Air Aspiration Method Desc'].str.contains('Turbo', na=False),
    'Turbocharged', 'Naturally Aspirated'
)

analysis_data['fuel_type'] = np.where(
    analysis_data['Fuel usage Desc - Conventional Fuel'].str.contains('Premium', na=False),
    'Premium', 'Regular'
)

analysis_data['drive_type'] = np.where(
    analysis_data['Drive Desc'].str.contains('All wheel', na=False), 'AWD',
    np.where(analysis_data['Drive Desc'].str.contains('Front', na=False), 'FWD', 'Other')
)

# Remove rows with missing data in key analysis variables
analysis_data = analysis_data.dropna(subset=['Eng_Disp1', 'Comb_FE'])

# Display clean dataset dimensions
print("Clean dataset dimensions:", analysis_data.shape)
```

The final analysis dataset contained 868 vehicles with 169 variables after creating additional analysis columns.

```python
# HYPOTHESIS 1 ANALYSIS: Turbocharged vs Naturally Aspirated engines
# Split data by aspiration type for comparison
turbo_data = analysis_data[analysis_data['is_turbocharged'] == 'Turbocharged']
na_data    = analysis_data[analysis_data['is_turbocharged'] == 'Naturally Aspirated']

# Display summary statistics for both groups
print("Turbocharged group:")
print("  n =", len(turbo_data))
print("  mean Comb_FE =", turbo_data['Comb_FE'].mean())
print("  sd   Comb_FE =", turbo_data['Comb_FE'].std())
print("  mean Eng_Disp1 =", turbo_data['Eng_Disp1'].mean())

print("\nNaturally Aspirated group:")
print("  n =", len(na_data))
print("  mean Comb_FE =", na_data['Comb_FE'].mean())
print("  sd   Comb_FE =", na_data['Comb_FE'].std())
print("  mean Eng_Disp1 =", na_data['Eng_Disp1'].mean())
```

The dataset contained 586 turbocharged vehicles averaging 22.8 MPG with mean displacement 2.70 L, and 282 naturally aspirated vehicles averaging 26.2 MPG with mean displacement 3.48 L.

## Visualization 1: Turbocharged vs Naturally Aspirated Comparison

```python
# PLOT SECTION 1: Turbocharged vs Naturally Aspirated
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Boxplot comparing fuel economy by aspiration type
box_data = [turbo_data['Comb_FE'].dropna(), na_data['Comb_FE'].dropna()]
bp = axes[0, 0].boxplot(box_data, labels=['Turbocharged', 'Naturally Aspirated'],
                        patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen']):
    patch.set_facecolor(color)
axes[0, 0].set_title('Combined Fuel Economy\nby Aspiration Type', fontsize=10)
axes[0, 0].set_xlabel('Aspiration Type', fontsize=8)
axes[0, 0].set_ylabel('Combined MPG', fontsize=8)
axes[0, 0].tick_params(labelsize=8)

# Histogram of fuel economy for turbocharged vehicles
axes[0, 1].hist(turbo_data['Comb_FE'].dropna(), bins=15, color='lightblue',
                edgecolor='black')
axes[0, 1].set_title('Turbocharged Vehicles', fontsize=10)
axes[0, 1].set_xlabel('Combined MPG', fontsize=8)
axes[0, 1].set_ylabel('Frequency', fontsize=8)
axes[0, 1].tick_params(labelsize=8)

# Histogram of fuel economy for naturally aspirated vehicles
axes[1, 0].hist(na_data['Comb_FE'].dropna(), bins=15, color='lightgreen',
                edgecolor='black')
axes[1, 0].set_title('Naturally Aspirated Vehicles', fontsize=10)
axes[1, 0].set_xlabel('Combined MPG', fontsize=8)
axes[1, 0].set_ylabel('Frequency', fontsize=8)
axes[1, 0].tick_params(labelsize=8)

# Scatter plot showing relationship between displacement and MPG
colors = np.where(analysis_data['is_turbocharged'] == 'Turbocharged', 'red', 'blue')
axes[1, 1].scatter(analysis_data['Eng_Disp1'], analysis_data['Comb_FE'],
                   c=colors, s=16)
axes[1, 1].set_title('Engine Displacement vs Combined MPG', fontsize=10)
axes[1, 1].set_xlabel('Engine Displacement (L)', fontsize=8)
axes[1, 1].set_ylabel('Combined MPG', fontsize=8)
axes[1, 1].tick_params(labelsize=8)
from matplotlib.patches import Patch
axes[1, 1].legend(handles=[Patch(facecolor='red', label='Turbocharged'),
                           Patch(facecolor='blue', label='Naturally Aspirated')],
                  loc='upper right', fontsize=8)

fig.suptitle('Hypothesis 1: Turbocharged vs Naturally Aspirated Engines', fontsize=12)
plt.tight_layout()
plt.show()
```

In the Hypothesis 1 Visualization, the boxplot comparison showed naturally aspirated engines with both higher median values and greater variability in fuel economy. Turbocharged vehicles clustered in a relatively normal distribution around 20-25 MPG, while naturally aspirated engines exhibited a pronounced bimodal distribution with concentrations in both high-efficiency (small engines achieving 35+ MPG) and low-efficiency ranges (large engines below 20 MPG). The scatter plot showed that turbocharged engines (red points) predominantly occupying the medium displacement range (2-3L) with moderate efficiency, while naturally aspirated engines (blue points) demonstrated both extremely efficient small engines and inefficient large engines, visually explaining why the overall mean comparison might be misleading.

## Statistical Tests for Hypothesis 1

```python
# Statistical Test 1: Independent t-test comparing MPG by aspiration type
turbo_mpg = turbo_data['Comb_FE'].dropna()
na_mpg    = na_data['Comb_FE'].dropna()

t_stat, p_val = stats.ttest_ind(na_mpg, turbo_mpg, equal_var=False)
df_welch = (na_mpg.var()/len(na_mpg) + turbo_mpg.var()/len(turbo_mpg))**2 / (
    (na_mpg.var()/len(na_mpg))**2/(len(na_mpg)-1) +
    (turbo_mpg.var()/len(turbo_mpg))**2/(len(turbo_mpg)-1)
)
se_diff = np.sqrt(na_mpg.var()/len(na_mpg) + turbo_mpg.var()/len(turbo_mpg))
ci_low  = (na_mpg.mean() - turbo_mpg.mean()) - 1.96*se_diff
ci_high = (na_mpg.mean() - turbo_mpg.mean()) + 1.96*se_diff

print("Welch Two Sample t-test")
print(f"t = {t_stat:.4f}, df = {df_welch:.2f}, p-value = {p_val:.3e}")
print(f"95 percent confidence interval: {ci_low:.4f}  {ci_high:.4f}")
print(f"mean in group Naturally Aspirated: {na_mpg.mean():.5f}")
print(f"mean in group Turbocharged:        {turbo_mpg.mean():.5f}")
```
The t-test showed a statistically significant difference (p = 6.43e-08) with naturally aspirated engines averaging 3.43 MPG higher than turbocharged engines.

```python
# Statistical Test 2: Linear model controlling for engine displacement
lm_model1 = smf.ols('Comb_FE ~ C(is_turbocharged) + Eng_Disp1', data=analysis_data).fit()
print(lm_model1.summary())
```

The linear model revealed that even after controlling for engine displacement, turbocharged engines still showed a 7.06 MPG disadvantage (p < 2e-16), and each additional liter of displacement reduced fuel economy by 4.65 MPG per liter. The model explained 59.8% of the variance in fuel economy.

## Hypothesis 2 Analysis: Premium vs Regular Fuel Vehicles

### Descriptive Statistics

```python
# Split data by fuel type for comparison
premium_data = analysis_data[analysis_data['fuel_type'] == 'Premium']
regular_data = analysis_data[analysis_data['fuel_type'] == 'Regular']

# Display summary statistics for both fuel types
print("Premium fuel vehicles:")
print("  n =", len(premium_data))
print("  mean Comb_FE =", premium_data['Comb_FE'].mean())
print("  mean City_FE =", premium_data['City_FE'].mean())
print("  mean Hwy_FE  =", premium_data['Hwy_FE'].mean())

print("\nRegular fuel vehicles:")
print("  n =", len(regular_data))
print("  mean Comb_FE =", regular_data['Comb_FE'].mean())
print("  mean City_FE =", regular_data['City_FE'].mean())
print("  mean Hwy_FE  =", regular_data['Hwy_FE'].mean())
```

The dataset was nearly balanced with 433 premium fuel vehicles (21.8 MPG combined) and 435 regular fuel vehicles (26.0 MPG combined). Regular fuel vehicles showed advantages across all three fuel economy measures.

## Visualization 2: Fuel Type Analysis

```python
# PLOT SECTION 2: Fuel Type Analysis
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Boxplot for combined MPG by fuel type
bp1 = axes[0, 0].boxplot([premium_data['Comb_FE'].dropna(), regular_data['Comb_FE'].dropna()],
                         labels=['Premium', 'Regular'], patch_artist=True, widths=0.5)
for patch, color in zip(bp1['boxes'], ['lightcoral', 'lightyellow']):
    patch.set_facecolor(color)
axes[0, 0].set_title('Combined MPG by Fuel Type', fontsize=10)
axes[0, 0].set_xlabel('Fuel Type', fontsize=8)
axes[0, 0].set_ylabel('Combined MPG', fontsize=8)
axes[0, 0].tick_params(labelsize=8)

# Boxplot for city MPG by fuel type
bp2 = axes[0, 1].boxplot([premium_data['City_FE'].dropna(), regular_data['City_FE'].dropna()],
                         labels=['Premium', 'Regular'], patch_artist=True, widths=0.5)
for patch, color in zip(bp2['boxes'], ['lightcoral', 'lightyellow']):
    patch.set_facecolor(color)
axes[0, 1].set_title('City MPG by Fuel Type', fontsize=10)
axes[0, 1].set_xlabel('Fuel Type', fontsize=8)
axes[0, 1].set_ylabel('City MPG', fontsize=8)
axes[0, 1].tick_params(labelsize=8)

# Boxplot for highway MPG by fuel type
bp3 = axes[1, 0].boxplot([premium_data['Hwy_FE'].dropna(), regular_data['Hwy_FE'].dropna()],
                         labels=['Premium', 'Regular'], patch_artist=True, widths=0.5)
for patch, color in zip(bp3['boxes'], ['lightcoral', 'lightyellow']):
    patch.set_facecolor(color)
axes[1, 0].set_title('Highway MPG by Fuel Type', fontsize=10)
axes[1, 0].set_xlabel('Fuel Type', fontsize=8)
axes[1, 0].set_ylabel('Highway MPG', fontsize=8)
axes[1, 0].tick_params(labelsize=8)

# Information panel showing sample sizes
axes[1, 1].axis('off')
axes[1, 1].set_title('Fuel Type Comparison Summary', fontsize=10)
axes[1, 1].text(0.5, 0.5, f"Premium: n = {len(premium_data)}\nRegular: n = {len(regular_data)}",
                ha='center', va='center', fontsize=10)

fig.suptitle('Hypothesis 2: Premium vs Regular Fuel Vehicles', fontsize=12)
plt.tight_layout()
plt.show()
```

Hypothesis 2 Visualizations through the triple boxplot arrangement demonstrated consistent superiority of regular fuel vehicles across all three fuel economy measures. The nearly identical sample sizes (433 premium vs 435 regular) shown in the information panel strengthened the statistical validity of these comparisons. The consistent pattern across combined, city, and highway measures suggested a systematic difference rather than measurement artifact.

## Statistical Tests for Hypothesis 2

```python
# Statistical tests for different MPG measures by fuel type
t_comb, p_comb = stats.ttest_ind(premium_data['Comb_FE'].dropna(),
                                  regular_data['Comb_FE'].dropna(), equal_var=False)
t_city, p_city = stats.ttest_ind(premium_data['City_FE'].dropna(),
                                  regular_data['City_FE'].dropna(), equal_var=False)
t_hwy,  p_hwy  = stats.ttest_ind(premium_data['Hwy_FE'].dropna(),
                                  regular_data['Hwy_FE'].dropna(), equal_var=False)

print("Welch Two Sample t-test (Combined):")
print(f"  t = {t_comb:.4f}, p-value = {p_comb:.3e}")
print(f"  Premium mean = {premium_data['Comb_FE'].mean():.4f}, Regular mean = {regular_data['Comb_FE'].mean():.4f}")

print("\nWelch Two Sample t-test (City):")
print(f"  t = {t_city:.4f}, p-value = {p_city:.3e}")
print(f"  Premium mean = {premium_data['City_FE'].mean():.4f}, Regular mean = {regular_data['City_FE'].mean():.4f}")

print("\nWelch Two Sample t-test (Highway):")
print(f"  t = {t_hwy:.4f}, p-value = {p_hwy:.3e}")
print(f"  Premium mean = {premium_data['Hwy_FE'].mean():.4f}, Regular mean = {regular_data['Hwy_FE'].mean():.4f}")
```

All three t-tests showed statistically significant differences (p < 0.001) with regular fuel vehicles demonstrating superior fuel economy across all measures: 4.21 MPG higher for combined, 4.86 MPG for city, and 2.92 MPG for highway driving.

## Additional Comprehensive Analyses

```python
# CORRELATION ANALYSIS
# Calculate correlation matrix between key continuous variables
cor_vars = analysis_data[['Comb_FE', 'Eng_Disp1', 'City_FE', 'Hwy_FE']]
cor_matrix = cor_vars.corr()
print(cor_matrix)
```

```python
# PLOT SECTION 3: Correlation Matrix Visualization
from pandas.plotting import scatter_matrix

scatter_matrix(cor_vars.dropna(), figsize=(10, 10), alpha=0.5, diagonal='none',
               hist_kwds={'bins': 15})
plt.suptitle('Scatterplot Matrix of Key Variables', fontsize=12)
plt.tight_layout()
plt.show()
```

Correlation Matrix Visualization provided crucial validation of measurement consistency and revealed fundamental engineering relationships. The extremely high correlations between different fuel economy measures (r = 0.986 between combined and city MPG) confirmed internal consistency, while the clear negative relationship between engine displacement and all efficiency measures visually confirmed the fundamental physics of automotive efficiency.

```python
# TWO-WAY ANOVA: Testing interaction between fuel type and aspiration type
anova_model = smf.ols('Comb_FE ~ C(fuel_type) * C(is_turbocharged)', data=analysis_data).fit()
print(anova_lm(anova_model, typ=2))
```

```
                             sum_sq     df          F        PR(>F)
C(fuel_type)              3838.38    1.0    92.74      2.06e-16
C(is_turbocharged)         698.00    1.0    16.88      4.37e-05
C(fuel_type):C(is_turbocharged)  2138.00  1.0    51.66      1.43e-12
Residual                  35758.00  864.0      NaN           NaN
```

The two-way ANOVA revealed significant main effects for both fuel type (F = 92.74, p < 2e-16) and aspiration type (F = 16.88, p = 4.37e-05), along with a significant interaction (F = 51.66, p = 1.43e-12), indicating that the relationship between fuel type and MPG differs depending on whether the engine is turbocharged.

```python
# DISPLACEMENT CATEGORY ANALYSIS
# Create displacement categories for stratified analysis
bins   = [0, 2, 3, 4, analysis_data['Eng_Disp1'].max()]
labels = ['Small (<2L)', 'Medium (2-3L)', 'Large (3-4L)', 'Very Large (>4L)']
analysis_data['disp1_category'] = pd.cut(analysis_data['Eng_Disp1'],
                                          bins=bins, labels=labels)

# Calculate mean MPG by displacement category and aspiration type
disp1_summary = analysis_data.groupby(['disp1_category', 'is_turbocharged'])['Comb_FE'].agg(
    Comb_FE_mean='mean', Comb_FE_count='count')
print(disp1_summary)
```

```python
# PLOT SECTION 4: Displacement categories
means = analysis_data.groupby(['disp1_category', 'is_turbocharged'])['Comb_FE'].mean().reset_index()
bar_names = [f"{r.disp1_category}\n{r.is_turbocharged}" for _, r in means.iterrows()]

colors = ['lightblue' if 'Naturally' in n else 'lightgreen' for n in bar_names]
plt.figure(figsize=(12, 6))
plt.bar(range(len(means)), means['Comb_FE'], color=colors)
plt.xticks(range(len(means)), bar_names, rotation=90, fontsize=8)
plt.ylabel('Combined MPG', fontsize=9)
plt.title('Average MPG by Displacement and Aspiration Type', fontsize=11)
plt.ylim(0, means['Comb_FE'].max() * 1.1)
plt.tight_layout()
plt.show()
```

Displacement Category Analysis through the bar plot showed a nuanced nature of the turbocharging effect. The visualization showed that the natural aspiration advantage was most dramatic in smaller engine categories (37.6 vs 26.7 MPG in <2L category) and diminished in larger categories, suggesting that turbocharging technology might be more beneficial for larger engines where the efficiency penalty is reduced.

## Final Comprehensive Model and Diagnostics

```python
# Build the final comprehensive model
final_model = smf.ols(
    'Comb_FE ~ C(is_turbocharged) + Eng_Disp1 + C(fuel_type) + C(drive_type)',
    data=analysis_data
).fit()

print(final_model.summary())

# Extract diagnostics
influence = OLSInfluence(final_model)
std_residuals = influence.resid_studentized_internal
leverage = influence.hat_matrix_diag
fitted = final_model.fittedvalues
```

```python
# PLOT SECTION 5: Final Model Diagnostic Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Residuals vs Fitted
axes[0, 0].scatter(fitted, final_model.resid, c='blue', s=16)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_title('Residuals vs Fitted Values', fontsize=10)
axes[0, 0].set_xlabel('Fitted Values', fontsize=8)
axes[0, 0].set_ylabel('Residuals', fontsize=8)
axes[0, 0].tick_params(labelsize=8)

# Plot 2: Normal Q-Q
sm.qqplot(std_residuals, line='s', ax=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q Plot', fontsize=10)
axes[0, 1].tick_params(labelsize=8)

# Plot 3: Scale-Location
sqrt_abs_resid = np.sqrt(np.abs(std_residuals))
axes[1, 0].scatter(fitted, sqrt_abs_resid, c='blue', s=16)
# lowess-style smoother
z = np.polyfit(fitted, sqrt_abs_resid, 1)
p = np.poly1d(z)
axes[1, 0].plot(sorted(fitted), p(sorted(fitted)), color='darkgreen', lw=2)
axes[1, 0].set_title('Scale-Location Plot', fontsize=10)
axes[1, 0].set_xlabel('Fitted values', fontsize=8)
axes[1, 0].set_ylabel('√|Standardized Residuals|', fontsize=8)
axes[1, 0].tick_params(labelsize=8)

# Plot 4: Residuals vs Leverage
axes[1, 1].scatter(leverage, std_residuals, c='blue', s=16)
axes[1, 1].axhline(y=0, color='red', linestyle='--')
axes[1, 1].set_title('Residuals vs Leverage', fontsize=10)
axes[1, 1].set_xlabel('Leverage', fontsize=8)
axes[1, 1].set_ylabel('Standardized Residuals', fontsize=8)
axes[1, 1].tick_params(labelsize=8)

fig.suptitle('Final Model Diagnostic Plots', fontsize=12)
plt.tight_layout()
plt.show()
```

Model Diagnostic Plots provided essential validation of the regression assumptions. The relatively random scatter in the residuals vs fitted plot, the approximately linear pattern in the Q-Q plot, and the absence of extreme leverage points all supported the validity of the multivariate conclusions.

```python
# Calculate and display key comparative metrics for summary
mean_turbo   = turbo_data['Comb_FE'].mean()
mean_na      = na_data['Comb_FE'].mean()
mean_premium = premium_data['Comb_FE'].mean()
mean_regular = regular_data['Comb_FE'].mean()

diff_turbo_na        = mean_turbo - mean_na
diff_premium_regular = mean_premium - mean_regular

print(f"mean_turbo:   {mean_turbo}")
print(f"mean_na:      {mean_na}")
print(f"mean_premium: {mean_premium}")
print(f"mean_regular: {mean_regular}")
print(f"diff_turbo_na:        {diff_turbo_na}")
print(f"diff_premium_regular: {diff_premium_regular}")
print(f"p_value_turbo: {p_val}")
print(f"p_value_fuel:  {p_comb}")

cor_displ_mpg = analysis_data['Eng_Disp1'].corr(analysis_data['Comb_FE'])
print(f"cor_displ_mpg: {cor_displ_mpg}")
```

The metrics quantify the central findings: naturally aspirated engines achieve 3.43 MPG higher fuel economy than turbocharged engines (p = 6.43e-08), regular fuel vehicles show 4.21 MPG advantage over premium fuel vehicles in bivariate analysis (p = 1.94e-19), and engine displacement shows a strong negative correlation with fuel economy (r = -0.629).

## Research Abstract and Conclusion

This comprehensive analysis of 2025 model year vehicle fuel economy yields definitive conclusions regarding the two research hypotheses. For Hypothesis 1, we must reject the alternative hypothesis that turbocharged engines provide better fuel economy. The evidence strongly supports the opposite conclusion: naturally aspirated engines demonstrate significantly higher fuel economy both in simple comparisons (26.21 vs 22.78 MPG, p = 6.43e-08) and after controlling for engine displacement (7.06 MPG disadvantage for turbocharged engines, p < 2e-16). The visualization analysis revealed that this relationship is most pronounced in smaller engine categories, suggesting that the efficiency benefits of natural aspiration are particularly valuable in compact vehicle applications.

For Hypothesis 2, we find mixed evidence. While initial bivariate comparisons show substantial differences favoring regular fuel vehicles (25.99 vs 21.79 MPG, p < 1.94e-19), these differences become statistically non-significant in multivariate analysis — the relationship is largely confounded by other vehicle characteristics, particularly that premium fuel requirements are associated with vehicles that have other characteristics (larger engines, turbocharging) that independently reduce fuel economy.

The strong negative correlation between engine displacement and fuel economy (r = -0.629) remains a fundamental relationship in the 2025 vehicle fleet, while drive type emerges as a significant factor with front-wheel drive vehicles showing approximately 2.9 MPG advantage over all-wheel drive configurations. The comprehensive visualization approach proved essential for understanding the complex relationships and provided context that pure numerical analysis could not capture.

These findings have important implications for both automotive engineering and consumer decision-making, suggesting that the pursuit of turbocharging for fuel efficiency goals may be misguided, and that fuel type requirements should be considered in the broader context of overall vehicle design rather than as independent efficiency factors.

## References

U.S. Department of Energy, & U.S. Environmental Protection Agency. (2025). *2025 Fuel Economy Data* [Data set]. FuelEconomy.gov. https://www.fueleconomy.gov/feg/download.shtml
