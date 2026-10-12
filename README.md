# 2025 Car Fuel Economy Analysis

Requires: `pandas`, `numpy`, `matplotlib`, `scipy`, `statsmodels`, `openpyxl` (Excel reader used by pandas).

```python
# Uncomment if packages are missing
# %pip install pandas numpy matplotlib scipy statsmodels openpyxl
```

```python
import io
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.nonparametric.smoothers_lowess import lowess
```

## Load data

```python
URL = "https://www.fueleconomy.gov/feg/epadata/25data.zip"

req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    zf = zipfile.ZipFile(io.BytesIO(resp.read()))

print(zf.namelist())

xlsx_name = [n for n in zf.namelist() if n.lower().endswith(".xlsx")][0]
df = pd.read_excel(io.BytesIO(zf.read(xlsx_name)), sheet_name="FEguide")

print(df.shape)
print(list(df.columns[:10]))
```

```
['20250814 MY25 ICE,EV,PHEV For DOE R2public.xlsx']
(868, 162)
['Model Year', 'Mfr Name', 'Division', 'Carline', 'Verify Mfr Cd', 'Index (Model Type Index)', 'Eng Displ', '# Cyl', 'Transmission', 'City FE (Guide) - Conventional Fuel']
```

## Clean data and build variables

```python
# Only 2025 model year
data = df[df["Model Year"] == 2025].copy()

# Numeric columns
data["Eng_Displ"] = pd.to_numeric(data["Eng Displ"], errors="coerce")
data["Comb_FE"] = pd.to_numeric(data["Comb FE (Guide) - Conventional Fuel"], errors="coerce")
data["City_FE"] = pd.to_numeric(data["City FE (Guide) - Conventional Fuel"], errors="coerce")
data["Hwy_FE"] = pd.to_numeric(data["Hwy FE (Guide) - Conventional Fuel"], errors="coerce")

# Categorical variables
data["is_turbocharged"] = np.where(
    data["Air Aspiration Method Desc"].str.contains("Turbo", na=False),
    "Turbocharged", "Naturally Aspirated")

data["fuel_type"] = np.where(
    data["Fuel Usage Desc - Conventional Fuel"].str.contains("Premium", na=False),
    "Premium", "Regular")

drive = data["Drive Desc"]
data["drive_type"] = np.select(
    [drive.str.contains("All Wheel", na=False),
     drive.str.contains("Front", na=False),
     drive.str.contains("Rear", na=False)],
    ["AWD", "FWD", "RWD"], default="Other")

# Drop rows missing key variables
data = data.dropna(subset=["Eng_Displ", "Comb_FE"])
print(data.shape)
```

```
(868, 169)
```

## Hypothesis 1: Turbocharged vs Naturally Aspirated

```python
turbo_data = data[data["is_turbocharged"] == "Turbocharged"]
na_data = data[data["is_turbocharged"] == "Naturally Aspirated"]

for name, d in [("Turbocharged", turbo_data), ("Naturally Aspirated", na_data)]:
    print(f"{name}: n={len(d)}, mean MPG={d['Comb_FE'].mean():.3f}, "
          f"sd={d['Comb_FE'].std():.3f}, mean displ={d['Eng_Displ'].mean():.3f}")
```

```
Turbocharged: n=586, mean MPG=22.778, sd=4.636, mean displ=2.703
Naturally Aspirated: n=282, mean MPG=26.209, sd=9.914, mean displ=3.484
```

### Plot 1: Aspiration type

```python
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# Boxplot
bp = axes[0, 0].boxplot([na_data["Comb_FE"], turbo_data["Comb_FE"]],
                        tick_labels=["Naturally Aspirated", "Turbocharged"],
                        patch_artist=True)
for patch, c in zip(bp["boxes"], ["lightblue", "lightgreen"]):
    patch.set_facecolor(c)
axes[0, 0].set(title="Combined Fuel Economy\nby Aspiration Type",
               xlabel="Aspiration Type", ylabel="Combined MPG")

# Histograms
axes[0, 1].hist(turbo_data["Comb_FE"], bins=15, color="lightblue", edgecolor="black")
axes[0, 1].set(title="Turbocharged Vehicles", xlabel="Combined MPG", ylabel="Frequency")

axes[1, 0].hist(na_data["Comb_FE"], bins=15, color="lightgreen", edgecolor="black")
axes[1, 0].set(title="Naturally Aspirated Vehicles", xlabel="Combined MPG", ylabel="Frequency")

# Scatter
colors = np.where(data["is_turbocharged"] == "Turbocharged", "red", "blue")
axes[1, 1].scatter(data["Eng_Displ"], data["Comb_FE"], c=colors, s=15)
axes[1, 1].scatter([], [], c="red", label="Turbocharged")
axes[1, 1].scatter([], [], c="blue", label="Naturally Aspirated")
axes[1, 1].legend(loc="upper right", fontsize=8)
axes[1, 1].set(title="Engine Displacement vs Combined MPG",
               xlabel="Engine Displacement (L)", ylabel="Combined MPG")

fig.suptitle("Hypothesis 1: Turbocharged vs Naturally Aspirated Engines")
plt.tight_layout()
plt.show()
```
<img width="1090" height="789" alt="plot1" src="https://github.com/user-attachments/assets/c0bd5387-aa95-4579-8f32-90e6ba6b9ad4" />


In the Hypothesis 1 Visualization, the boxplot comparison showed naturally aspirated engines with both higher median values and greater variability in fuel economy. Turbocharged vehicles clustered in a relatively normal distribution around 20-25 MPG, while naturally aspirated engines exhibited a pronounced bimodal distribution with concentrations in both high-efficiency (small engines achieving 35+ MPG) and low-efficiency ranges (large engines below 20 MPG). The scatter plot showed that turbocharged engines (red points) predominantly occupying the medium displacement range (2-3L) with moderate efficiency, while naturally aspirated engines (blue points) demonstrated both extremely efficient small engines and inefficient large engines, visually explaining why the overall mean comparison might be misleading.

### Statistical Tests for Hypothesis 1

```python
# Welch t-test (same as R's default t.test)
def welch(df, value, group):
    levels = sorted(df[group].unique())
    a = df.loc[df[group] == levels[0], value]
    b = df.loc[df[group] == levels[1], value]
    res = stats.ttest_ind(a, b, equal_var=False)
    ci = res.confidence_interval()
    print(f"{value} by {group}: {levels[0]} (mean={a.mean():.3f}) vs {levels[1]} (mean={b.mean():.3f})")
    print(f"  t = {res.statistic:.4f}, df = {res.df:.2f}, p = {res.pvalue:.4g}")
    print(f"  95% CI of difference: [{ci.low:.3f}, {ci.high:.3f}]\n")
    return res

t_test_result = welch(data, "Comb_FE", "is_turbocharged")

# Linear model controlling for displacement
lm_model1 = smf.ols("Comb_FE ~ C(is_turbocharged) + Eng_Displ", data=data).fit()
print(lm_model1.summary())
```

```
Comb_FE by is_turbocharged: Naturally Aspirated (mean=26.209) vs Turbocharged (mean=22.778)
  t = 5.5283, df = 341.45, p = 6.434e-08
  95% CI of difference: [2.210, 4.652]

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                Comb_FE   R-squared:                       0.598
Model:                            OLS   Adj. R-squared:                  0.597
Method:                 Least Squares   F-statistic:                     643.6
Date:                Sun, 11 Oct 2026   Prob (F-statistic):          6.16e-172
Time:                        16:46:39   Log-Likelihood:                -2524.1
No. Observations:                 868   AIC:                             5054.
Df Residuals:                     865   BIC:                             5068.
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
======================================================================================================
                                         coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------------
Intercept                             42.3941      0.541     78.302      0.000      41.331      43.457
C(is_turbocharged)[T.Turbocharged]    -7.0627      0.339    -20.845      0.000      -7.728      -6.398
Eng_Displ                             -4.6450      0.136    -34.256      0.000      -4.911      -4.379
==============================================================================
```

The t-test showed a statistically significant difference (p = 6.43e-08) with naturally aspirated engines averaging 3.43 MPG higher than turbocharged engines.

The linear model revealed that after controlling for engine displacement, turbocharged engines actually showed 7.06 MPG lower fuel economy than naturally aspirated engines. Engine displacement had a strong negative effect (-4.65 MPG per liter). The model explained 59.8% of the variance in fuel economy.

## Hypothesis 2: Premium vs Regular fuel

```python
premium_data = data[data["fuel_type"] == "Premium"]
regular_data = data[data["fuel_type"] == "Regular"]

for name, d in [("Premium", premium_data), ("Regular", regular_data)]:
    print(f"{name}: n={len(d)}, Comb={d['Comb_FE'].mean():.3f}, "
          f"City={d['City_FE'].mean():.3f}, Hwy={d['Hwy_FE'].mean():.3f}")
```

```
Premium: n=433, Comb=21.785, City=19.319, Hwy=25.965
Regular: n=435, Comb=25.991, City=24.179, Hwy=28.885
```

The dataset was nearly balanced with 433 premium fuel vehicles (21.8 MPG combined) and 435 regular fuel vehicles (26.0 MPG combined). Regular fuel vehicles showed advantages across all three fuel economy measures.

### Plot 2: Fuel type

```python
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

def fuel_box(ax, col, title, ylabel):
    bp = ax.boxplot([premium_data[col].dropna(), regular_data[col].dropna()],
                    tick_labels=["Premium", "Regular"], patch_artist=True)
    for patch, c in zip(bp["boxes"], ["lightcoral", "lightyellow"]):
        patch.set_facecolor(c)
    ax.set(title=title, xlabel="Fuel Type", ylabel=ylabel)

fuel_box(axes[0, 0], "Comb_FE", "Combined MPG by Fuel Type", "Combined MPG")
fuel_box(axes[0, 1], "City_FE", "City MPG by Fuel Type", "City MPG")
fuel_box(axes[1, 0], "Hwy_FE", "Highway MPG by Fuel Type", "Highway MPG")

# Info panel
axes[1, 1].axis("off")
axes[1, 1].set_title("Fuel Type Comparison Summary")
axes[1, 1].text(0.5, 0.5, f"Premium: n = {len(premium_data)}\nRegular: n = {len(regular_data)}",
                ha="center", va="center")

fig.suptitle("Hypothesis 2: Premium vs Regular Fuel Vehicles")
plt.tight_layout()
plt.show()
```
<img width="1090" height="789" alt="plot2" src="https://github.com/user-attachments/assets/e12c0f0d-133c-4d18-b560-ccdd0a17a1d9" />


Hypothesis 2 Visualizations through the triple boxplot arrangement demonstrated consistent superiority of regular fuel vehicles across all three fuel economy measures. The nearly identical sample sizes (433 premium vs 435 regular) shown in the information panel strengthened the statistical validity of these comparisons. The consistent pattern across combined, city, and highway measures suggested a systematic difference rather than measurement artifact.

### Statistical Tests for Hypothesis 2

```python
t_comb = welch(data, "Comb_FE", "fuel_type")
t_city = welch(data, "City_FE", "fuel_type")
t_hwy = welch(data, "Hwy_FE", "fuel_type")
```

```
Comb_FE by fuel_type: Premium (mean=21.785) vs Regular (mean=25.991)
  t = -9.2908, df = 693.41, p = 1.943e-19
  95% CI of difference: [-5.094, -3.317]

City_FE by fuel_type: Premium (mean=19.319) vs Regular (mean=24.179)
  t = -10.2189, df = 635.83, p = 8.429e-23
  95% CI of difference: [-5.795, -3.927]

Hwy_FE by fuel_type: Premium (mean=25.965) vs Regular (mean=28.885)
  t = -6.5804, df = 773.60, p = 8.644e-11
  95% CI of difference: [-3.791, -2.049]
```

All three t-tests showed statistically significant differences (p < 0.001) with regular fuel vehicles demonstrating superior fuel economy across all measures: 4.21 MPG higher for combined, 4.86 MPG for city, and 2.92 MPG for highway driving.

## Correlation analysis

```python
cor_vars = data[["Comb_FE", "Eng_Displ", "City_FE", "Hwy_FE"]]
cor_matrix = cor_vars.corr()
cor_matrix
```

|           | Comb_FE  | Eng_Displ | City_FE  | Hwy_FE   |
|-----------|----------|-----------|----------|----------|
| Comb_FE   | 1.000000 | -0.629437 | 0.986073 | 0.959177 |
| Eng_Displ | -0.629437| 1.000000  | -0.587986| -0.645510|
| City_FE   | 0.986073 | -0.587986 | 1.000000 | 0.905128 |
| Hwy_FE    | 0.959177 | -0.645510 | 0.905128 | 1.000000 |

### Plot 3: Scatterplot matrix

```python
axes = pd.plotting.scatter_matrix(cor_vars, figsize=(9, 9), s=8, alpha=0.6, diagonal="hist")
plt.suptitle("Scatterplot Matrix of Key Variables")
plt.show()
```
<img width="761" height="846" alt="plot3" src="https://github.com/user-attachments/assets/bb5a051c-e08e-4a34-af60-4b6c0b5906d4" />


Correlation Matrix Visualization provided crucial validation of measurement consistency and revealed fundamental engineering relationships. The extremely high correlations between different fuel economy measures (r = 0.986 between combined and city MPG) confirmed internal consistency, while the clear negative relationship between engine displacement and all efficiency measures visually confirmed the fundamental physics of automotive efficiency.

## Two-way ANOVA: fuel type × aspiration type

```python
anova_model = smf.ols("Comb_FE ~ C(fuel_type) * C(is_turbocharged)", data=data).fit()
anova_lm(anova_model, typ=1)  # sequential SS, matches R's aov()
```

|                                 | df    | sum_sq       | mean_sq     | F         | PR(>F)       |
|---------------------------------|-------|--------------|-------------|-----------|--------------|
| C(fuel_type)                    | 1.0   | 3838.047092  | 3838.047092 | 92.735516 | 6.421179e-21 |
| C(is_turbocharged)              | 1.0   | 698.490023   | 698.490023  | 16.877029 | 4.367368e-05 |
| C(fuel_type):C(is_turbocharged) | 1.0   | 2138.109358  | 2138.109358 | 51.661345 | 1.426571e-12 |
| Residual                        | 864.0 | 35758.389242 | 41.387025   | NaN       | NaN          |

The two-way ANOVA revealed significant main effects for both fuel type (F = 92.74, p = 6.42e-21) and aspiration type (F = 16.88, p = 4.37e-05), plus a significant interaction effect (F = 51.66, p = 1.43e-12), indicating the relationship between fuel type and MPG differs depending on whether the engine is turbocharged.

## Displacement category analysis

```python
data["displ_category"] = pd.cut(
    data["Eng_Displ"],
    bins=[0, 2, 3, 4, data["Eng_Displ"].max()],
    labels=["Small (<2L)", "Medium (2-3L)", "Large (3-4L)", "Very Large (>4L)"])

displ_summary = (data.groupby(["displ_category", "is_turbocharged"], observed=True)["Comb_FE"]
                     .agg(mean="mean", count="count")
                     .reset_index())
displ_summary
```

|   | displ_category   | is_turbocharged     | mean      | count |
|---|------------------|---------------------|-----------|-------|
| 0 | Small (<2L)      | Naturally Aspirated | 37.629032 | 62    |
| 1 | Small (<2L)      | Turbocharged        | 26.735849 | 212   |
| 2 | Medium (2-3L)    | Naturally Aspirated | 31.153846 | 78    |
| 3 | Medium (2-3L)    | Turbocharged        | 21.714286 | 259   |
| 4 | Large (3-4L)     | Naturally Aspirated | 21.238095 | 63    |
| 5 | Large (3-4L)     | Turbocharged        | 18.380952 | 84    |
| 6 | Very Large (>4L) | Naturally Aspirated | 16.329114 | 79    |
| 7 | Very Large (>4L) | Turbocharged        | 16.516129 | 31    |

### Plot 4: MPG by displacement and aspiration type

```python
labels = displ_summary["displ_category"].astype(str) + "\n" + displ_summary["is_turbocharged"]
bar_colors = ["lightblue", "lightgreen"] * (len(displ_summary) // 2 + 1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(len(displ_summary)), displ_summary["mean"],
       color=bar_colors[:len(displ_summary)], edgecolor="black")
ax.set_xticks(range(len(displ_summary)))
ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set(title="Average MPG by Displacement and Aspiration Type", ylabel="Combined MPG",
       ylim=(0, displ_summary["mean"].max() * 1.1))
plt.tight_layout()
plt.show()
```
<img width="990" height="590" alt="plot4" src="https://github.com/user-attachments/assets/dc857910-a662-4b7d-8846-b515990f9449" />


Displacement Category Analysis through the bar plot showed a nuanced nature of the turbocharging effect. The visualization showed that the natural aspiration advantage was most dramatic in smaller engine categories (37.6 vs 26.7 MPG in <2L category) and diminished in larger categories, suggesting that turbocharging technology might be more beneficial for larger engines where the efficiency penalty is reduced.

## Final comprehensive model

```python
final_model = smf.ols("Comb_FE ~ C(fuel_type) + C(is_turbocharged) + Eng_Displ + C(drive_type)",
                      data=data).fit()
print(final_model.summary())
```

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                Comb_FE   R-squared:                       0.634
Model:                            OLS   Adj. R-squared:                  0.631
Method:                 Least Squares   F-statistic:                     248.2
Date:                Sun, 11 Oct 2026   Prob (F-statistic):          6.90e-184
Time:                        16:46:41   Log-Likelihood:                -2483.9
No. Observations:                 868   AIC:                             4982.
Df Residuals:                     861   BIC:                             5015.
Df Model:                           6                                         
Covariance Type:            nonrobust                                         
======================================================================================================
                                         coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------------
Intercept                             40.2004      0.732     54.888      0.000      38.763      41.638
C(fuel_type)[T.Regular]               -0.1636      0.358     -0.457      0.648      -0.867       0.539
C(is_turbocharged)[T.Turbocharged]    -6.3004      0.380    -16.594      0.000      -7.046      -5.555
C(drive_type)[T.FWD]                   2.8796      0.450      6.400      0.000       1.996       3.763
C(drive_type)[T.Other]                -1.6986      0.422     -4.020      0.000      -2.528      -0.869
C(drive_type)[T.RWD]                  -0.6539      0.409     -1.600      0.110      -1.456       0.148
Eng_Displ                             -4.0573      0.157    -25.810      0.000      -4.366      -3.749
==============================================================================
```

Insights from final comprehensive model:

- Turbocharged engines showed 6.30 MPG lower fuel economy after controlling for other factors
- Engine displacement remained strongly negative (-4.06 MPG per liter)
- Front-wheel drive vehicles showed 2.88 MPG advantage over all-wheel drive
- Fuel type became non-significant (p = 0.648) when controlling for other vehicle characteristics
- The model explained 63.4% of the variance in fuel economy

### Plot 5: Model diagnostics

```python
resid = final_model.resid
fitted = final_model.fittedvalues
infl = final_model.get_influence()
std_resid = infl.resid_studentized_internal   # equivalent to R's rstandard()
leverage = infl.hat_matrix_diag

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# Residuals vs Fitted
ax = axes[0, 0]
ax.scatter(fitted, resid, s=12, color="blue")
ax.axhline(0, color="red", ls="--")
sm_line = lowess(resid, fitted)
ax.plot(sm_line[:, 0], sm_line[:, 1], color="darkgreen", lw=2)
ax.set(title="Residuals vs Fitted Values", xlabel="Fitted Values", ylabel="Residuals")

# Normal Q-Q
ax = axes[0, 1]
(osm, osr), _ = stats.probplot(std_resid, dist="norm")
ax.scatter(osm, osr, s=12, color="blue")
x1, x2 = stats.norm.ppf([0.25, 0.75])
y1, y2 = np.quantile(std_resid, [0.25, 0.75])
slope = (y2 - y1) / (x2 - x1)
ax.plot(osm, y1 + slope * (osm - x1), color="red", lw=2)   # same as R's qqline()
ax.set(title="Normal Q-Q Plot", xlabel="Theoretical Quantiles", ylabel="Standardized Residuals")

# Scale-Location
ax = axes[1, 0]
sqrt_abs = np.sqrt(np.abs(std_resid))
ax.scatter(fitted, sqrt_abs, s=12, color="blue")
sm_line = lowess(sqrt_abs, fitted)
ax.plot(sm_line[:, 0], sm_line[:, 1], color="darkgreen", lw=2)
ax.set(title="Scale-Location Plot", xlabel="Fitted Values",
       ylabel="sqrt(|Standardized Residuals|)")

# Residuals vs Leverage
ax = axes[1, 1]
ax.scatter(leverage, std_resid, s=12, color="blue")
ax.axhline(0, color="red", ls="--")
ax.set(title="Residuals vs Leverage", xlabel="Leverage", ylabel="Standardized Residuals")

fig.suptitle("Final Model Diagnostic Plots")
plt.tight_layout()
plt.show()
```
<img width="1090" height="789" alt="plot5" src="https://github.com/user-attachments/assets/29a7b2d8-ccb1-4056-b556-c415662af1aa" />


Model Diagnostic Plots provided essential validation of the regression assumptions. The relatively random scatter in the residuals vs fitted plot, the approximately linear pattern in the Q-Q plot, and the absence of extreme leverage points all supported the validity of the multivariate conclusions.

## Key comparative metrics

```python
mean_turbo = turbo_data["Comb_FE"].mean()
mean_na = na_data["Comb_FE"].mean()
mean_premium = premium_data["Comb_FE"].mean()
mean_regular = regular_data["Comb_FE"].mean()

print(f"Mean MPG - Turbocharged:        {mean_turbo:.3f}")
print(f"Mean MPG - Naturally Aspirated: {mean_na:.3f}")
print(f"Mean MPG - Premium:             {mean_premium:.3f}")
print(f"Mean MPG - Regular:             {mean_regular:.3f}")
print(f"Turbo - NA difference:          {mean_turbo - mean_na:.3f}")
print(f"Premium - Regular difference:   {mean_premium - mean_regular:.3f}")
print(f"p-value (turbo vs NA):          {t_test_result.pvalue:.4g}")
print(f"p-value (premium vs regular):   {t_comb.pvalue:.4g}")
print(f"Correlation (displacement, MPG): {data['Eng_Displ'].corr(data['Comb_FE']):.3f}")
```

```
Mean MPG - Turbocharged:        22.778
Mean MPG - Naturally Aspirated: 26.209
Mean MPG - Premium:             21.785
Mean MPG - Regular:             25.991
Turbo - NA difference:          -3.431
Premium - Regular difference:   -4.206
p-value (turbo vs NA):          6.434e-08
p-value (premium vs regular):   1.943e-19
Correlation (displacement, MPG): -0.629
```

The metrics quantify the central findings: naturally aspirated engines achieve 3.43 MPG higher fuel economy than turbocharged engines (p = 6.43e-08), regular fuel vehicles show 4.21 MPG advantage over premium fuel vehicles in bivariate analysis (p = 1.94e-19), and engine displacement shows a strong negative correlation with fuel economy (r = -0.629).

## Research Abstract and Conclusion

This comprehensive analysis of 2025 model year vehicle fuel economy yields definitive conclusions regarding the two research hypotheses. For Hypothesis 1, we must reject the alternative hypothesis that turbocharged engines provide better fuel economy. The evidence strongly supports the opposite conclusion: naturally aspirated engines demonstrate significantly higher fuel economy both in simple comparisons (26.21 vs 22.78 MPG, p = 6.43e-08) and after controlling for engine displacement (7.06 MPG disadvantage for turbocharged engines, p < 2e-16). The visualization analysis revealed that this relationship is most pronounced in smaller engine categories, suggesting that the efficiency benefits of natural aspiration are particularly valuable in compact vehicle applications.

For Hypothesis 2, we find mixed evidence. While initial bivariate comparisons show substantial differences favoring regular fuel vehicles (25.99 vs 21.79 MPG, p < 1.94e-19), these differences become statistically non-significant in multivariate models that control for engine displacement, turbocharging, and drive type (p = 0.648). This pattern suggests that the apparent fuel type effect is largely confounded by other vehicle characteristics, particularly that premium fuel requirements are associated with vehicles that have other characteristics (larger engines, turbocharging) that independently reduce fuel economy.

The strong negative correlation between engine displacement and fuel economy (r = -0.629) remains a fundamental relationship in the 2025 vehicle fleet, while drive type emerges as a significant factor with front-wheel drive vehicles showing approximately 2.9 MPG advantage over all-wheel drive configurations. The comprehensive visualization approach proved essential for understanding the complex relationships and provided context that pure numerical analysis could not capture.

These findings have important implications for both automotive engineering and consumer decision-making, suggesting that the pursuit of turbocharging for fuel efficiency goals may be misguided, and that fuel type requirements should be considered in the broader context of overall vehicle design rather than as independent efficiency factors.

## References

U.S. Department of Energy, & U.S. Environmental Protection Agency. (2025). *2025 Fuel Economy Data* [Data set]. FuelEconomy.gov. https://www.fueleconomy.gov/feg/download.shtml
