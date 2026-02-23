# Effect of Physicochemical Parameters on Physical Fitness Components of Female Adolescents Around Qua Iboe Estuary Using Machine Learning

> **Authors:** Yusuf Oluwaseyi Olayinka, Ford Dwedor, Nkwocha Chukwuemeka Reginald, Lawal Ayoade  
> **📓 Full Code Notebook:** [View on Google Colab](https://colab.research.google.com/drive/1v1lPMSEFN_CFdoeSyFgJqW8T2ovvPSnF?usp=sharing)

---

## Table of Contents

1. [Title](#title)
2. [Project Overview / Problem Statement](#project-overview--problem-statement)
3. [Data Sources](#data-sources)
4. [Tools](#tools)
5. [Data Cleaning](#data-cleaning)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [Machine Learning Analysis](#machine-learning-analysis)
8. [Results](#results)
9. [Recommendations](#recommendations)
10. [Limitations](#limitations)

---

## Title

**Effect of Physicochemical Parameters on Physical Fitness Components of Female Adolescents Around Qua Iboe Estuary Using Machine Learning**

This study investigates how environmental pollution — measured through physicochemical parameters of air and water — affects the physical fitness of female adolescents residing near the Qua Iboe Estuary in Akwa Ibom State, Nigeria. Machine learning techniques were applied to predict and compare fitness outcomes between adolescents in ecologically degraded and relatively clean environments.

---

## Project Overview / Problem Statement

The intersection of environmental health and physical wellbeing has attracted increasing scholarly attention, particularly among vulnerable populations such as adolescents. The Qua Iboe Estuary — a region characterized by offshore and onshore oil activities, fishing, and urbanization — presents a compelling case study. Heavy industrial activity has significantly altered the physicochemical profile of air and water in the region, raising serious public health concerns.

**The core research questions are:**

- How do physicochemical parameters (carbonate ions, chloride ions, carbon monoxide, suspended particulate matter, and heavy metals like chromium, lead, and cadmium) at the Qua Iboe Estuary compare to those in a control environment?
- Does exposure to environmental pollution measurably affect the physical fitness components (cardiovascular endurance, muscular strength, muscular endurance, flexibility, and BMI) of female adolescents?
- Can machine learning models effectively classify and predict fitness outcomes based on group membership (Estuary vs. Non-Estuary)?

**Why this matters:** Female adolescents represent a demographic of concern due to their susceptibility to environmental stressors, hormonal fluctuations, and evolving physical fitness capacities. Physical fitness components are critical indicators of adolescent health and are known to be influenced by environmental conditions. Despite this, there remains a gap in empirical research linking localized environmental parameters to physical fitness indicators — especially among young females in ecologically sensitive regions like the Qua Iboe Estuary.

---

## Data Sources

Two parallel data streams were collected simultaneously from two carefully selected sites:

### Environmental Data
Air and water samples were collected using standardized analytical methods:

- **Air samples** were gathered via high-volume air samplers at 12 designated points within both the Estuary and control sites. Filter papers pre-treated with sodium hydroxide were used, exposed for a 5-day duration, then weighed and analyzed for pollutants.
- **Water samples** were collected using a Grab sampling method with polyethylene basins and bottles to avoid contamination.

**Analytical methods used:**
- Colorimetry
- Titrimetry
- Spectrophotometry
- Gravimetry

### Human Subject Data
Physical fitness data was collected from **160 female adolescents aged 12–17 years**, divided into two groups:

| Group | Location | n |
|---|---|---|
| **Estuary Group** | Ibeno (Qua Iboe Estuary) | 80 |
| **Non-Estuary Group** | Uyo (Control Site, outside the Estuary) | 80 |

Participants were matched for age, socio-economic background, and school enrollment to minimize confounding factors.

**Physical fitness tests performed:**

| Test | Fitness Component Measured |
|---|---|
| 20-Meter Shuttle Run Test | Cardiovascular Endurance (VO2Max) |
| Standing Long Jump Test | Muscular Strength (meters) |
| Push-Up Test | Muscular Endurance |
| Fingertip-to-Floor Test | Flexibility |
| BMI Calculation | Body Composition |

**Source files used in code:**
- `Estuary group.xlsx` — fitness data for the Estuary group
- `Non estuary group.xlsx` — fitness data for the Non-Estuary group

---

## Tools

| Tool / Library | Purpose |
|---|---|
| **Python 3** | Primary programming language |
| **Pandas** | Data loading, manipulation, and tabular analysis |
| **NumPy** | Numerical computations |
| **Matplotlib** | Data visualization (histograms, bar charts) |
| **Seaborn** | Enhanced statistical plots (KDE histograms) |
| **SciPy (`ttest_ind`)** | Independent samples t-test for statistical comparison |
| **IPython Display** | Rendering results tables in Jupyter/Colab |
| **Google Colab** | Cloud-based notebook execution environment |
| **Microsoft Excel / openpyxl** | Input data format (`.xlsx`) and output results export |

---

## Data Cleaning

Data preparation was carried out systematically to ensure quality and consistency before analysis. The following steps were taken:

### 1. Data Loading
Both Excel files (`Estuary group.xlsx` and `Non estuary group.xlsx`) were loaded separately using `pd.read_excel()`.

```python
estuary_data = pd.read_excel('/content/Estuary group.xlsx')
non_estuary_data = pd.read_excel('/content/Non estuary group.xlsx')
```

### 2. Group Labeling
A `Group` column was added to each dataset before merging, to clearly distinguish participants' environmental group throughout the analysis:

```python
estuary_data['Group'] = 'Estuary'
non_estuary_data['Group'] = 'Non-Estuary'
```

### 3. Dataset Merging
Both datasets were concatenated into a single unified DataFrame for joint analysis:

```python
data = pd.concat([estuary_data, non_estuary_data], ignore_index=True)
```

### 4. Column Name Standardization
Column headers were stripped of any leading or trailing whitespace to prevent lookup errors:

```python
data.columns = data.columns.str.strip()
```

### 5. Feature Definition
The following five features were explicitly defined as the target analysis columns:

```python
features = [
    'Cardiovascular_Endurance(VO2Max: Ml/kg/Min)',
    'Muscular_Strenght(Meters)',
    'Body_Composition(BMI)',
    'Muscular_Endurance(Push-Up Test)',
    'Flexibility'
]
```

### 6. Handling Mixed Data Types
`Flexibility` was identified as a **categorical** feature with values such as `Pass`, `Average`, `Fair`, and `Poor`. The code treated this differently from numerical features — skipping t-tests and mean calculations for flexibility, and instead using value counts for group comparison.

### 7. HFZ (Healthy Fitness Zone) Standards Encoding
Age-stratified HFZ benchmark thresholds were encoded in a dictionary structure and used as reference standards against which both groups were evaluated. This enabled comparison of each group's mean performance against established health standards.

---

## Exploratory Data Analysis

### Overview of Groups
The dataset contains 80 Estuary and 80 Non-Estuary female adolescents, observed across 5 physical fitness metrics. Initial inspection was done using `data.head(100)` to verify merging and labeling.

### Distribution Visualization (Histograms with KDE)
For each numerical feature, the code generated overlapping histograms with KDE (Kernel Density Estimation) curves for both groups, overlaid with HFZ threshold lines:

```python
sns.histplot(group_data[feature], kde=True, label=group, bins=20, alpha=0.6)
plt.axvline(x=thresholds[0], color='red', linestyle='--', label=f'Lower HFZ ({thresholds[0]})')
plt.axvline(x=thresholds[1], color='green', linestyle='--', label=f'Upper HFZ ({thresholds[1]})')
```

**Insight:** These plots revealed how each group's performance distribution aligns with or deviates from HFZ healthy standards — visually distinguishing the two groups' performance profiles.

### Bar Chart Comparison (Binned Data)
A second round of plots used binned bar charts to compare frequency distributions of each feature across both groups, with HFZ threshold lines added as vertical references.

### Age-Wise Implementation
A separate section of the analysis (`AGE WISE IMPLEMENTATION`) re-ran the HFZ comparisons with updated feature names precisely matching the merged dataset's column headers, enabling more accurate age-stratified HFZ lookups — since HFZ standards vary by age (10–18 years).

### Flexibility Category Distribution
Since flexibility is categorical, group comparisons were done via value counts:

- **Estuary Group:** Pass: 56 | Average: 15 | Fair: 4 | Poor: 4 | poor (typo): 1
- **Non-Estuary Group:** Pass: 52 | Average: 14 | Fair: 14

This revealed the Estuary group had slightly more participants achieving a "Pass" in flexibility, while the Non-Estuary group had a higher proportion in the "Fair" category.

---

## Machine Learning Analysis

### Approach
The analysis employed a **supervised comparative classification and statistical inference** approach, combining machine learning with classical hypothesis testing. The core analytical pipeline includes:

### Step 1 — Feature Engineering & HFZ Benchmarking
For each participant and each numerical feature, the code compared group means to age-specific HFZ thresholds, producing Boolean flags (`Within HFZ: True/False`) as derived features.

```python
thresholds = hfz_standards[feature].get(age_col, None)
feature_results[f"{group} Within HFZ"] = thresholds[0] <= group_mean <= thresholds[1]
```

### Step 2 — Independent Samples T-Test
For all numerical fitness features, an independent samples t-test was applied to determine whether observed differences between the Estuary and Non-Estuary groups were statistically significant:

```python
from scipy.stats import ttest_ind
t_stat, p_val = ttest_ind(estuary_values, non_estuary_values)
```

The t-statistic (direction and magnitude) and p-value (significance at α = 0.05) were stored for each feature.

### Step 3 — Results Compilation
All feature-level statistics were collected into a structured results list of dictionaries, then converted into a Pandas DataFrame for tabular review and export:

```python
results_df = pd.DataFrame(results)
results_df.to_excel("results.xlsx", index=False)
```

### Step 4 — Categorical Handling for Flexibility
Flexibility was cleanly separated from numerical processing — mean and t-test fields were set to `None` for this feature, while count-based distributions were computed and retained.

### HFZ Standards Applied (Age 12–17)

| Feature | Metric | HFZ Threshold (example, age 15) |
|---|---|---|
| Cardiovascular Endurance | VO2Max (ml/kg/min) | 36.1 – 39.0 |
| Body Composition | BMI | 24.4 – 29.2 |
| Muscular Endurance | Push-Ups (reps) | ≥ 7 |
| Muscular Strength | Standing Long Jump (meters) | 1.73 – 1.85 |
| Flexibility | Fingertip-to-Floor | Pass = Hands Reach Floor |

---

## Results

### Summary Statistics and Statistical Test Outcomes

| Feature | Estuary Mean | Non-Estuary Mean | t-statistic | p-value | Significant? |
|---|---|---|---|---|---|
| Cardiovascular Endurance (VO2Max) | Lower | Higher | Negative | < 0.05 | ✅ Yes |
| Muscular Strength (meters) | 2.13 | 2.33 | -5.52 | 1.37e-07 | ✅ Yes |
| Muscular Endurance (Push-Ups) | 8.21 | 10.60 | -7.05 | 5.22e-11 | ✅ Yes |
| Flexibility | 56 Pass | 52 Pass | N/A (categorical) | N/A | ➡ Estuary slightly better |
| Body Composition (BMI) | 22.64 | 22.75 | -0.24 | 0.81 | ❌ No |

### Key Findings

**Cardiovascular Endurance (VO2Max):** The Non-Estuary group significantly outperformed the Estuary group. The negative t-statistic and small p-value (< 0.05) confirm the Estuary group's inferior cardiovascular capacity — likely linked to chronic exposure to air pollutants like carbon monoxide and suspended particulate matter.

**Muscular Strength (Standing Long Jump):** The Non-Estuary group recorded a meaningfully higher mean (2.33m vs. 2.13m), with an extremely significant p-value of 1.37×10⁻⁷. This gap suggests that pollutant exposure or related lifestyle effects are adversely impacting the explosive strength of Estuary adolescents.

**Muscular Endurance (Push-Up Test):** The largest performance gap was observed here — Non-Estuary participants averaged 10.6 push-ups versus 8.21 for the Estuary group. The p-value of 5.22×10⁻¹¹ is among the most statistically significant results in the study, firmly establishing this as a critical area of concern.

**Flexibility:** This was the only fitness dimension where the Estuary group showed a comparable or marginally better performance, with 56 participants achieving a "Pass" versus 52 in the Non-Estuary group. However, the Non-Estuary group had more participants in the "Fair" category (14 vs. 4), suggesting the Non-Estuary group's flexibility profile is more dispersed.

**Body Composition (BMI):** Both groups had nearly identical mean BMI (22.64 vs. 22.75), with no statistically significant difference (p = 0.81). This suggests environmental pollution does not markedly affect body weight/composition in the short term for this population.

### Overall Pattern
The Non-Estuary group consistently outperformed the Estuary group across all numerical fitness metrics (VO2Max, Muscular Strength, Push-Ups). The Estuary group showed relative strength only in flexibility. These findings strongly suggest that chronic environmental degradation in the Qua Iboe Estuary negatively impacts the physical fitness capacity of female adolescents residing in the area.

---

## Recommendations

Based on the findings of this study, the following recommendations are made:

**1. Targeted Physical Fitness Interventions for Estuary Adolescents**
Tailored training programs should be developed specifically for adolescents in the Qua Iboe Estuary region, with emphasis on building cardiovascular endurance, muscular strength, and muscular endurance — the three fitness areas where the Estuary group showed the most significant deficits.

**2. Strict Environmental Regulation of Industrial Activities**
Government agencies and regulatory bodies should implement and enforce stringent measures to curtail unregulated industrial activities — particularly oil exploration and related operations — in and around the Qua Iboe Estuary. Elevated concentrations of CO₃²⁻, Cl⁻, CO, SPM, Cr, Pb, and Cd represent active pollution risks that must be addressed.

**3. Integration of Environmental Monitoring with Public Health Programs**
Public health policies should be redesigned to integrate environmental quality monitoring with adolescent health programming. Schools and health centers in the Estuary region should be equipped to routinely assess and respond to pollution-linked health impacts.

**4. Expanded Machine Learning and Health Research in the Region**
Continued interdisciplinary research combining environmental science, public health, and artificial intelligence is critical. Machine learning tools should be routinely used to track changing environmental conditions and predict health outcomes — enabling proactive rather than reactive interventions.

**5. Special Focus on Female Adolescent Health**
Since female adolescents are disproportionately underrepresented in environmental health research and are especially susceptible to environmental stressors due to hormonal and physiological development, dedicated health surveillance programs should be established for this group in ecologically sensitive regions.

**6. Community Awareness and Stakeholder Engagement**
Local communities, schools, and parents in the Estuary region should be educated on the link between environmental quality and physical health, empowering them to advocate for cleaner environments and healthier lifestyles.

---

## Limitations

This study, while comprehensive in scope, is subject to several limitations that should be considered when interpreting findings:

**1. Sample Size Constraints**
The study involved 80 participants per group (160 total), which, while adequate for initial comparative analysis, may limit the statistical power for detecting smaller effect sizes and may not fully represent all age sub-groups within the 12–17 year range.

**2. Cross-Sectional Design**
Data were collected at a single point in time. A longitudinal study tracking the same participants over time would better establish causal links between environmental exposure and physical fitness decline.

**3. Confounding Variables**
Although participants were matched for age, socio-economic status, and school enrollment, other confounding factors — such as dietary habits, physical activity levels outside school, sleep patterns, genetic predispositions, and access to healthcare — were not fully controlled for.

**4. Self-Reported and Proxy Data**
Some demographic information may have relied on self-report or school records, introducing potential for measurement inconsistency.

**5. Generalizability**
The findings are geographically specific to the Qua Iboe Estuary region and the Uyo control area. Extrapolating these results to other estuarine or polluted environments should be done cautiously.

**6. Absence of Individual-Level Environmental Exposure Data**
While environmental sampling was conducted at site level, individual-level exposure data (e.g., how many hours each participant spends outdoors, proximity to specific pollution sources) was not collected. This limits the ability to establish dose-response relationships.

**7. Categorical Flexibility Measurement**
The flexibility metric was measured categorically (Pass/Average/Fair/Poor) rather than as a continuous numerical variable, making it impossible to apply standard statistical tests (e.g., t-test) or include it meaningfully in regression or classification models.

**8. Machine Learning Depth**
The current analysis primarily employed statistical comparison (t-tests) and HFZ benchmarking. Deeper machine learning models (e.g., Random Forest, Logistic Regression, or Support Vector Machines) could be applied in future work for more robust predictive modeling and feature importance analysis.

---

## Citation

 Yusuf, O. O., Ford, D., Nkwocha, C. R., & Lawal, A. (2025). *Effect of Physicochemical Parameters on Physical Fitness Components of Female Adolescents Around Qua Iboe Estuary Using Machine Learning.* Research Group Publication.

---

## Links

| Resource | Link |
|---|---|
| 📓 Full Analysis Code (Google Colab) | [https://colab.research.google.com/drive/1v1lPMSEFN_CFdoeSyFgJqW8T2ovvPSnF?usp=sharing](https://colab.research.google.com/drive/1v1lPMSEFN_CFdoeSyFgJqW8T2ovvPSnF?usp=sharing) |

---


