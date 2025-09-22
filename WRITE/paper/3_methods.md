# METHODS

## Study design

This was a retrospective cohort study analyzing consecutive patients who underwent PCI with DCB for de novo coronary lesions at a single center. Data were extracted from a prospectively maintained institutional database containing detailed procedural and follow-up information. The study period encompassed patients treated between the initiation of the DCB program and December 2024. Patients were recruited from the CARDIO-FR registry(NCT04185285). This registry is conducted in accordance with the Declaration of Helsinki, has received approval from the local ethics committee (003-REP-CER-FR), and written informed consent was obtained from every participant prior to enrolment.
Inclusion criteria were any patients aged 18 years or older who underwent DCB treatment for de novo CAD with available baseline, post-procedural, and follow-up coronary angiography performed at least 7 days after the index procedure.
Patients were excluded if the treated lesion represented in-stent restenosis, if angiographic image quality was inadequate for quantitative coronary angiography (QCA) analysis, if follow-up angiography was performed within 7 days of the index procedure.

## Procedure description

Angiographic images were recorded at a minimum of 12.5 frames per second with a monoplane radiographic system (Azurion 3, Philips Healthcare, NL). Videos in which angiographic images acquisition has been realized to minimize vessel overlap and foreshortening were selected for µQFR assessment. The contrast medium was injected manually with a forceful and stable injection with ACIST CVi contrast injector (Bracco, I).

Offline QCA and µQFR assessments were performed with Angioplus Galley (Pulse Medical Imaging Technology, Shanghai, China) by two independent analysts who were blinded, mutually independent, and experienced in physiology‐derived imaging. 
For each lesion the analysts first selected the projection that optimally visualised the vessel from ostium to distal segment, minimising overlap and foreshortening. 
Operators were instructed to modify manual analysis as little as possible, with lumen contours refined only when automated delineation proved inadequate.
A second projection ≥25° apart of diagnostic quality were selected to generate a full 3D-QCA reconstruction, and the corresponding 3D-μQFR was calculated, only when both projections were available and of good quality.
The frame with the clearest anatomic definition of the lesion served as the reference for velocity computation and Murray law–based diameter scaling, which applies stepwise calibre changes at bifurcations. 
If there was a significant jump in the vessel flow, the fixed flow proposed by the software was used.
The software then automatically derived flow velocity profiles and calculated physiologic indices for the main vessel and any relevant side branches.

DCB procedures were performed according to standard institutional protocols. Lesion preparation was performed at operator discretion using conventional balloons, cutting balloons, scoring balloons, OPN balloons, or atherectomy devices including rotational atherectomy, orbital atherectomy, or intravascular lithotripsy. Following adequate lesion preparation, DCB inflation was performed using either paclitaxel-coated or sirolimus-coated balloons. DCB diameter was selected to achieve a balloon-to-vessel ratio between 1.0 and 1.2, with inflation pressures and duration recorded for each case. Intravascular imaging guidance was utilized based on operator preference and clinical indication.

## Study definitions

Minimal lumen diameter (MLD) was measured at standardized segments using quantitative coronary angiography. MLD late lumen enlargement was calculated as the difference between follow-up pre-intervention MLD and initial post-procedural MLD. MLD acute gain represented the immediate post-procedural MLD minus baseline MLD. MLD net gain was defined as follow-up MLD minus baseline MLD. Late recoil was calculated as the difference between maximal balloon diameter and follow-up MLD. Late functional change was defined as the difference between follow-up µQFR and post-procedural µQFR. Target lesion failure (TLF) comprised cardiac death, target vessel myocardial infarction, or clinically driven target lesion revascularization. Target vessel failure (TVF) included cardiac death, target vessel myocardial infarction, or any target vessel revascularization. All definitions followed the Drug-Coated Balloon Academic Research Consortium criteria [@fezziDefinitionsStandardizedEndpoints2025]. 

## Study endpoints

The primary endpoints included late functional change assessed by μFR and late lumen change measured by MLD. Secondary endpoints comprised individual components of lesion evolution including MLD acute gain, MLD net gain, acute and late recoil, μFR acute functional gain, μFR net functional gain across timepoints.

## Statistical analysis

Continuous variables were expressed as mean ± standard deviation or median (interquartile range), and categorical variables as frequencies and percentages. Paired t-tests or Wilcoxon signed-rank tests compared post-procedural and follow-up measurements based on normality assessment.
Univariate analyses employed linear regression for continuous outcomes and logistic regression for binary outcomes (gain versus loss), yielding coefficients or odds ratios with 95% confidence intervals. Variables with p<0.10 entered multivariate models, maintaining at least 10 events per variable for logistic regression. Multicollinearity was assessed using variance inflation factors. Model performance was evaluated using R-squared for linear models and area under the curve for logistic models.
Missing outcome data were excluded; continuous predictors were imputed using median values. Forest plots displayed significant predictors with 95% confidence intervals. All tests were two-sided with significance at p<0.05. Analyses were performed using Python 3.13.0 (NumPy, pandas, SciPy, statsmodels, scikit-learn) with PostgreSQL 14.0 database via SQLAlchemy.