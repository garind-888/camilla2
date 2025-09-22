# Figure Generation Prompt

Generate comprehensive figure content including title, description, and abbreviations based on the provided figure data or description.

## Output Format:

### **Figure Title**
- Concise, descriptive title (typically 5-15 words)
- Should clearly indicate what the figure shows
- Use sentence case (capitalize only first word and proper nouns)

### **Figure Description**
- Detailed explanation of what the figure shows
- ONLY descriptive, do not interpret the data.
- DONT rewrite statistical results of the figure, only describe the figure.


### **Figure Abbreviations**
- **Format**: `ABBREVIATION, full term; ABBREVIATION, full term; ...`
- **Alphabetical order** by abbreviation
- **Semicolon separation** between entries
- **Comma separation** between abbreviation and definition
- **Period** at the end.
- Include only abbreviations used in the figure title and description

## Example Output:
**Figure Title:** Comparison of primary endpoints between drug-coated balloon and drug-eluting stent groups

**Figure Description:** Primary endpoint results comparing DCB angioplasty versus DES implantation in patients with CAD. Error bars represent 95% CI.

CI, confidence interval; CAD, coronary artery disease; DCB, drug-coated balloon; DES, drug-eluting stent,

Generate figure content for the following:

---
