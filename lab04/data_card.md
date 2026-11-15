# Data Card: UCI Adult (OpenML)

## Source and size

- **Source:** UCI Adult dataset distributed through OpenML (`adult`, version 2).
- **Size:** 48,842 rows and 14 input features, plus the income target.
- **Task:** Predict whether annual income is greater than $50K.

## License and use

The dataset is publicly accessible through UCI/OpenML. This run does not independently verify a single governing license identifier; consult the current UCI/OpenML metadata for the applicable public-domain or Creative Commons terms before redistribution.

## Known limitations and bias

The data reflects 1994 US Census records. Demographic imbalance by sex and race, and historical income and occupation biases, are present in the source data. Model performance should not be interpreted as a causal or fair assessment of people or groups.

## Privacy considerations

The features include sensitive or potentially identifying attributes, including race, sex, native-country, and marital-status. Use appropriate access controls, minimize unnecessary retention, and avoid using predictions for high-impact decisions without additional governance.
