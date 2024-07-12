# Model Card

## Model Details
The model is constituted of a SKLearn pipeline. The first part of the pipeline applies one hot encording on categorical columns, and standard scaling on numerical columns.  
Then, a logistic regressor infers salaries based on the values.

## Intended Use
The model is intended to infer the salary range (either <=50K or >50K) of a given person based on sensus data.

## Preprocessing
The original data used a comma followed by a space as separator, rather than a single comma like a typical csv file. The sep option of `pd.read_csv` has been set to ', ' to properly load the csv.  
Furthermore, there were a few duplicate rows and they were dropped.  
There were 32,537 rows left after deduplication.

## Training Data
The training data was created by randomly choosing 80% of the 32,537 rows. The random state was set to 42 for reproducibility.

## Evaluation Data
The evaluation data corresponds to the other 20% which was not used for training.

## Metrics
  - f1\_score: 0.679
  - precision\_score: 0.744
  - recall\_score: 0.625

## Ethical Considerations
This model outputs unbalanced data for some races. For example, for persons of White race, the scores are as such:
  - f1\_score: 0.683
  - precision\_score: 0.746
  - recall\_score: 0.629
But for people of Amer-Indian/Eskimo race:
  - f1\_score: 0.4,
  - precision\_score: 1.0,
  - recall\_score: 0.25
We can see that for poeple of Amer-Indian/Eskimo race, there was no false positive, but many false negatives. On the other hand, the scores are more balanced for people of White race.  
The model performs very uequally depending on the race.

## Caveats and Recommendations
Ideally, larger samples of the less represented races should be included in the data to avoid distortions. In this sample, 85% of the people are of White race and only 0.96% of AMer-Indian/Eskimo
