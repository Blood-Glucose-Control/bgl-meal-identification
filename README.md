# Semi-Supervised Change Point Detection of Meals from Continuous Glucose Monitor Time Series Data

## About Us

- [Blood Glucose Control AI Design Team (BGC)](https://blood-glucose-control.streamlit.app/) - the organization that built this repo.
- [WAT.ai](https://watai.ca/) - BGC was formed through WAT.ai
- [Gluroo Imaginations Inc](https://gluroo.com/) - Industry Partner
- [sktime](https://www.sktime.net/en/stable/) - Research Partner


## Main Project Deliverables

1. Publication of an open source simulated semi-supervised change point detection dataset and benchmark.
2. Evaluation of state-of-the-art change point detection algorithms on our benchmark.
3. Evaluation of state-of-the-art change point detection algorithms on a de-identified private cgm time series dataset provided by Gluroo.

### Description

The main goal of the project is to develop efficient machine learning pipelines for **automatic meal identification** from blood glucose readings in a semi-supervised setting.

This was chosen to be the first BGC project because we consider this a _stage 0_ task for a significant majority of diabetes machine learning and artificial intelligence modeling tasks.
There are few blood glucose modeling projects that do not directly depend or benefit from improved meal time labelling.
The benefit is derived from the fact that the most impactful factors on blood glucose dynamics are insulin and consumption of carbohydrates.
However, unlike most open source continuous glucose monitoring (cgm) datasets and research studies, having detailed annotations of meals is unrealistic in real-world settings.

In order to improve the speed and likelihood of transferring research to practice, we created a semi-supervised change point detection benchmark that can serve as a foundation for many future cgm studies.
Some of the foreseeable benefits from this research:

1. Improved feature engineer for downstream modeling tasks like blood glucose forecasting and causal modeling of blood glucose levels.
2. Improved understanding of prandial (meal-time) blood glucose dynamics.
3. Reduced cognitive burden for PWDs because these methods can be incorporated into streamlining data logging.

**_AI/ML Topics:_** _Feature Engineering, Time Series Annotation, Time Series Change Point Detection, Time Series Representation Learning, Semi-Supervised Learning, Benchmarking_
