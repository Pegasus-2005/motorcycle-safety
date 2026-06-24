> Rodegast, Philipp; Maier, Steffen; Kneifl, Jonas; Fehr, Jörg

# Simulation Data from Motorcycle Sensors in Operational and Crash Scenarios

Globally, motorcycles attract vast and varied users. However, since the rate of severe injury and fatality in motorcycle accidents far exceeds passenger car accidents, efforts have been directed toward increasing passive safety systems. Impact simulations show that the risk of severe injury or death in the event of a motorcycle-to-car impact can be greatly reduced if the motorcycle is equipped with passive safety measures such as airbags and seat belts. For the passive safety systems to be activated, a collision must be detected within milliseconds for a wide variety of impact configurations, but under no circumstances may it be falsely triggered. For the challenge of reliable detection, this paper presents an investigation into the applicability of machine learning algorithms. First, a series of simulations of accidents and driving operations are introduced to collect data to train machine learning classification models. Their performance is henceforth assessed and compared via multiple representative and application-oriented criteria.

See REQUIREMENTS for dependencies.

> Corresponding research article will be published soon

> If you refer to this data for scientific purposes, please cite the original research article.

## Content

1. Time trajectories of sensor signals for operational and crash scenarios (*.csv files): time-dependent sensor measurements resulting from a variety of simulated scenarios 
    * TrainingData.csv (~40.000 Samples): Scenarios used for training
    * TestData.csv (~9000 Samples): Scenarios used for testing
    * ControlScenarios (including 39 .csv files): Scenarios used for validation (including ISO 13232 scenarios) 
2. Script to load the data with data description (LoadData.py)
