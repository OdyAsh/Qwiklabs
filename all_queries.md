One of the projects you are working on needs to provide analysis based on real world data that will help in the selection of new bicycle models for public bike share systems. Your role in this project is to develop and evaluate machine learning models that can predict average trip durations for bike schemes using the public data from Austin's public bike share scheme to train and evaluate your models.

Two of the senior data scientists in your team have different theories on what factors are important in determining the duration of a bike share trip and you have been asked to prioritise these to start. The first data scientist maintains that the key factors are the start station, the location of the start station, the day of the week and the hour the trip started. While the second data scientist argues that this is an over complication and the key factors are simply start station, subscriber type, and the hour the trip started.

You have been asked to develop a machine learning model based on each of these input features. Given the fact that stay-at-home orders were in place for Austin during parts of 2020 as a result of COVID-19 you will be working on data from previous years. You have been instructed to train your models on data from 2018 and then evaluate them against data from 2019 on the basis of Mean Absolute Error and the square root of Mean Squared Error.

You can access the public data for the Austin bike share scheme in your project by opening this link to the Austin bike share dataset in the browser tab for your lab.

As a final step you must create and run a query that uses the model that includes subscriber type as a feature, to predict the average trip duration for all trips from the busiest bike sharing station in 2019 (based on the number of trips per station in 2019) where the subscriber type is 'Single Trip'.

# Task 1

Create the first machine learning model to predict the trip duration for bike trips. The features of this model must incorporate the starting station name, the hour the trip started, the weekday of the trip, and the address of the start station labeled as location. You must use 2018 data only to train this model.
## Solution (not working due to indentation problems with google cloud)
```sql
CREATE OR REPLACE MODEL bikes_test.location_model
OPTIONS (MODEL_TYPE = 'LINEAR_REG') #no need for INPUT_LABEL_COLS as duration_columns column will be set as 'label'
AS SELECT 
btrips.duration_minutes AS label,
btrips.start_station_name AS start_station_name,
EXTRACT (HOUR FROM btrips.start_time) AS hour,
EXTRACT (DAYOFWEEK FROM btrips.start_time) AS day_of_week,
bstations.address AS location
FROM 
`bigquery-public-data.austin_bikeshare.bikeshare_stations` AS bstations
JOIN
`bigquery-public-data.austin_bikeshare.bikeshare_trips` AS btrips
ON btrips.start_station_id = bstations.station_id 
AND EXTRACT (YEAR FROM btrips.start_time) = 2018 
AND btrips.duration_minutes > 0 #extract returns number not string, and final AND is to check that the row is valid
```
## Solution (Working)
```sql
CREATE OR REPLACE MODEL bikes_test.location_model

OPTIONS

  (model_type='linear_reg', labels=['duration_minutes']) AS

SELECT

    start_station_name,

    EXTRACT(HOUR FROM start_time) AS start_hour,

    EXTRACT(DAYOFWEEK FROM start_time) AS day_of_week,

    duration_minutes,

    address as location

FROM

    `bigquery-public-data.austin_bikeshare.bikeshare_trips` AS trips

JOIN

    `bigquery-public-data.austin_bikeshare.bikeshare_stations` AS stations

ON

    trips.start_station_name = stations.name

WHERE

    EXTRACT(YEAR FROM start_time) = 2018

    AND duration_minutes > 0
```

# Task 2

Create the second machine learning model to predict the trip duration for bike trips. The features of this model must incorporate the starting station name, the bike share subscriber type and the start time for the trip. You must also use 2018 data only to train this model.

## Solution (not working due to indentation problems with google cloud)
```sql
CREATE OR REPLACE MODEL bikes_test.subscriber_model
OPTIONS (MODEL_TYPE = 'LINEAR_REG') #no need for INPUT_LABEL_COLS as duration_columns column will be set as 'label'
AS SELECT 
btrips.duration_minutes AS label,
btrips.start_station_name AS start_station_name,
btrips.subscriber_type AS subscriber_type,
btrips.start_time AS start_time
FROM 
`bigquery-public-data.austin_bikeshare.bikeshare_trips` AS btrips
WHERE
EXTRACT (YEAR FROM btrips.start_time) = 2018 
AND btrips.duration_minutes > 0 #extract returns number not string, and final AND is to check that the label is valid
```

## Solution (Working)
```sql
CREATE OR REPLACE MODEL bikes_test.subscriber_model

OPTIONS

  (model_type='linear_reg', labels=['duration_minutes']) AS

SELECT

    start_station_name,

    EXTRACT(HOUR FROM start_time) AS start_hour,

    subscriber_type,

    duration_minutes

FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips` AS trips

WHERE EXTRACT(YEAR FROM start_time) = 2018
```

# Task 3
Evaluate each of the machine learning models against 2019 data only using separate queries. Your queries must report both the Mean Absolute Error and the Root Mean Square Error.

## Solution (Query 1) (working)
```sql
SELECT SQRT(mean_squared_error) AS rmse, mean_absolute_error
FROM ML.EVALUATE(MODEL bikes_test.location_model, 
(
    SELECT 
    btrips.duration_minutes AS duration_minutes,
    btrips.start_station_name AS start_station_name,
    EXTRACT (HOUR FROM btrips.start_time) AS start_hour,
    EXTRACT (DAYOFWEEK FROM btrips.start_time) AS day_of_week,
    bstations.address AS location
    FROM 
    `bigquery-public-data.austin_bikeshare.bikeshare_stations` AS bstations
    JOIN
    `bigquery-public-data.austin_bikeshare.bikeshare_trips` AS btrips
    ON btrips.start_station_id = bstations.station_id 
    AND EXTRACT (YEAR FROM btrips.start_time) = 2019 
    AND btrips.duration_minutes > 0 #extract returns number not string, and final AND is to check that the row is valid
))
```

## Solution (Query 2) (working)
```sql
SELECT SQRT(mean_squared_error) AS rmse, mean_absolute_error
FROM ML.EVALUATE(MODEL bikes_test.subscriber_model,
(
    SELECT 
    btrips.duration_minutes AS duration_minutes,
    btrips.start_station_name AS start_station_name,
    btrips.subscriber_type AS subscriber_type,
    EXTRACT(HOUR FROM btrips.start_time) AS start_hour,
    FROM 
    `bigquery-public-data.austin_bikeshare.bikeshare_trips` AS btrips
    WHERE
    EXTRACT (YEAR FROM btrips.start_time) = 2019
    AND btrips.duration_minutes > 0 #extract returns number not string, and final AND is to check that the label is valid
))
```

# Task 4
When both models have been created and evaulated, use the second model, that uses subscriber_type as a feature, to predict average trip length for trips from the busiest bike sharing station in 2019 where the subscriber type is Single Trip.

## Solution (Query to find busiest station) (Working)
```sql
#busiest station(21st & Speedway @PCL)
SELECT 
btrips.start_station_name, COUNT(*) AS num_of_people_in_station
FROM 
`bigquery-public-data.austin_bikeshare.bikeshare_trips` AS btrips
WHERE 
EXTRACT (YEAR FROM btrips.start_time) = 2019 
AND btrips.duration_minutes > 0 #extract returns number not string, and final AND is to check that the row is valid
GROUP BY btrips.start_station_name
ORDER BY num_of_people_in_station DESC
LIMIT 1
```
## Solution (Working)
```sql
SELECT  AVG(predicted_duration_minutes) AS average_predicted_trip_length
FROM ML.PREDICT(MODEL bikes_test.subscriber_model, 
(
    SELECT 
    btrips.start_station_name AS start_station_name,
    EXTRACT(HOUR FROM btrips.start_time) AS start_hour,
    btrips.subscriber_type AS subscriber_type,
    btrips.duration_minutes AS duration_minutes,
    FROM 
    `bigquery-public-data.austin_bikeshare.bikeshare_trips` AS btrips
    WHERE
    EXTRACT (YEAR FROM btrips.start_time) = 2019 
    AND btrips.duration_minutes > 0 #extract returns number not string, and final AND is to check that the label is valid
    AND btrips.subscriber_type = 'Single Trip'
    AND btrips.start_station_name = '21st & Speedway @PCL' #busiest_station is the query above, LIKE compares strings
))
```

#### Solution (Not working, as the LIKE should be true when start_station_name equals the first row only (hence the LIMIT 1 in the query above), but instead the LIMIT doesn't work and it returns all the station names for comparison with the LIKE operator)
```sql
WITH busiest_station AS #stores the output of the query below (21st & Speedway @PCL) in a 'variable' called busiest_station to be used later
(
    SELECT 
    btrips.start_station_name, COUNT(*) AS num_of_people_in_station
    FROM 
    `bigquery-public-data.austin_bikeshare.bikeshare_trips` AS btrips
    WHERE 
    EXTRACT (YEAR FROM btrips.start_time) = 2019 
    AND btrips.duration_minutes > 0 #extract returns number not string, and final AND is to check that the row is valid
    GROUP BY btrips.start_station_name
    ORDER BY num_of_people_in_station DESC
    LIMIT 1
)
SELECT  *
FROM ML.PREDICT(MODEL bikes_test.subscriber_model, 
(
    SELECT 
    btrips.start_station_name AS start_station_name,
    EXTRACT(HOUR FROM btrips.start_time) AS start_hour,
    btrips.subscriber_type AS subscriber_type,
    btrips.duration_minutes AS duration_minutes,
    FROM 
    `bigquery-public-data.austin_bikeshare.bikeshare_trips` AS btrips
    WHERE
    EXTRACT (YEAR FROM btrips.start_time) = 2019 
    AND btrips.duration_minutes > 0 #extract returns number not string, and final AND is to check that the label is valid
    AND btrips.subscriber_type = 'Single Trip'
    AND btrips.start_station_name LIKE (SELECT btrips.start_station_name FROM busiest_station) #busiest_station is the query above, LIKE compares strings
))
```
