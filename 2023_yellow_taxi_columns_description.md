# 2023 Yellow Taxi Trip Data — Column Descriptions

**Source:** NYC TLC, dataset `4b4i-vvec` on NYC Open Data

| Column | Type | Description |
|---|---|---|
| `VendorID` | int | TPEP provider. 1 = Creative Mobile Technologies, 2 = VeriFone Inc. |
| `tpep_pickup_datetime` | datetime | Meter engaged timestamp |
| `tpep_dropoff_datetime` | datetime | Meter disengaged timestamp |
| `passenger_count` | int | Driver-reported passenger count |
| `trip_distance` | float | Trip distance in miles (from taximeter) |
| `RatecodeID` | int | Rate code. 1=Standard, 2=JFK, 3=Newark, 4=Nassau/Westchester, 5=Negotiated, 6=Group ride |
| `store_and_fwd_flag` | string | Y/N — whether trip record was held in vehicle memory before sending |
| `PULocationID` | int | TLC taxi zone ID of pickup |
| `DOLocationID` | int | TLC taxi zone ID of dropoff |
| `payment_type` | int | 1=Credit card, 2=Cash, 3=No charge, 4=Dispute, 5=Unknown, 6=Voided |
| `fare_amount` | float | Time-and-distance fare from meter |
| `extra` | float | Misc. extras/surcharges (rush hour $1, overnight $0.50, etc.) |
| `mta_tax` | float | $0.50 MTA tax triggered automatically |
| `tip_amount` | float | Tip amount (auto-populated for credit card; cash tips not captured) |
| `tolls_amount` | float | Total tolls paid |
| `improvement_surcharge` | float | $0.30 surcharge at trip flag-drop |
| `total_amount` | float | Total charged to passenger (excludes cash tips) |
| `congestion_surcharge` | float | $2.50 for trips in Manhattan congestion zone |
| `airport_fee` | float | $1.25 pickup fee at LaGuardia/JFK |
