# surface_classification
Just run the python files, they will automatically upload the corresponding config in the same folder.

Both surface_classifier and surface_classifier_v2 use range profile. v2 just plots more statistic parameters. (not very useful anyways)

grass_no_grass does a simple grass classification with cos similarity, tested and works ok

grass_no_grass_v2 has a tunable height tolerance (HEIGHT_TOLERANCE_CM), and it tries to shift the range bins within the height tolerance to find highest cos similarity score. Hopefully this can make detection more robust under slight variations of sensor height.
