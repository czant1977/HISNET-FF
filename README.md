# HISNIFF-NET
1. Get a skull dataset with labeled information (51 categories).
2. Use concat.py to cut the labeled skull into teeth and get a dataset with teeth and skull corresponding to each other.
2. Use the split51-18.py file to divide the skull and teeth datasets into 18 categories.
3. Use the split18-2level.py file to divide the skull and teeth datasets into hierarchical network data formats.
4. Use argument18.py and argument2level.py to perform data enhancement (skull and teeth) on the 18-genus classification dataset and the hierarchical network dataset respectively.
5. Use the EB7_Train.py file to train the skull and teeth 18-genus classification datasets respectively.
6. Use the HIS-EB7_Train.py file to train the skull and teeth hierarchical network datasets respectively, that is, the classification training of polymorphic genera.
7. Use the Extract_Featrue.py file to extract skull and tooth features respectively using the previously trained model.
8. Use Feature_Fusion.py file to fuse the skull and teeth features.
9. Use MLP_Train.py to train the fusion features.
10. HISNIFF-Test.py is used for the final species identification. Input the image addresses of the skull and teeth respectively to output the identification results.

# the steps for using the programs
Step 1: Build a skull image dataset and a tooth image dataset containing 51 species;

Step 2: Use the split51-18.py file to convert the dataset into an 18-category format, and change the test_path and train_path parameters in it to the addresses of the train folder and test folder of the 51-category dataset;

Step 3: Use split18-2level.py to convert the 18-category dataset into a hierarchical classification dataset, and set the path parameter to the address of the 18-category dataset. At this point, the dataset construction is complete. The skull and tooth datasets both contain a 51-category, an 18-category, and a hierarchical classification dataset;

Step 4: For data enhancement, use argument18.py and argument2level.py to enhance the 18-category dataset and the hierarchical network dataset respectively, and obtain the enhanced 18-category dataset. Class and hierarchical classification data sets;

Step 5: Use EB7_Train.py and HIS-EB7_Train.py to train 18 classification and hierarchical classification data sets (same for skull and teeth) respectively, and get the model file of EB7;

Step 6: Use Extract_Feature.py to extract skull and tooth features respectively, use the existing model files to extract, get the feature sequences of skull and teeth respectively and save them;

Step 7: Use Feature_Fusion.py file to fuse the extracted features to get the fusion features;

Step 8: Use MLP_Train.py to train the fusion features and get the model of fusion features;

Step 9: Use HISNIFF-Test.py for identification, set the head_path and teeth_path addresses to the corresponding skull and tooth pictures (single), and get the identification results.
