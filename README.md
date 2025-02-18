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
