# A-HIOT-
A-HIOT stands for automated hit identification and optimization tool which comprise of the stacked ensemble, deep learning architectures and combines conventional approaches based on the chemical space (AI-dependent predictive model derived from standard ligand information for respective targets) and protein space (target structure information collection and artificial intelligence dependent model extracted from the interaction pattern of target protein-ligand complexes).





> ### A-HIOT

** 
A- HIOT stands for automated hit identification and
optimization tool which is an advanced virtual screening framework. It comprise of a deep learning architecture based on chemical space (CS) which is an artificial intelligence dependent prognostic model extracted from interaction pattern of the target protein ligand complexes. the A-HIOT is an approach for bridging long standing gap between the ligand based and the structured based virtual screening to find the number of optimized hit for desired receptor. 


### **Prerequisites**
the following are some tool and packages that are required:
* PaDEL-Descriptor (http://www.yapcwsoft.com/dd/padeldescriptor/)
* Open Babel – An open chemical toolbox (http://openbabel.org/wiki/Main_Page)
* Cavity (http://mdl.ipc.pku.edu.cn/mdlweb/download-cn.php)
* Pocket v3 (http://mdl.ipc.pku.edu.cn/mdlweb/download-cn.php)
* AutoDock Tools (http://mgltools.scripps.edu/downloads)
* AutoDock Vina (http://vina.scripps.edu/download.html)
* Protein Ligand Interaction Profiler (PLIP) (https://github.com/pharmai/plip)
* PyMOL (https://pymol.org/2/)
* R v3.6 or above (https://www.r-project.org/)
* H2o package in R for Artificial Intelligence (https://h2o-release.s3.amazonaws.com/h2o/rel-zipf/1/index.html)


2. The second requirement is to select specific or similar receptor structures that belong to the same family and well established profile inhibitors or modulators for the respective protein

### **Steps**

**_Establishing Chemical Space (CN)_**

**Step 1: ** Calculating the features and Preprocessing the Training dataset

1. Collect all the molecules and using Open Babel convert them into sdf format. 
````
 $sh mol_to_sdf.sh

And move all the sdf files into a single directory.

$ mkdir directioy_name (change 'directioy_name' as per convenience)
$ mv *.sdf directioy_name
````
** Step 2: ** Open PaDEL descriptor. Now select the molecules that contains directory and calculate 1D and 2D descriptors.
Name it descriptors.csv file. The file further require preprocessing to the data to overcome the curse of dimensionality. Therefore, the following perl programs will be applied. 

````
$ perl removal_of_zeros.pl descriptors.csv > refined_zeros_descriptors.csv
$ perl sd_csv.pl refined_zeros_descriptors.csv > refined_zeros_and_sd_descriptors.csv
````
** Step 3: **Correlation between each descriptor employing R package corrplot need to be calculated as:
````
$ R
Data <- read.csv(file=” refined_zeros_and_sd_descriptors.csv”, header=T) 
             require (corrplot)
	my_corr <- cor(data, method = “Pearson”, use = “complete.obs”)
	write.csv(my_corr, “/home/user/data_preprocessing/ refined_corr_descriptors.csv”,    row_names=TRUE)
````

**Step 4:** The file needs to be processed, descriptors having value more than 0.90 are removed to maintain data consistency as
``$ perl corr.pl refined_corr_descriptors.csv > corr_processed.csv``

**Step 5: ** The descriptor names are copied from corr_processed.csv file that will  be used as an input in "**ext_final.pl**" to extract the  final file as initial:
``$ perl ext_final.pl descriptors.csv > Final_ML_ready_file.csv``

**Step 6: ** Now label the molecules as 1s (inhibitors) and 0s (non-inhibitors) in the final Final_ML_ready_file.csv and make it ready for machine learning model. 
````
$ perl removal_of_zeros.pl descriptors.csv > refined_zeros_descriptors.csv
$ perl sd_csv.pl refined_zeros_descriptors.csv > refined_zeros_and_sd_descriptors.csv
````


- Machine Learning model: 
- we need to train the random forest model  (RF), so keep Final_ML_ready_file.csv file into a defined path and follow accordingly
- ````
$ R
source(“RF_train.R”)
````
The automated RF_train script will produce the  AUC-ROC plot and the confusion matrices for train and test dataset and top 30 features.
To find the true positives (the identified hits of out test data)  internal training:
``$ sh RF_prediction_training.sh``
We get the result as "**Identified_hits_for_internal_training.txt**"
Application of predictive model for independent validation dataset
````
source (“RF_valid.R”)
$ sh RF_prediction_training.sh
````
Now it produces "**Identified_hits_from_independent_set.txt**"
For the  training of extreme gradient boost (XGB) model keep "**Final_ML_ready_file.csv**" file into a defined path and follow the steps:
````
$ R
source(“xgb_train.R”) 
````
The automated **xgb_train** script produces AUC-ROC plot and confusion matrices for train and test dataset and top 30 features.

Application of the predictive model for independent validation dataset will be: ``source (xgb_valid.R”)``
Training of DNN (deep neural networks): for this keep 
"**Final_ML_ready_file.csv**" file into a defined path and follow the steps:
```
$ R
source(“DL_train.R”)
```
The automated DL_train script will again produce AUC-ROC plot and confusion

matrices for the train and the  test dataset.
To find true positives (Identified hits) for internal training
``$ sh DL_prediction_training.sh``
which produces "**Identified_hits_for_internal_training.txt**"
````
source (“DL_valid.R”)
$ sh DL_prediction_training.sh
```` 
And it produces "**Identified_hits_from_independent_set.txt**"
