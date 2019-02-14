# csedm-Data-Challenge
code for csedm Data Challenge
requirements for running this code
    python 3.6
    numpy
    pandas
    sklearn
    tensorflow
    
To run the code:
    first run preprocess.py. 
        first edit the preprocess.py file
        Two important parameters 'mainTableFile' and 'codeStateFile' should be assigned according to those two files' path
        Left 'embeddingSize' to the default value 8 
        After the edition, just use 'python preprocess.py' to run the code
    Then run makePredictions.py to generate prediction on test files. This will generate the 'cv_predict.csv' file recording all the predctions
        edit the file , just to assign the CV folder's path 
        Using 'python makePredictions.py' to run the code
    LThe last step is to run evaluation.py to caculate indices. 
        just run the file using similar commond
        
This would generate Three required file 'cv_predict.csv' , 'evaluation_by_problem' and 'evaluation_overall' 
