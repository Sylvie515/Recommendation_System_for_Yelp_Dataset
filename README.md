# Recommendation System for Yelp Dataset  

Hybrid recommender that combines item-based collaborative filtering (MinHash + LSH + Pearson similarity) with a XGBoost regressor. Implemented with PySpark RDD for scalable CF and XGBoost for feature-based predictions.  

## Project Structure  

root/  
├─ competition.py  
├─ Evaluation_and_Runtime.ipynb&nbsp;&nbsp;&nbsp;&nbsp;# val metrics  
├─ data/  
│&nbsp;&nbsp;&nbsp;&nbsp;├─ yelp_train.csv&nbsp;&nbsp;&nbsp;&nbsp;# train data  
│&nbsp;&nbsp;&nbsp;&nbsp;├─ yelp_val.csv&nbsp;&nbsp;&nbsp;&nbsp;# validation data  
│&nbsp;&nbsp;&nbsp;&nbsp;├─ user.json&nbsp;&nbsp;&nbsp;&nbsp;# user metadata  
│&nbsp;&nbsp;&nbsp;&nbsp;├─ business.json&nbsp;&nbsp;&nbsp;&nbsp;# business metadata  
│&nbsp;&nbsp;&nbsp;&nbsp;├─ tip.json  
│&nbsp;&nbsp;&nbsp;&nbsp;├─ photo.json  
│&nbsp;&nbsp;&nbsp;&nbsp;└─ checkin.json  
└─ README.md  
 
## Requirements & Environment  

```bash
pip install xgboost scikit-learn numpy pandas pyspark
```

- Originally implemented under Python 3.6 for course submission (Vocareum)  
- Evaluation_and_Runtime.ipynb was originally tested on Google Colab environment  

## Pipeline  

### CF + XGBoost

- Item-based CF:  
    - Build [(user_id, business_id, stars), ......] records from yelp_train.csv.  
    - Filter out users with only 1 rating or more than 500 ratings, and keep only businesses with at least 100 ratings when constructing LSH signatures.  
    - MinHash + LSH with 20 bands and 5 rows per band generates candidate business pairs.  
    - Compute Pearson similarity for candidate pairs, and scale it by min(count, 60)/60 to downweight pairs with very few co-rated users.  
    - For each business, keep at most top-35 similar neighbors.  
    - For prediction, only neighbors that the target user has rated are used. The CF score is a similarity-weighted average, then lightly smoothed toward the global average and clipped to [1, 5].  
- XGBoost model:  
    - Extract additional features from user.json and business.json. Non-finite feature values (NaN/inf) are converted to 0.0.  
    - Hyperparameters are tuned by offline GridSearch CV on a subsample of training data:  
      - 2-stage: A coarse search over a wider grid, followed by a finer search around the best region.  
      - Both use 5-fold CV and RMSE metric, searching over learning_rate, max_depth, n_estimators, min_child_weight, reg_alpha, etc.  
    - Several CV parameters achieved similar RMSE around 0.98. The final parameters are chosen based on the performance of the full CF + XGBoost hybrid pipeline on validation data, rather than purely on the standalone XGBoost CV scores.  
    - XGBoost outputs are smoothed toward the global average and clipped to [1, 5].  
- Hybrid scheme:
    - For each (user, business) pair in the test set:  
      - If the target business has at least 25 CF neighbors and a CF prediction exists:  
        - Mix a neighbor-count–weighted CF score with a simple baseline.  
        - Then combine CF score with XGBoost score using an alpha that grows with log(neighbor_cnt) (capped at 0.2).  
      - Otherwise, use the XGBoost prediction only.  
    - Apply a global smoothing and clip the result into [1, 5] as final prediction.  

### Two-Stage GridSearch  

For the model-based part, I followed a two-stage hyperparameter tuning strategy.  

- #### Stage 1: coarse search over a wide parameter range using 5-fold CV  
  Performed a coarse GridSearchCV over a wide parameter range using 5-fold cross-validation on a randomly subsampled subset of the training data (around 80,000 samples) due to computational constraints on Colab.  
- #### Stage 2: fine search over a narrower range centered on the best results from Stage 1.  
  Refined the search in a smaller neighborhood centered around the best configuration from Stage 1 (including learning_rate, max_depth, n_estimators, min_child_weight, and reg_alpha) again with 5-fold CV.  

Several configurations achieved very similar cross-validation RMSE. The final hyperparameters used in the hybrid system were selected based on the performance of the full CF+XGBoost pipeline on the validation set, not only on the standalone XGBoost CV score.  

## Execution  

Run the full pipeline using Spark:

```bash
spark-submit competition.py <folder_path> <test_file_name> <output_file_name>  
```

##  Evaluation & Runtime (Validation data)  

The model was evaluated on the validation dataset (yelp_val.csv) using RMSE and error distribution.  

### Error Distribution  

| Error Range | Count |  
|------------|------|   
| [0, 1)     | 102,076 |  
| [1, 2)     | 33,047 |  
| [2, 3)     | 6,155 |  
| [3, 4)     | 766 |  
| ≥ 4        | 0 |  

### RMSE  
RMSE = 0.9782  

### Execution Time  
~490 seconds (measured on local/Colab environment)  

### Notes  
- Runtime includes CF (LSH + similarity computation), feature engineering, model training, and hybrid prediction.  
- GridSearch is performed offline on a subsampled dataset (~80k samples) to reduce computational cost.  
- Final evaluation is based on the full hybrid pipeline rather than standalone model performance.  
