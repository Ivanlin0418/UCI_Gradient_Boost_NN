import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_sparse_data(file_path):
    y = []
    rows, cols, data = [], [], []
    row_idx = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[0])
            # Convert -1 labels to 0
            if label == -1: 
                label = 0
            y.append(label)
            for item in parts[1:]:
                col, val = item.split(':')
                rows.append(row_idx)
                cols.append(int(col))
                data.append(float(val))
            row_idx += 1
    
    X = csr_matrix((data, (rows, cols)))
    y = np.array(y)
    return X, y

def create_interaction_features(X):
    n_features = X.shape[1]
    interactions = [X[:, i].multiply(X[:, j]) for i in range(n_features) for j in range(i+1, min(i+5, n_features))]
    return hstack([X] + interactions)

def add_polynomial_features(X):
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    return poly.fit_transform(X)

def dynamic_feature_selection(X_train, y_train):
    best_k = 0
    best_score = -np.inf
    best_selector = None
    for k in range(300, min(800, X_train.shape[1]), 100):
        selector = SelectKBest(f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        score = np.mean(cross_val_score(xgb.XGBClassifier(), X_train_selected, y_train, cv=5, scoring='accuracy', n_jobs=-1))
        if score > best_score:
            best_score = score
            best_k = k
            best_selector = selector
    logging.info(f'Best K: {best_k} with Accuracy: {best_score}')
    return best_selector

def objective(trial, X, y, feature_selector):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 0.5, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
        'eval_metric': 'logloss'
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        X_train_selected = feature_selector.transform(X_train)
        X_val_selected = feature_selector.transform(X_val)
        
        model = xgb.XGBClassifier(**params)
        
        model.fit(
            X_train_selected,
            y_train,
            eval_set=[(X_val_selected, y_val)],
            verbose=False
        )
        
        preds = model.predict(X_val_selected)
        score = accuracy_score(y_val, preds)
        scores.append(score)
    
    return np.mean(scores)

def train_stacking_ensemble(X_train_selected,y_train,best_params):
    base_models=[
        ('xgb',xgb.XGBClassifier(**best_params)),
        ('lgb',lgb.LGBMClassifier(device='gpu' ,boosting_type='gbdt' ,random_state=42)),
        ('cat',CatBoostClassifier(task_type='GPU' ,random_seed=42 ,verbose=False)) ]

    meta_model=LogisticRegression()

    stacking_model=StackingClassifier(estimators=base_models ,final_estimator=meta_model)

    stacking_model.fit(X_train_selected,y_train )

    return stacking_model

def iterative_optimization(X_train_scaled,y_train_scaled ,feature_selector):
    study=optuna.create_study(direction='maximize')

    study.optimize(lambda trial:objective(trial,X_train_scaled,y_train_scaled ,feature_selector),n_trials=50)

    best_params=study.best_params

    logging.info(f"Best score: {study.best_value:.4f}")

    return best_params


def load_and_preprocess_data(file_path):
    X,y=load_sparse_data(file_path)

    # Remove constant features
    selector=VarianceThreshold() # Default threshold is 0 to remove all-zero variance features
    X=selector.fit_transform(X)

    X=create_interaction_features(X)
    
    X=add_polynomial_features(X) # Add polynomial features

    scaler=StandardScaler(with_mean=False)
    
    X_scaled=scaler.fit_transform(X)

    return X_scaled,y


def main():
	logging.info("Starting main function...")

	X_train_scaled,y_train=load_and_preprocess_data('datasets/testing_data.txt')
    
	X_test_scaled,y_test=load_and_preprocess_data('datasets/training_data.txt')

	X_train,X_val,y_train,y_val=train_test_split(X_train_scaled,y_train,test_size=0.2 ,random_state=42)

	feature_selector=dynamic_feature_selection(X_train,y_train) # Use dynamic feature selection

	best_params=iterative_optimization(X_train,y_train ,feature_selector)

	X_train_selected=feature_selector.transform(X_train)

	stacking_model=train_stacking_ensemble(X_train_selected,y_train,best_params) # Train stacking ensemble

	X_test_selected=feature_selector.transform(X_test_scaled)

	test_preds=stacking_model.predict(X_test_selected) # Use stacking model for predictions

	test_accuracy=accuracy_score(y_test,test_preds )
    
	test_f1=f1_score(y_test,test_preds )

	logging.info(f"Test Accuracy: {test_accuracy:.4f}")
    
	logging.info(f"Test F1 Score: {test_f1:.4f}")


if __name__=="__main__":
	main()