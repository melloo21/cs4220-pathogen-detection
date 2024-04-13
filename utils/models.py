import joblib
import numpy as np
import datetime as dt
from typing import Union
from sklearn import ensemble, svm, tree, linear_model
from sklearn.model_selection import GridSearchCV, RepeatedKFold, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

# Fold references: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

class Classification:
    def __init__(self)-> None:
        super().__init__()


    def _fold_generator(
        self, 
        kfold_type:str,
        n_splits:int=2,
        n_repeats:int=1,
        random_state:Union[int,None]=None
    ):
        fold_dict = {
            "repeatedKfold" :
            RepeatedKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state
                ),
            "kfold": KFold( 
                n_splits=n_splits,
                shuffle=False,
                random_state=random_state),
            "StratifiedKFold" :StratifiedKFold(
                n_splits=n_splits,
                shuffle=False,
                random_state=random_state
            )
                
        }
        return fold_dict[kfold_type]

    def _model_manager(
        self,
        model_name:str
    ):
        # To initialise the model
        classifiers = {
            "decision_tree": tree.DecisionTreeClassifier(),
            "random_forest": ensemble.RandomForestClassifier(),
            "svc":svm.SVC(),
            "lr": linear_model.LogisticRegression(),
            "knn": KNeighborsClassifier()

        }
        return classifiers[model_name]

    def hyperparameter_tuning(
        self,
        df_tuple:tuple,
        score_metric:str,
        kfold_type:str,
        n_splits:int,
        params:dict,
        model_name:str
    ):
        """
        Summary: To tune the params
        Args: 
            df_tuple : (x_train, y_train)
            score_metric: List of score metrics available in json.
            kfold_type:str
            n_splits:int
            params:dict
            model_name:str            
        Returns:
        """

        # Training vals 
        x_train , y_train = df_tuple

        print(f"Model Tuning Starttime :: {dt.datetime.now()}")
        
        # Grid search to do hyperparameter tuning
        tuner = GridSearchCV(
            self._model_manager(model_name),
            param_grid=params,
            scoring=score_metric,
            cv=self._fold_generator(
                kfold_type=kfold_type,
                n_splits=n_splits
            ),
            n_jobs=-1,
            refit=True
        ).fit(x_train , y_train )

        print(f"Model Tuning Completed :: {dt.datetime.now()}")

        best_params = tuner.best_params_
        # feature_importance = tuner.best_estimator_.feature_importances_
        tuner_results = tuner.cv_results_
        return tuner, best_params, tuner_results
