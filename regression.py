import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import lightgbm as lgb

# Optional: PCA function if used
def apply_pca(x, n_components=50):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_components, x.shape[1]))
    x_pca = pca.fit_transform(x)
    return pd.DataFrame(x_pca)


#REGRESSION
def regression(x,y):
    st.subheader("📉 Regression")

    

    # ------------------ COMMON EVALUATION FUNCTION ------------------

    def evaluate_model(model, train_x, train_y, test_x, test_y):
        train_pred = model.predict(train_x)
        test_pred = model.predict(test_x)

        train_rmse = np.sqrt(mean_squared_error(train_y, train_pred))
        test_rmse = np.sqrt(mean_squared_error(test_y, test_pred))

        train_r2 = r2_score(train_y, train_pred)
        test_r2 = r2_score(test_y, test_pred)

        return train_rmse, test_rmse, train_r2, test_r2, model

    # ------------------ REGRESSION MODEL FUNCTIONS WITH GRID SEARCH ------------------

    def grid_search_model(model_class, param_grid, train_x, train_y, test_x, test_y):
        grid = GridSearchCV(model_class(), param_grid, cv=5)
        grid.fit(train_x, train_y)
        best_model = grid.best_estimator_
        return evaluate_model(best_model, train_x, train_y, test_x, test_y)

    def linear_regression(train_x, train_y, test_x, test_y):
        model = LinearRegression()
        model.fit(train_x, train_y)
        return evaluate_model(model, train_x, train_y, test_x, test_y)

    def ridge_regression(train_x, train_y, test_x, test_y):
        param_grid = {'alpha': np.linspace(0.01, 10.0, 10)}
        return grid_search_model(Ridge, param_grid, train_x, train_y, test_x, test_y)

    def lasso_regression(train_x, train_y, test_x, test_y):
        param_grid = {'alpha': np.linspace(0.01, 1.0, 10)}
        return grid_search_model(Lasso, param_grid, train_x, train_y, test_x, test_y)

    def decision_tree_reg(train_x, train_y, test_x, test_y):
        param_grid = {'max_depth': list(range(3, 16))}
        return grid_search_model(DecisionTreeRegressor, param_grid, train_x, train_y, test_x, test_y)

    def random_forest_reg(train_x, train_y, test_x, test_y):
        param_grid = {'n_estimators': list(range(50, 151, 25))}
        return grid_search_model(RandomForestRegressor, param_grid, train_x, train_y, test_x, test_y)

    def xgb_reg(train_x, train_y, test_x, test_y):
        param_grid = {'n_estimators': list(range(50, 151, 25)), 'max_depth': list(range(3, 7))}
        return grid_search_model(XGBRegressor, param_grid, train_x, train_y, test_x, test_y)

    def lgbm_reg(train_x, train_y, test_x, test_y):
        param_grid = {'num_leaves': list(range(20, 61, 10)), 'max_depth': list(range(-1, 16, 5))}
        return grid_search_model(lgb.LGBMRegressor, param_grid, train_x, train_y, test_x, test_y)

    def knn_reg(train_x, train_y, test_x, test_y):
        param_grid = {'n_neighbors': list(range(3, 11))}
        return grid_search_model(KNeighborsRegressor, param_grid, train_x, train_y, test_x, test_y)

    def elastic_net_reg(train_x, train_y, test_x, test_y):
        param_grid = {
            'alpha': np.linspace(0.01, 1.0, 10),
            'l1_ratio': np.linspace(0.1, 0.9, 5)}
        return grid_search_model(ElasticNet, param_grid, train_x, train_y, test_x, test_y)
    
    def svr_reg(train_x, train_y, test_x, test_y):
        param_grid = {
            'C': [0.1, 1, 10],
            'epsilon': [0.01, 0.1, 1],
            'kernel': ['rbf', 'linear']}
        return grid_search_model(SVR, param_grid, train_x, train_y, test_x, test_y)


    def drop_columns(d):
        st.subheader("Drop Unwanted Columns")
        
        cols_to_drop = st.multiselect("Select columns to drop", options=d.columns)

        if st.button("Drop Selected Columns"):
            d_dropped = d.drop(columns=cols_to_drop)
            st.success(f"✅ Dropped columns: {', '.join(cols_to_drop)}")
            st.subheader("Updated Data")
            st.dataframe(d_dropped)

            # Optional: Return cleaned data
            return d_dropped
        else:
            return d

    # ------------------ DATAFRAME GENERATOR ------------------

    def df_regression(train_x, train_y, test_x, test_y):
        models = [
            linear_regression,
            ridge_regression,
            lasso_regression,
            decision_tree_reg,
            random_forest_reg,
            xgb_reg,
            lgbm_reg,
            knn_reg,
            elastic_net_reg, 
            svr_reg
        ]
        results = []
        names = ['Linear', 'Ridge', 'Lasso', 'DecisionTree', 'RandomForest', 'XGB', 'LGBM', 'KNN', "ElasticNet",'SVR']

        for model_func in models:
            result = model_func(train_x, train_y, test_x, test_y)
            results.append(result)

        df_result = pd.DataFrame(results, columns=["Train RMSE", "Test RMSE", "Train R2", "Test R2", "Model"], index=names)
        return df_result

    # ------------------ MAIN LOGIC ------------------


    start_time = time.time()

    if x is not None and y is not None:
        

        st.dataframe(x.head())
        st.dataframe(y.head())


        st.info("📞 Use PCA for better functionality and efficiency in modeling.")

        if st.checkbox('🚠Apply PCA'):
            x= apply_pca(x)

        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

        data = df_regression(train_x, train_y.values.ravel(), test_x, test_y.values.ravel())
        st.dataframe(data)
        

        # DOWNLOAD MODELS
        st.sidebar.header("Download Trained Regression Models")
        for model_name in data.index:
            model = data.loc[model_name, 'Model']
            filename = f"{model_name}_reg_model.pkl"

            with open(filename, "wb") as f:
                pickle.dump(model, f)
            with open(filename, "rb") as f:
                st.sidebar.download_button(
                    label=f"📥 Download {model_name} Model",
                    data=f.read(),
                    file_name=filename,
                    mime="application/octet-stream",
                    key=f"download_{model_name}"
                )

        time_taken = time.time() - start_time
        st.success(f"Task completed in {time_taken:.2f} seconds")

        
        metric_choice = st.radio("Select metric to choose best model:", ['Test RMSE', 'Test R2'], horizontal=True)

        if metric_choice == 'Test RMSE':
            best_model_row = data.sort_values(by='Test RMSE').iloc[0]
        else:
            best_model_row = data.sort_values(by='Test R2', ascending=False).iloc[0]

        best_model_name = best_model_row.name
        best_model = best_model_row['Model'] 
        st.success(f"🏆 Best Model Based on {metric_choice}: **{best_model_name}**")
        st.write(f"✅ Test RMSE: {best_model_row['Test RMSE']:.4f}")
        st.write(f"✅ Test R²: {best_model_row['Test R2']:.4f}")
        


    return data,best_model
