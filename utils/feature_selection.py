from sklearn.ensemble import RandomForestRegressor

def random_forest_selector(X,y):
    # Using feature importance to select features
    model = RandomForestRegressor(random_state=1, max_depth=10)
    model.fit(X,y)
    features = df.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[-9:]  # top 10 features
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()