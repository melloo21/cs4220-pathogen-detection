from sklearn.ensemble import RandomForestRegressor

def random_forest_selector(X,y):
    # Using feature importance to select features
    model = RandomForestRegressor(random_state=1, max_depth=10)
    model.fit(X,y)
    features = X.columns
    importances = model.feature_importances_
    # Create a dataframe for feature importance
    feature_importance_df = pd.DataFrame({"features": list(features) ,"importances": list(importances)} )
    indices = np.argsort(importances)[-9:]  # top 10 features
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

    return feature_importance_df

def get_kmer_analysis_map(dataset, method:str="mean"):
    test = dataset.X_mapped.copy()
    test["labels"] = dataset.Y   
    if method == "mean":
        return test.groupby("labels").mean().reset_index()
    elif method == "median":
        return test.groupby("labels").median().reset_index()

def get_kmer_profile_by_label(kmer_analysis):
    kmer_profile_by_label = dict()
    # Obtaining the profile
    for elem in kmer_analysis.index:
        kmer_profile_by_label[elem] = kmer_analysis.iloc[elem,1:-1]
    return kmer_profile_by_label

def getting_no_kmer_existence(kmer_analysis):
    kmer_profile_by_label = get_kmer_profile_by_label(kmer_analysis)

    # Getting label without that kmer
    for elem in kmer_analysis.index:
        zero_kmer = list(kmer_profile_by_label[elem][kmer_profile_by_label[elem]==0].index)
        if (len(zero_kmer) > 0):
            print(f" label {elem} ::  {zero_kmer}")

    return zero_kmer

def get_label_by_kmer(kmer_analysis):
    label_profile_by_kmer = dict()

    # Obtaining the profile
    for elem in range(1,len(kmer_analysis.columns)-1):
        label_profile_by_kmer[kmer_analysis.columns[elem]] = kmer_analysis.iloc[:,elem]
    return label_profile_by_kmer

def get_std_across_labels_by_kmer(kmer_analysis):
    label_profile_by_kmer = get_label_by_kmer(kmer_analysis)

    std_accross_labels = dict()

    for key, values in label_profile_by_kmer.items():
        std_accross_labels[key] = np.std(values)

    # Sort by variation
    std_accross_labels_sorted = dict(sorted(std_accross_labels.items(), key=lambda item: item[1]))

    return std_accross_labels_sorted