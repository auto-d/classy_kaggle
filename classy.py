import re 
import argparse
from datetime import datetime
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn import metrics
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification

def find_na(df): 
    """
    Track down any elusive NA values
    """
    for col in df.columns: 
        na = len(df[df[col].isna()])
        if na > 0: 
            raise ValueError(f"{df}/{col} has {na} na values!") 

def scale(df, range=(0,1), omit=[]):
    """
    Scale a column into a given range, omitting one or more columns if provided """
    for column in df.columns: 
        if column not in omit: 
            df[column] = min_max_scale(df, column, (0,1))

    return df

def sum_by_index(df): 
    """
    Sum a (hopefully numeric) DF by it's index and return it
    """
    squashed = df.groupby(df.index).sum()
    squashed.fillna(0, inplace=True)

    return squashed

def ordinal_feature(df, column): 
    """
    Scale a column in a DF to the provided range     
    """
    encoder = OrdinalEncoder()
    encoder.fit(df[[column]])    
    ordinals = encoder.transform(df[[column]])
    return ordinals

def onehot_feature(df, column): 
    """
    One-hot encode a feature
    """
    encoder = OneHotEncoder(handle_unknown='ignore') 
    encoder.fit(df[[column]])

    columns = []
    for feature in encoder.get_feature_names_out(): 
        name = feature.replace('_', '.')
        name = name.replace(' ', '.')
        columns.append(name)
                       
    new_df = pd.DataFrame(encoder.transform(df[[column]]).toarray(), columns=columns, index=df.index) 

    return new_df

def min_max_scale(df, column, range=(0,1)): 
    """
    Scale a column in a DF to the provided range     
    """
    scaler = MinMaxScaler(feature_range=range)
    scaled = scaler.fit_transform(df[[column]]) 
    return scaled.transpose()[0]

def canonicalize_zip(zip, print_invalid=False) -> int: 
    """
    Normalize zipcode data
    """
    out = 0
        
    if int == type(zip): 
        # Normal zip
        if zip > 0 and zip < 99950: 
            out = zip
        
        # Perhaps international, but no reference...
        else: 
            pass 
            
    if str == type(zip):
        
        if zip.isdigit(): 
            # 3-digit ZIP (PR) missing the leading zeros 
            if len(zip) == 3: 
                out = int(zip) 
            
            # 4-digit missing a leading zero (e.g. some NJ, mass zips)
            elif len(zip) == 4: 
                out = int(zip) 
                
            # 5-digit ZIP
            elif len(zip) == 5: 
                out = int(zip) 

            else: 
                pass
                
        elif len(zip) > 5 and zip[5] == '-':
            # USPS extended ZIP
            out = int(zip[0:5]) 

        # Perhaps international 
        else: 
            pass
    else: 
        # Cruft
        pass 

    if out is None and print_invalid: 
        print(f"Ignoring unknown ZIP format:{zip}")

    return(out) 

def code_international(zip) -> bool: 
    """
    Heuristic to guess at international zips 
    """
    intl = 0
        
    if str == type(zip):
        if not zip.isdigit(): 
            intl = 1
            if len(zip) == 10 and zip[5] == '-':
                intl = 0

    return(intl) 

def compute_wpc(population, wages): 
    """
    Compute ages per capita 
    """
    return wages/population

def extract_pois(df, column): 
    """
    Extract people of interest (conductors, etc) from a text field
    """
    pois = set()
    pattern = r".*?([A-Za-z ]{3,30}).*?" 
    
    for id, row in df.iterrows(): 
        if type(row.loc[column]) == str: 
            for match in re.findall(pattern, row.loc[column], flags=re.DOTALL):        
                if len(match.strip()) > 4 and match.strip()[0].isupper(): 
                    pois.add(match.strip())
    return pois

def extract_composers(df, column): 
    """
    Extract composers from a text field
    """
    composers = set()
    pattern = r".*?([A-Z][A-Z .]{1,30}).*?" 
    
    for id, row in df.iterrows(): 
        if type(row.loc[column]) == str: 
            for match in re.findall(pattern, row.loc[column], flags=re.DOTALL):        
                if len(match.strip()) > 4: 
                    composers.add(match.strip())
    return composers

def map_price_level(level): 
    """
    Use a heuristic to convert the price level to values suitable for encoding as an ordinal... this is based on 
    the hunch that the price levels have a relative importance
    """
    result = None
    if level == 'Adult': 
        result = -1
    elif level == 'Youth': 
        result = -2
    elif level == 'GA': 
        result = -3
    elif type(level) == float: 
        print(level)
        result = int(level)
    elif type(level) == str: 
        try: 
            result = int(level) 
        except ValueError: 
            result = int(float(level))
    else: 
        result = level 
    return result

def load_train(file='data/train.csv'): 
    """
    Load the training data

    Columns: 
     - account.id
     - label
    """
    # Subset of accounts/patrons and whether they purchased a 2014-2015 subscription
    return pd.read_csv(file, index_col='account.id')

def load_test(file='data/test.csv'): 
    """
    Load the test accounts data
    
    Columns: 
      - ID (account)
    """
    # Test set containing accounts to generate predictions for 
    return pd.read_csv('data/test.csv', index_col='ID') 

def load_zip(file='data/zipcodes.csv'): 
    """
    Load the ZIP data

    Columns: 
      - Zipcode	
      - ZipCodeType
      - City
      - State
      - LocationType
      - Lat
      - Long
      - Location
      - Decommisioned
      - TaxReturnsFiled
      - EstimatedPopulation
      - TotalWages
    """
    df = pd.read_csv(file) 
    df['Zipcode'].apply(canonicalize_zip, args=[True]) 
    df.drop(['Decommisioned'], axis=1, inplace=True)

    return df

def load_accounts(zip_df, file='data/account.csv'): 
    """
    Load account information 
    
    Columns: 
      - account.id 
      - shipping.zip.code
      - billing.zip.code
      - shipping.city
      - billing.city
      - relationship
      - amount.donated.2013
      - amount.donated.lifetime
      - no.donations.lifetime
      - first.donated
    """
    # Presence of 0xc3 character suggests account.csv is encoded with ISO-8859-2
    df = pd.read_csv(file, encoding='ISO-8859-2', index_col='account.id')

    # Normalize the zipcodes
    df['shipping.zip'] = df['shipping.zip.code'].apply(canonicalize_zip, args=[False]) 
    df['billing.zip'] = df['billing.zip.code'].apply(canonicalize_zip, args=[False]) 
    
    # Heuristic for international coding 
    df['international'] = df['billing.zip.code'].apply(code_international)

    # Housekeeping
    df.drop(['shipping.zip.code', 'billing.zip.code'], axis=1, inplace=True) 
    df.rename(columns={'billing.zip': 'Zipcode'}, inplace=True)
    df['Zipcode'] = df['Zipcode'].astype(int)

    # Enrich our account data with zip info
    zip_df.set_index('Zipcode', inplace=True)
    df = df.join(other=zip_df, on='Zipcode', how='left', lsuffix='_a', rsuffix='_z')

    # New feature, wages per capita
    # Func on multiple rows with help from https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe
    df['wages.per.capita'] = df.apply(lambda x: compute_wpc(x.EstimatedPopulation, x.TotalWages), axis=1)
    df['wages.per.capita'] = df['wages.per.capita'].fillna(value=0) 

    # Housekeeping
    df['Lat'] = df['Lat'].fillna(value=0) 
    df['Long'] = df['Long'].fillna(value=0) 
    df.drop(['shipping.city', 'shipping.zip', 'ZipCodeType', 'City', 'State', 'Location', 'LocationType', 'TaxReturnsFiled', 'EstimatedPopulation', 'TotalWages'], axis=1, inplace=True) 

    # Opt for ordinal encoding of relationship - 
    # TODO: this should be one-hot!
    rel_df = onehot_feature(df, 'relationship') 
    df = pd.concat([df, rel_df], axis=1)
    
    # Housekeeping
    df.drop(['relationship'], axis=1, inplace=True)

    # Convert date and scale
    basetime = datetime(1970, 1, 1)
    df['first.donated'] = pd.to_datetime(df['first.donated'])
    df['first.donated'] = pd.to_numeric(df['first.donated'].fillna(basetime))
    df['first.donated.scaled'] = min_max_scale(df, 'first.donated')
    
    # Housekeeping
    df.drop(['first.donated'], axis=1, inplace=True)
    df.drop(['billing.city'], axis=1, inplace=True)

    # Alarm on NAs...  
    find_na(df) 

    return df

def load_concerts(file='data/Concerts.csv'): 
    """
    Load concert data

    Columns: 
      - season
      - concert.name
      - set
      - who
      - what
      - location
    """
    df = pd.read_csv(file) 
    
    # New feature
    df['set.id'] = df['season'] + "-s" + df['set'].astype(str)

    # A deduplicated DF with just the mapping of year/set to composer et al 
    df.drop_duplicates(subset='set.id', keep='first', inplace=True) 
    df.set_index('set.id', inplace=True) 
    df['location'] = df['location'].fillna(value="")
    pois = extract_pois(df, 'who') 
    composers = extract_composers(df, 'what')     
    
    # Alarm on NAs...  
    find_na(df) 

    return df, pois, composers

def load_upcoming_concerts(pois, composers, file='data/Concerts_2014-15.csv'):
    """
    Load the upcoming concerts for this season 
    
    Columns: 
      - season
      - concert.name
      - set
      - who
      - what
    """
    df = pd.read_csv(file) 
    planned_pois = extract_pois(df, 'who') 

    # Need a fixed index on these folks, use list in lieu of the set going forward
    salient_pois = list(planned_pois.intersection(pois))
    planned_composers = extract_composers(df, 'what') 
    planned_composers.intersection(composers)

    # The intersection between the composers' pieces that were performed and the planned composers is small, 
    # and has to be correlated manually here short of some NLP magic. Conveniently, though we can rely on 
    # presence of the below strings to indicate historic or planned presence of the associated composer. We
    # perhaps shamefully throw all the BACHs in one bucket here. All other composers will be represented by
    # the absence of any information in the associated one-hot encodings. 
    #
    # This may be the only real feature worth applying from the 2014-2015 data!
    salient_composers = [ 'BACH', 'HAYDN', 'VIVALDI', 'TELEMANN', 'HANDEL' ]

    return df, salient_pois, salient_composers

def load_tickets(pois, composers, concerts_df, file='data/tickets_all.csv'):
    """
    Load the tickets data 
    
    Columns: 
      - account.id
      - price.level
      - no.seats
      - marketing.source
      - season
      - location
      - set
      - multiple.tickets
    """
    df = pd.read_csv(file, index_col='account.id') 
    
    # Housekeeping 
    df['set'] = df['set'].fillna(value=0) 
    df['set'] = df['set'].astype(int)

    # New feature 
    df['set.id'] = df['season'] + "-s" + df['set'].astype(str)

    # Enrich with conert info
    df = df.join(concerts_df, on='set.id', how='left', rsuffix='_temp')
    
    # Ordinal encode season
    df['season.ordinal'] = ordinal_feature(df, 'season')

    # One-hot encode location categorical 
    loc_df = onehot_feature(df, 'location') 
    df = pd.concat([df, loc_df], axis=1)

    # Ordinal encode the ticket feature 
    df['multiple.tix.ordinal'] = ordinal_feature(df, 'multiple.tickets') 

    # Housekeeping 
    df.drop(['marketing.source', 'season', 'season_temp', 'location', 'location.nan', 'multiple.tickets', 'set.id', 'set_temp', 'location_temp'], axis=1, inplace=True)
    df['price.level'] = df['price.level'].fillna(value=0)

    # Map the price levels into something more suitable for machine inference
    df['price.level'] = df['price.level'].apply(map_price_level)

    # One-hot encode the subset of composers/people of interest that we suspect will have predictive 
    # value (because they occur in both historical concerts and upcoming concerts)
    def find_poi(text, poi): 
        if type(text) == str: 
            return 1 if poi in text else 0
    
    for poi in pois: 
        column = poi.replace(' ', '.')
        df[column] = [0] * len(df) 
        df[column] = df.apply(lambda x: find_poi(x.who, poi), axis=1)

    for composer in composers: 
        column = composer.replace(' ', '.') 
        df[column] = [0] * len(df) 
        df[column] = df.apply(lambda x: find_poi(x.what, composer), axis=1)

    # Housekeeping 
    df.drop(['concert.name', 'who', 'what', 'price.level', 'season.ordinal'], axis=1, inplace=True)
    df.fillna(value=0, axis=1, inplace=True)

    # Canary     
    find_na(df) 

    return df

def load_subscriptions(file='data/Subscriptions.csv'):
    """
    Load the subscriptions data 
    
    Columns: 
      - account.id	
      - season
      - package
      - no.seats
      - location
      - section
      - price.level
      - subscription_tier
      - multiple.subs
    """
    df = pd.read_csv(file, index_col='account.id')
    
    # Seasons have a relative ordering, encode as ordinals
    df['season.ordinal'] = ordinal_feature(df, 'season') 
    
    # One-hot encode categoricals
    df.rename(columns={'package': 'pkg'}, inplace=True)
    df.rename(columns={'location': 'loc'}, inplace=True)
    df.rename(columns={'section': 'sn'}, inplace=True)
    pkg_df = onehot_feature(df, 'pkg') 
    loc_df = onehot_feature(df, 'loc') 
    sn_df = onehot_feature(df, 'sn') 
    df = pd.concat([df, loc_df], axis=1)
    df = pd.concat([df, pkg_df], axis=1)
    df = pd.concat([df, sn_df], axis=1)

    # Multiple subscriptions have a relative ordering... 
    df['multiple.subs'] = ordinal_feature(df, 'multiple.subs') 
    
    # Price level *seems* to have a relative ordering... 
    df['price.level'] = df['price.level'].fillna(value=0) 
    
    # Season has a relative ordering 
    df['season.ordinal'] = df['season.ordinal'].astype(int) 

    # Remove cruft and check for any residual empty cells 
    df.drop(['season', 'pkg', 'loc', 'sn', 'sn.nan', 'season.ordinal'], axis=1, inplace=True) 
    
    # Canary 
    find_na(df) 

    return df

def predict_subscribers(model, seasons, df, num_prior=1): 
    """
    Try to fit the provided model on the prior season(s) data to predict the current season's subscription renewals. 
    """
    results = [] 

    # Iterate over the seasons, in order, looking to see how we fare at predicting the current year's 
    # subscriptions based on the previous
    for season in seasons: 
        first_season = season - num_prior
        next_season = season + 1
        score = 0 

        # Ensure our window overlaps with the available data
        if first_season in seasons and next_season in seasons: 
            prior_df = df[(df['season.ordinal'] >= first_season) & (df['season.ordinal'] < season)]
            this_df = df[df['season.ordinal'] == season]
            next_df = df[df['season.ordinal'] == next_season] 
            
            # Figure out which accounts from prior year(s) also subscribed this year
            current_subscribers = [1 if index in this_df.index else 0 for index in prior_df.index]
            future_subscribers = [1 if index in next_df.index else 0 for index in this_df.index]

            model.fit(model, prior_df, current_subscribers)
            
            # Check for utility 
            probs = model.predict_proba(this_df)[:,1]
            score = metrics.roc_auc_score(future_subscribers, probs)
            print(f"Season {season} result: {score:.4f}")
            results.append(score) 
    
    return results

def make_train_test_sets(train_df, test_df, df):
    """
    Merge train and test accounts with cleaned, engineered DF for prediction
    """
    train_df = train_df.join(df, how='left', rsuffix="_acc") 
    X_train = train_df.drop(['label'], axis=1)
    y_train = train_df['label']
    X_test = test_df.join(df, how='left')

    return X_train, y_train, X_test 

def categorize_prediction(label, pred): 
    """
    What the name says 
    """
    if label and pred: 
        return "TP"
    elif label and not pred: 
        return "FN"
    elif not label and pred: 
        return "FP"
    elif not label and not pred: 
        return "TN"

def pca2(df, cluster_method='kmeans', clusters=8): 
    """
    Compress the provided dataframe into 2 dimensions to support visualization, 
    note scaling is implicit
    """
    scaled_df = scale(df.copy())

    pca = PCA(n_components=2) 
    pca.fit(scaled_df)
    
    lowD = pca.transform(scaled_df) 

    if cluster_method == 'kmeans':         
        km_model = KMeans(n_clusters=clusters, random_state=0, n_init="auto")
        km_model.fit(lowD) 
        centroids = km_model.cluster_centers_
        labels = km_model.labels_
    elif cluster_method =='dbscan': 
        db = DBSCAN(eps=0.5, min_samples=10)
        db.fit(lowD)
        centroids = None
        labels = db.labels_ 

    return lowD, centroids, labels

def apply_pca2(df, clusters=8):
    """
    Enhance a df with 2-component PCA and the clusters that k-means identifies
    in that 2-d space
    """
    lowD, centroids, labels = pca2(df)
    
    df = df.copy()
    df['d1'] = lowD.transpose()[0]
    df['d2'] = lowD.transpose()[1]

    df['cluster'] = labels

    return df, centroids

def project_results_2d(X_train, y_train, probs, threshold=0.5, clusters=8): 
    """
    Use PCA to prepare a flattened version of the training data and our performance on the TRAINING data predictions
    """
    X_train_2d, cluster_centers = apply_pca2(X_train, clusters)
    
    # Create a subset just for training data, and then subsets for the classes
    viz_df = X_train_2d.join(y_train, how='inner', rsuffix='_y') 

    viz_df['preds'] = [ True if prob > threshold else False for prob in probs]
    viz_df['result'] = viz_df.apply(lambda x: categorize_prediction(x.label, x.preds), axis=1)

    return viz_df, cluster_centers

def visualize_results_2d(viz_df, cluster_centers, title, c_filter=None): 
    """
    Plot the 2d visualization, optionally with cluster centroids and/or a cluster 
    centroid filter
    """
    if c_filter is not None and c_filter != []: 
        viz_df = viz_df[viz_df['cluster'].isin(c_filter)]

    tp_sub = viz_df[viz_df['result'] == "TP"]
    fn_sub = viz_df[viz_df['result'] == "FN"]
    fp_sub = viz_df[viz_df['result'] == "FP"]
    tn_sub = viz_df[viz_df['result'] == "TN"]
    
    fig = plt.figure()     
    fig.set_size_inches(16,10) 
    
    plt.title(title)
    plt.scatter(tn_sub['d1'], tn_sub['d2'], color='gray', marker='o', label='TN')     
    plt.scatter(tp_sub['d1'], tp_sub['d2'], color='blue', marker='o', label='TP')     
    plt.scatter(fn_sub['d1'], fn_sub['d2'], color='orange', marker='.', label='FN') 
    plt.scatter(fp_sub['d1'], fp_sub['d2'], color='red', marker='.', label='FP') 

    # Note we are not plotting the DBscan clusters here, something to improve on 
    if cluster_centers is not None: 
        for cluster in range(1,len(cluster_centers)):  
            if c_filter is None or cluster in c_filter: 
                center = cluster_centers[cluster] 
                plt.scatter(center[0], center[1], color='yellow', marker='D', label='Centroids') 
                plt.annotate(cluster, (center[0], center[1]), bbox=dict(boxstyle="round", fc="0.8"))

    plt.show()

@ignore_warnings(category=ConvergenceWarning)
def bakeoff(experiments, X_train, y_train, visualize=False): 
    """
    Iterate over various options, looking for an optimal model given the data. This 
    function expects all cross-validation to happen within the model pipeline. 
    """
    winner_roc = 0 
    winner = None    

    for i, experiment in enumerate(experiments): 
        experiment.fit(X_train, y_train)
        
        probs = None 
        if hasattr(experiment, 'predict_proba'): 
            probs = experiment.predict_proba(X_train)[:,1]
        elif hasattr(experiment, 'predict'):
            probs = experiment.predict(X_train)

        roc = metrics.roc_auc_score(y_train, probs)

        print(f"==== \nExperiment {i}: {roc}\n")

        if 'model' in experiment.named_steps.keys():
            model = experiment.named_steps['model']
        elif 'grid' in experiment.named_steps.keys(): 
            model = experiment.named_steps['grid'].best_estimator_
        else: 
            raise ValueError("Cannot extract estimator from pipeline!")

        if winner_roc < roc: 
            winner_roc = roc 
            winner = model 
    
        if visualize: 
            viz_df, centroids = project_results_2d(X_train, y_train, probs, threshold=0.5, clusters=8)
            visualize_results_2d(viz_df, centroids, title=f"Pipeline {i}, Model: {str(winner)}", c_filter=[])

    return winner, winner_roc

def build_train_set():
    """
    Load, transform, and apply engineered features, returning the training set
    """
    # Import train and test indices
    train_df = load_train()
    test_df = load_test()

    # Import, clean and prepare reference data, memorializing the intersection of historical and upcoming composers/performers
    concerts_df, pois, composers = load_concerts()
    zip_df = load_zip()
    _, salient_pois, salient_composers = load_upcoming_concerts(pois, composers)

    # Import, clean and prepare predictors
    accounts_df = load_accounts(zip_df)
    tickets_df = load_tickets(salient_pois, salient_composers, concerts_df)
    subs_df = load_subscriptions()

    # Collapse and join predictors
    feature_df = accounts_df.join(sum_by_index(tickets_df), how="left", rsuffix="_tix")
    feature_df = feature_df.join(sum_by_index(subs_df), how="left", rsuffix="_sub")
    feature_df.fillna(0, inplace=True)    
    feature_df, _ = apply_pca2(feature_df)

    # Train any tributary models or inputs to the CBBDF
    #scores = predict_subscribers(lr_model, seasons, subs_df, 1) 
     
    # Generate dataframes for training
    find_na(feature_df)

    return make_train_test_sets(train_df, test_df, feature_df)

def generate_submission(model, X_test, directory='submissions'): 
    """
    Generate a submission in the format expected by the competition
    """
    submit_df = pd.DataFrame() 
    submit_df['ID'] = X_test.index
    probs = model.predict_proba(X_test)
    submit_df['Predicted'] = probs[:,1]
        
    file_path = directory + '/' + datetime.now().strftime("%m%d_%H%M") + '.csv'    
    submit_df.to_csv(file_path, index=False)
    print(f"Wrote submission for model {model} to {file_path}.")

lr_hparams = { 'penalty' : ('l1', 'l2', 'elasticnet'), 'C' : [x / 10 for x in range(0,10)]}
rf_hparams = { 'min_samples_leaf' : range(3,5,1), 'n_estimators': range(40,50,5), 'max_depth': range(7,9,1)}
sv_hparams = { 'C' : [x / 10 for x in range(1,5,2)], 'kernel' : ['sigmoid', 'rbf'] }
kn_hparams = { 'n_neighbors' : range(7,9)}

def search(splits=3, submit=False, visualize=False):  
    """
    Perform a hyperparameter and model search across all promising algorithms
    """
    X_train, y_train, X_test = build_train_set()

    experiments = [
        # Control
        #Pipeline([('model', DummyClassifier())]),

        # Experimental models and hyperparameter tuning stuff -- these should all be gridsearch estimators, as
        # we are delegating cross-validation to that object and relying on the ability to retrieve the optimal model 
        Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures()), ('grid', GridSearchCV(SVC(), sv_hparams, error_score=0))]),
        Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=15)), ('grid', GridSearchCV(SVC(), sv_hparams, error_score=0))]),
        Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=12)), ('grid', GridSearchCV(SVC(), sv_hparams, error_score=0))]),

        Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures()), ('grid', GridSearchCV(KNeighborsClassifier(), kn_hparams, error_score=0))]), 
        
        # Logistic Regression w/ L1/L2 norm penalties         
        Pipeline([('grid', GridSearchCV(LogisticRegression(),lr_hparams, cv=splits, scoring='roc_auc', error_score=0))]),

        # Random forest         
        Pipeline([('grid', GridSearchCV(RandomForestClassifier(),rf_hparams, cv=splits, scoring='roc_auc', error_score=0))]),
        
        # PCA + LR
        Pipeline([('scaler', StandardScaler()), ('pca3', PCA(n_components=12)), ('model', SVC(kernel='rbf'))]),
        Pipeline([('scaler', StandardScaler()), ('pca3', PCA(n_components=5)), ('model', SVC(kernel='rbf'))]),
        Pipeline([('scaler', StandardScaler()), ('pca3', PCA(n_components=12)), ('model', SVC(kernel='sigmoid'))]),
        Pipeline([('scaler', StandardScaler()), ('pca6', PCA(n_components=26)), ('model', LogisticRegression(penalty=None))]),
        Pipeline([('scaler', StandardScaler()), ('pca8', PCA(n_components=8)), ('model', LogisticRegression(penalty=None))]),
    ]
    
    # Check algorithm outcomes on enriched account data and submit the winner
    winner, roc = bakeoff(experiments, X_train, y_train, visualize)

    print(f"Best model identified: {winner} (AUROC of {roc}).")

    if submit:
        generate_submission(winner, X_test)

def test(splits=5, submit=False, visualize=False): 
    """
    Test the most promising model configurations using cross validation    
    """
   
    candidates = [        
        # Candidate models and hyperparameters reseulting from search operations 
        # Note estimator must be tagged w/ 'model' 
        Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=7)), ('model', LogisticRegression(penalty=None))]),
        Pipeline([('scaler', StandardScaler()), ('pca4', PCA(n_components=34)), ('model', LogisticRegression(penalty=None))]),
        Pipeline([('model', RandomForestClassifier(max_depth=8, min_samples_leaf=4, n_estimators=45))]),
        Pipeline([('scaler', StandardScaler()), ('model', KNeighborsClassifier(n_neighbors=8))]), 
        Pipeline([('model', VotingClassifier(
            estimators=[
                ('RF', RandomForestClassifier(max_depth=9, min_samples_leaf=4, n_estimators=40)), 
                ('KNN', Pipeline([('scaler', StandardScaler()), ('model', KNeighborsClassifier(n_neighbors=8))])), 
                ],
            voting='soft'))]),
        Pipeline([('model', VotingClassifier(
            estimators=[
                ('RF', RandomForestClassifier(max_depth=8, min_samples_leaf=4, n_estimators=70)), 
                ('KNN', Pipeline([('scaler', StandardScaler()), ('model', KNeighborsClassifier(n_neighbors=8))])), 
                ],
            voting='soft'))]),
        # This performs best on internal validation, but is a few tenths below the above voting classifier on the hidden 
        # test data. Keep it here for potential mutation. 
        # Pipeline([('model', VotingClassifier(
        #     estimators=[
        #         ('RF', RandomForestClassifier(max_depth=8, min_samples_leaf=4, n_estimators=45)), 
        #         ('KNN', Pipeline([('scaler', StandardScaler()), ('model', KNeighborsClassifier(n_neighbors=8))])), 
        #         ('LR', Pipeline([('scaler', StandardScaler()), ('pca4', PCA(n_components=34)), ('model', LogisticRegression(penalty=None))])),
        #         ],
        #     voting='soft'))]),
    ]

    X_train, y_train, X_test = build_train_set()    
    kf = KFold(n_splits=splits, shuffle=False)

    winner_roc = 0
    winner = None
    
    # Iterate over the candidate models and evalute on k folds of the data
    for candidate in candidates: 

        roc = 0 
        for train_ix, test_ix in kf.split(X_train, y_train):            
            candidate.fit(X_train.iloc[train_ix], y_train.iloc[train_ix])
        
            probs = None 
            if hasattr(candidate, 'predict_proba'): 
                probs = candidate.predict_proba(X_train.iloc[test_ix])[:,1]
            elif hasattr(candidate, 'predict'):
                probs = candidate.predict(X_train.iloc[test_ix])

            roc += metrics.roc_auc_score(y_train.iloc[test_ix], probs)/splits

        # Retrain in preparation for visualization or submission
        model = candidate.named_steps['model']
        print(f"======== \nCandidate {model}: {roc}\n")
        model.fit(X_train, y_train)
            
        if winner_roc < roc: 
            winner_roc = roc 
            winner = model 
        
        # If we're going to plot, we need updated predictions
        if visualize:          
            probs = None 
            if hasattr(model, 'predict_proba'): 
                probs = model.predict_proba(X_train)[:,1]
            elif hasattr(model, 'predict'):
                probs = model.predict(X_train)

            viz_df, centroids = project_results_2d(X_train, y_train, probs, threshold=0.5, clusters=8)
            visualize_results_2d(viz_df, centroids, title=f"Model: {str(model)} @ {roc}", c_filter=None)
    
    if submit:
        generate_submission(winner, X_test)

    # TODO: am I sampling with stratification? my positive class is poorly represented ... review what we're doing 
    # to address 

    # ❗️ TODO: implement these!! PCA could be epic... but somehow makes NO DIFFERENCE, revisit
    # apply_algorithms(tickets_df)
    # apply_algorithms(subs_df)
    # apply_algorithms(pca(accounts_df)

def main(**args): 
    """
    CLI entry point and arg handler
    """
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group() 
    group.add_argument("-s", "--search", action=argparse.BooleanOptionalAction)
    group.add_argument("-t", "--test", action=argparse.BooleanOptionalAction)
    parser.add_argument("-k", "--splits")
    parser.add_argument("-v", "--visualize", action=argparse.BooleanOptionalAction)
    parser.add_argument("-g", "--generate", action=argparse.BooleanOptionalAction)
    
    parser.set_defaults(search=False, test=False, visualize=False, generate=False)
    args = parser.parse_args()
    if args.search: 
        search(int(args.splits), args.generate, args.visualize)
    elif args.test: 
        test(int(args.splits), args.generate, args.visualize)
    else: 
        parser.print_help()

if __name__ == "__main__": 
    main()
    
