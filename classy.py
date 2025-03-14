import random
import re 
from datetime import datetime
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def find_na(df): 
    for col in df.columns: 
        na = len(df[df[col].isna()])
        if na > 0: 
            raise ValueError(f"{df}/{col} has {na} na values!") 

def ordinal_feature(df, column): 
    """
    Scale a column in a DF to the provided range     
    """
    encoder = OrdinalEncoder()
    encoder.fit(df[[column]])
    print('Found categories ' + str(encoder.categories_))
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
    out = 0
        
    if int == type(zip): 
        if zip > 0 and zip < 99950: 
            out = zip
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
            out = int(zip[0:5]) 

        else: 
            
            pass
    else: 
        pass 

    if out is None and print_invalid: 
        print(f"Ignoring unknown ZIP format:{zip}")

    return(out) 

def code_international(zip) -> bool: 
    intl = 0
        
    if str == type(zip):
        if not zip.isdigit(): 
            intl = 1
            if len(zip) == 10 and zip[5] == '-':
                intl = 0

    return(intl) 

def load_train(file='data/train.csv'): 
    # Subset of accounts/patrons and whether they purchased a 2014-2015 subscription
    train_df = pd.read_csv(file, index_col='account.id')
    train_df.head() 

def load_zip(file='data/zipcodes.csv'): 
    zip_df = pd.read_csv(file) 
    zip_df['Zipcode'].apply(canonicalize_zip, args=[True]) 
    zip_df.drop(['Decommisioned'], axis=1, inplace=True)

def compute_wpc(population, wages): 
    return wages/population

def load_accounts(file='data/account.csv'): 
    # Presence of 0xc3 character suggests account.csv is encoded with ISO-8859-2
    account_df = pd.read_csv(file, encoding='ISO-8859-2', index_col='account.id')

    account_df['shipping.zip'] = account_df['shipping.zip.code'].apply(canonicalize_zip, args=[False]) 
    account_df['billing.zip'] = account_df['billing.zip.code'].apply(canonicalize_zip, args=[False]) 
    account_df['international'] = account_df['billing.zip.code'].apply(code_international)
    account_df.drop(['shipping.zip.code', 'billing.zip.code'], axis=1, inplace=True) 
    account_df.rename(columns={'billing.zip': 'Zipcode'}, inplace=True)
    account_df['Zipcode'] = account_df['Zipcode'].astype(int)
    zip_df.set_index('Zipcode', inplace=True)
    account_df = account_df.join(other=zip_df, on='Zipcode', how='left', lsuffix='_a', rsuffix='_z')
    # With help from https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe
    account_df['wages.per.capita'] = account_df.apply(lambda x: compute_wpc(x.EstimatedPopulation, x.TotalWages), axis=1)
    account_df['wages.per.capita'] = account_df['wages.per.capita'].fillna(value=0) 
    account_df['Lat'] = account_df['Lat'].fillna(value=0) 
    account_df['Long'] = account_df['Long'].fillna(value=0) 
    account_df.drop(['shipping.city', 'shipping.zip', 'ZipCodeType', 'City', 'State', 'Location', 'LocationType', 'TaxReturnsFiled', 'EstimatedPopulation', 'TotalWages'], axis=1, inplace=True) 
    account_df['relationship.encoded'] = ordinal_feature(account_df, 'relationship')
    account_df['relationship.encoded'] = account_df['relationship.encoded'].fillna(value=0) 
    account_df.drop(['relationship'], axis=1, inplace=True)
    basetime = datetime(1970, 1, 1)
    account_df['first.donated'] = pd.to_datetime(account_df['first.donated'])
    account_df['first.donated'] = pd.to_numeric(account_df['first.donated'].fillna(basetime))
    account_df['first.donated.scaled'] = min_max_scale(account_df, 'first.donated')
    account_df.drop(['first.donated'], axis=1, inplace=True)
    account_df.drop(['billing.city'], axis=1, inplace=True)
    find_na(account_df) 

def extract_pois(df, column): 
    pois = set()
    pattern = r".*?([A-Za-z ]{3,30}).*?" 
    
    for id, row in df.iterrows(): 
        if type(row.loc[column]) == str: 
            for match in re.findall(pattern, row.loc[column], flags=re.DOTALL):        
                if len(match.strip()) > 4 and match.strip()[0].isupper(): 
                    pois.add(match.strip())
    return pois

def extract_composers(df, column): 
    composers = set()
    pattern = r".*?([A-Z][A-Z .]{1,30}).*?" 
    
    for id, row in df.iterrows(): 
        if type(row.loc[column]) == str: 
            for match in re.findall(pattern, row.loc[column], flags=re.DOTALL):        
                if len(match.strip()) > 4: 
                    composers.add(match.strip())
    return composers

def load_concerts(file='data/Concerts.csv'): 
    concerts_df = pd.read_csv(file) 
    concerts_df['set.id'] = concerts_df['season'] + "-s" + concerts_df['set'].astype(str)
    # A deduplicated DF with just the mapping of year/set to composer et al 
    concerts_df.drop_duplicates(subset='set.id', keep='first', inplace=True) 
    concerts_df.set_index('set.id', inplace=True) 
    pois = extract_pois(concerts_df, 'who') 
    composers = extract_composers(concerts_df, 'what') 
    find_na(concerts_df) 

def load_upcoming_concerts(file='data/Concerts_2014-15.csv'):
    upcoming_df = pd.read_csv(file) 
    planned_pois = extract_pois(upcoming_df, 'who') 
    # Need a fixed index on these folks, use list in lieu of the set going forward
    salient_pois = list(planned_pois.intersection(pois))
    planned_composers = extract_composers(upcoming_df, 'what') 
    planned_composers.intersection(composers)

    # The intersection between the composers' pieces that were performed and the planned composers is small, 
    # and has to be correlated manually here short of some NLP magic. Conveniently, though we can rely on 
    # presence of the below strings to indicate historic or planned presence of the associated composer. We
    # perhaps shamefully throw all the BACHs in one bucket here, but I'm already straining the limits of what
    # time will permit here. All other composers will be represented by the absence of any information in the
    # associated one-hot encodings. 
    #
    # This may be the only real feature worth applying from this  the 2014-2015 
    salient_composers = [ 'BACH', 'HAYDN', 'VIVALDI', 'TELEMANN', 'HANDEL' ]

def map_price_level(level): 
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

def load_tickets(file='data/tickets_all.csv'):
    tickets_df = pd.read_csv(file, index_col='account.id') 
    tickets_df['set'] = tickets_df['set'].fillna(value=0) 
    tickets_df['set'] = tickets_df['set'].astype(int)
    tickets_df['set.id'] = tickets_df['season'] + "-s" + tickets_df['set'].astype(str)
    tickets_df = tickets_df.join(concerts_df, on='set.id', how='left', rsuffix='_temp')
    tickets_df['season.ordinal'] = ordinal_feature(tickets_df, 'season')
    loc_df = onehot_feature(tickets_df, 'location') 
    tickets_df = pd.concat([tickets_df, loc_df], axis=1)
    tickets_df['multiple.tix.ordinal'] = ordinal_feature(tickets_df, 'multiple.tickets') 
    tickets_df.drop(['marketing.source', 'season', 'season_temp', 'location', 'location.nan', 'multiple.tickets', 'set.id', 'set_temp', 'location_temp'], axis=1, inplace=True)
    tickets_df['price.level'] = tickets_df['price.level'].fillna(value=0)
    tickets_df['price.level'] = tickets_df['price.level'].apply(map_price_level)

    # Now that we know the overlap, one-hot encode the people of interest
    def find_poi(text, poi): 
        if type(text) == str: 
            return 1 if poi in text else 0
    
    for poi in salient_pois: 
        column = poi.replace(' ', '.')
        tickets_df[column] = [0] * len(tickets_df) 
        tickets_df[column] = tickets_df.apply(lambda x: find_poi(x.who, poi), axis=1)

    for composer in salient_composers: 
        column = composer.replace(' ', '.') 
        tickets_df[column] = [0] * len(tickets_df) 
        tickets_df[column] = tickets_df.apply(lambda x: find_poi(x.what, composer), axis=1)

    tickets_df.drop(['concert.name', 'who', 'what', 'price.level'], axis=1, inplace=True)
    find_na(tickets_df) 

def load_subscriptions(file='data/Subscriptions.csv'):
    
    subs_df = pd.read_csv(file, index_col='account.id')
    
    # Seasons have a relative ordering, encode as ordinals
    subs_df['season.ordinal'] = ordinal_feature(subs_df, 'season') 
    
    # One-hot encode categoricals
    subs_df.rename(columns={'package': 'pkg'}, inplace=True)
    subs_df.rename(columns={'location': 'loc'}, inplace=True)
    subs_df.rename(columns={'section': 'sn'}, inplace=True)
    pkg_df = onehot_feature(subs_df, 'pkg') 
    loc_df = onehot_feature(subs_df, 'loc') 
    sn_df = onehot_feature(subs_df, 'sn') 
    subs_df = pd.concat([subs_df, loc_df], axis=1)
    subs_df = pd.concat([subs_df, pkg_df], axis=1)
    subs_df = pd.concat([subs_df, sn_df], axis=1)

    # Multiple subscriptions have a relative ordering... 
    subs_df['multiple.subs'] = ordinal_feature(subs_df, 'multiple.subs') 
    
    # Price level *seems* to have a relative ordering... 
    subs_df['price.level'] = subs_df['price.level'].fillna(value=0) 
    
    # Season has a relative ordering 
    subs_df['season.ordinal'] = subs_df['season.ordinal'].astype(int) 

    # Remove cruft and check for any residual empty cells 
    subs_df.drop(['season', 'pkg', 'loc', 'sn', 'sn.nan'], axis=1, inplace=True) 
    find_na(subs_df) 

def scale(df, range=(0,1), omit=[]):
    for column in df.columns: 
        if column not in omit: 
            df[column] = min_max_scale(df, column, (0,1))

def predict_subscribers(model, seasons, df, num_prior=1): 
    """
    Try to fit the provided model on the prior season(s) data to predict the current season's subscription renewals. 
    """
    results = [] 
    # Iterate over the seasons, in order, looking to see how we fare at predicting the current year's subscriptions based on the previous
    for season in seasons: 
        first_season = season - num_prior
        next_season = season + 1
        score = 0 
        if first_season in seasons and next_season in seasons: 
            prior_df = df[(df['season.ordinal'] >= first_season) & (df['season.ordinal'] < season)]
            this_df = df[df['season.ordinal'] == season]
            next_df = df[df['season.ordinal'] == next_season] 
            
            # Figure out which accounts from prior year(s) also subscribed this year
            current_subscribers = [1 if index in this_df.index else 0 for index in prior_df.index]
            future_subscribers = [1 if index in next_df.index else 0 for index in this_df.index]

            fit(model, prior_df, current_subscribers)
            
            # Check for utility 
            probs = predict_proba(model, this_df)[:,1]
            score = metrics.roc_auc_score(future_subscribers, probs)
            print(f"Season {season} result: {score:.4f}")
            results.append(score) 
    
    return results

def make_train_test_sets(account_df):
    train_df = train_df.join(account_df, how='left', rsuffix="_acc") 
    X_train = train_df.drop(['label'], axis=1)
    y_train = train_df['label']
    X_test = test_df.join(account_df, how='left')

    return X_train, y_train, X_test 



def build_experiments(): 

    scale(account_df) 
    scale(subs_df, omit=['season.ordinal']) 



def pca2(df): 

    pca = PCA(n_components=2) 
    pca.fit(df) 
    lowD = pca.transform(df) 
    km_model = KMeans(n_clusters=8, random_state=0, n_init="auto")
    km_model.fit(lowD) 

def plot_pca2(df): 
    pca2(account_df)
    subD = account_df.copy()
    subD['cluster'] = km_model.labels_ 
    subD['lowD0'] = lowD.transpose()[0]
    subD['lowD1'] = lowD.transpose()[1]
    train_sub = subD.join(y_train, how='inner') 
    pos_train_sub = train_sub[train_sub['label'] == 1]
    neg_train_sub = train_sub[train_sub['label'] == 0]
    test_sub = subD.join(X_test, how='inner', rsuffix='_') 

    fig = plt.figure() 
    fig.set_size_inches(16,12) 

    # All accounts in low-d space 
    plt.scatter(lowD[:, 0], lowD[:, 1], color='blue') 

    # training - negative class
    plt.scatter(neg_train_sub['lowD0'], neg_train_sub['lowD1'], color='red', marker='.') 

    # test set
    plt.scatter(test_sub['lowD0'], test_sub['lowD1'], marker='.', color='orange') 

    # training - positive class
    plt.scatter(pos_train_sub['lowD0'], pos_train_sub['lowD1'], color='lime', marker='.') 

    for cluster in range(1,km_model.n_clusters): 
        center = km_model.cluster_centers_[cluster] 
        plt.scatter(center[0], center[1], color='yellow', marker='D') 
        plt.annotate(cluster, (center[0], center[1]), bbox=dict(boxstyle="round", fc="0.8"))



@ignore_warnings(category=ConvergenceWarning)
def fit(model, X, y): 
    model.fit(X, y) 
    
def predict(model, X): 
    preds = model.predict(X)
    return preds

def predict_proba(model, X): 
    probs = model.predict_proba(X)
    return probs

def predict_random_proba(X): 
    probs = [random.random() for x in X.iterrows()]
    return probs

def baseline_predictions(): 
    dummy_classifier = 
    fit(dummy_classifier, train_df.index, train_df['label'])
    probs = predict_proba(dummy_classifier, train_df.index) 

    dummy_regressor = DummyRegressor(strategy='constant', constant=0.7) 
    fit(dummy_regressor, train_df.index, train_df['label'])
    probs = predict(dummy_regressor, train_df.index)

    probs = predict_random_proba(train_df) 
    metrics.roc_auc_score(train_df['label'], probs) 
    metrics.RocCurveDisplay.from_predictions(train_df['label'].to_numpy(), probs) 

def logistic_regression(): 

    experiments = \
        # Dummy/baseline
        Pipeline([
            ('Scaler', StandardScaler()), 
            ('Dummy', DummyClassifier(strategy='stratified'))])
    
        # Logistic Regression on accounts
        # TODO: transmute into a dynamicallt constructed pipeline that iterates over the relevant learning methods to identify the best scoring
        lr_model = LogisticRegression(penalty=None) 
        fit(lr_model, X_train, y_train) 
        probs = predict_proba(lr_model, X_train)
        positive_probs = probs[:,1]
        metrics.roc_auc_score(y_train.values, positive_probs) 
        metrics.RocCurveDisplay.from_predictions(y_train.values, positive_probs) 
        probs = predict_proba(lr_model, X_test) 
    
        # Tickets
        # TODO: as with above, rework into a generic solution for modeling and testing tickets
        tickets_df.dropna(inplace=True) 
        train_tickets = tickets_df.join(train_df, how='inner')
        X_train_tickets = train_tickets.drop(['label'], axis=1)
        y_train_tickets = train_tickets[['label']]
        fit(lr_model, X_train_tickets, y_train_tickets) 
        probs = predict_proba(lr_model, X_train_tickets) 
        positive_probs = probs[:,1]
        metrics.roc_auc_score(y_train_tickets.values, positive_probs)
        metrics.RocCurveDisplay.from_predictions(y_train_tickets.values, positive_probs) 

        # Subs
        ... 
        LR on subs 
    

        # Lasso 
        
        lasso_model = Lasso(alpha=0.1) 
        fit(lasso_model, X_train, y_train) 
        probs = predict(lasso_model, X_train)
        metrics.roc_auc_score(y_train.values, probs)  
        metrics.RocCurveDisplay.from_predictions(y_train.values, probs)

        # RF
        
        rf_model = RandomForestClassifier()
        fit(rf_model, X_train, y_train) 
        probs = predict_proba(rf_model, X_train) 
    
        positive_probs = probs[:, 1]
        metrics.roc_auc_score(y_train.values, positive_probs) 
    
        metrics.RocCurveDisplay.from_predictions(y_train.values, positive_probs)
    
        # SVM 
        
    
        # Should we use standardscaler here in lieu of minmaxscaler (see above)? 
        sv_model = SVC(gamma='scale', probability=True)
        fit(sv_model, X_train, y_train) 
        probs = predict_proba(sv_model, X_train) 
        positive_probs = probs[:,1]
        metrics.roc_auc_score(y_train.values, positive_probs) 
        metrics.RocCurveDisplay.from_predictions(y_train.values, positive_probs)
    
        # KNN 
        kn_model = KNeighborsClassifier(n_neighbors=1)
        fit(kn_model, X_train, y_train)
        probs = predict(kn_model, X_train)
        metrics.roc_auc_score(y_train.values, probs) 
        metrics.RocCurveDisplay.from_predictions(y_train.values, probs)


def evaluate_model(model): 

    # TODO: we need a method of evaluating the hyperparameters that can be applied to the models in question
    # 
    GridSearchCV(estimator=SVC(),
             param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})
    

    # END EXPERIMENT DEFS
    
def evaluate_dataset(df, pipelines)
    """
    Dataset evaluation routing to facilitate experimenting with different features. Accepts a
    dataframe to be tested, splits the data, evaluate eaech of the provided pipelines, reporting 
    the result. 
    """
    # TODO, any train/test split must be over a single dataset ... we should perhaps forgo the predictions on anything
    # but the CBBDF, as it represents our best idea about how to predict this stuff... we have no strategy for an 
    # bilayer optimizer, so just experiment with different formulations of the KBBDF, yeah? 
    # Pipeline construction with help from SKLearn docs (https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
    X = df -- drop labels
    y = df['label']
    X_train, X-test, y_train, y_test = train_test_split(X, y, random_state=0)
    for i, experiment in enumerate(experiments): 
        print("Running experiment")
        experiment.fit(X_train, y_train).score(X_test, y_test)

def generate_submission(model, test): 
    # Export 
    submit_df = pd.DataFrame() 
    submit_df['ID'] = X_test.index
    submit_df['Predicted'] = probs[:,1]
    submit_df.to_csv('submissions/09march_0949.csv', index=False)


def main(): 

    # Import train and test indices
    load_train()
    load_test()

    # Import, clean and prepare reference data 
    load_concerts()    
    load_zip()
    load_upcoming_concerts

    # Import, clean and prepare predictors
    load_accounts()
    load_tickets()
    load_subscriptions()

    # Train any tributary models or inputs to the CBBDF
    scores = predict_subscribers(lr_model, seasons, subs_df, 1) 

    # Manipulate data, testing each version to see what the outcome is


    # After selecting the best models, using a voting, max, something routine to join predictions
    
    