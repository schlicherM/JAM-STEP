import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, t, shapiro, wilcoxon, ttest_rel
from sklearn.model_selection import KFold
import numpy as np
import spacy 
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
#from spellchecker import SpellChecker
import torch
torch.manual_seed(42)
np.random.seed(42) 
torch.cuda.manual_seed_all(42)  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


spell = SpellChecker(language="de")

target_columns = ['MDBF_Valence_Score', 'MDBF_Arousal_Score', 'MDBF_Calmness_Score', 'PSS4_Score']
nlp = spacy.load('de_core_news_sm')

def load_data():
    """
    Loads JAM-STEP dataset

    Returns:
        data (pd.DataFrame): The loaded dataset.
    """

    data = pd.read_csv(
        'data/evaluated_data.csv',
        encoding='utf-8',
        delimiter=';',
        decimal='.',
        quotechar='"',
        parse_dates=['STARTED'],  
        header=0,
        na_values=['']
    )

    data = data.dropna(subset=target_columns + ['TEXT'])
    data = data.sort_values('CASE')
    print(f'Number of datapoints: {data.shape[0]}')
    return data

def split_by_serial(data, target_column, num_bins = 7):
    """
    Splits data by SERIAL into train and test sets and ensure that the target distribution is similar in both sets.

    Parameters:
        data (pd.DataFrame): The input data.
        target_column (str): The target column to stratify by.
        num_bins (int): The number of bins to use for stratification.

    Returns:
        train_data (pd.DataFrame): The training data.
        test_data (pd.DataFrame): The testing data.
    """
    serial_means = data.groupby('SERIAL')[target_column].mean()
    serial_means_binned = pd.qcut(serial_means, q=num_bins, labels=False, duplicates="drop")

    unique_serials = serial_means.index
    stratify_values = serial_means_binned

    data = data.copy()
    data.loc[:, 'stratify_values'] = data['SERIAL'].map(stratify_values)

    # Stratify split SERIALs based on their binned mean target values
    train_serials, test_serials = train_test_split(
        unique_serials,
        test_size=0.2,
        random_state=42,
        stratify=stratify_values
    )

    train_data = data[data['SERIAL'].isin(train_serials)].copy()
    test_data = data[data['SERIAL'].isin(test_serials)].copy()

    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    assert len(set(train_data['SERIAL']).intersection(set(test_data['SERIAL']))) == 0, "Train and test serials overlap"

    train_data = train_data[['SERIAL','TEXT', target_column]]
    test_data = test_data[['SERIAL','TEXT', target_column]]

    return train_data, test_data 

def preprocess_text(text):
    """
    Tokenize and lemmatize the text and remove stop words for German.
    
    Parameters:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    doc = nlp(text)

    tokens = [
        token.lemma_.lower()  
        for token in doc
        if token.is_alpha                  
        and not token.is_stop          
    ]
    corrected_tokens = [spell.correction(token) if token in spell else token for token in tokens]

    return ' '.join(corrected_tokens)

def normalize_target(data, target):
    """
    Normalize the target columns
    
    Parameters:
        data (pd.Series): The target values to normalize.
        target (str): The target name.

    Returns:
        pd.Series: The normalized target values.
    """
    if target == 'MDBF_Valence_Score':
        return (data - 1) / 6
    elif target == 'MDBF_Arousal_Score':
        return (data - 1) / 6
    elif target == 'MDBF_Calmness_Score':
        return (data - 1) / 6
    elif target == 'PSS4_Score':
        return data / 16

def inverse_normalize_target(data, target):
    """
    Inverse normalize the target columns
    
    Parameters:
        data (pd.Series): The target values to inverse normalize.
        target (str): The target name.
    
    Returns:
        pd.Series: The inverse normalized target values.
    """
    if target == 'MDBF_Valence_Score':
        return data * 6 + 1
    elif target == 'MDBF_Arousal_Score':
        return data * 6 + 1
    elif target == 'MDBF_Calmness_Score':
        return data * 6 + 1
    elif target == 'PSS4_Score':
        return data * 16
    
def confidence_interval(data, confidence=0.95):
    """
    Calculate the confidence interval for the given data.

    Parameters:
        data (np.array): The input data.
        confidence (float): The confidence level.
    
    Returns:
        tuple: The mean value, lower bound, and upper bound of the confidence interval.
    """

    n = len(data)
    mean_val = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return mean_val, mean_val - h, mean_val + h

def perform_linear_regression(train_set, test_set, target_column, dataset='', pca_components=2, testmode=False):
    """
    Perform Linear Regression with Bag of Word embedding using 5-fold CV.
    Saves plots for PCA components and testing predictions.

    Parameters:
        train_set (pd.DataFrame): The training data.
        test_set (pd.DataFrame): The testing data.
        target_column (str): The target column to predict.
        dataset (str): The name of the dataset.
        pca_components (int): The number of PCA components to use.
        testmode (bool): Whether to run in test mode.

    Returns:
        None
    """
    train_set['TEXT'] = train_set['TEXT'].apply(preprocess_text)
    test_set['TEXT'] = test_set['TEXT'].apply(preprocess_text)
    train_set[target_column] = normalize_target(train_set[target_column], target_column).copy()
    test_set[target_column] = normalize_target(test_set[target_column], target_column).copy()

    # Vectorize the text data using Bag of Words
    vectorizer = CountVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(train_set['TEXT'])
    X_test = vectorizer.transform(test_set['TEXT'])

    y_train = train_set[target_column]
    y_test = test_set[target_column]

    # Set up 5-fold cross-validation split by 'SERIAL'
    unique_ids = train_set['SERIAL'].unique()
    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train)
    eigenvalues = pca.explained_variance_
    n_components_kaiser = np.sum(eigenvalues > 1)
    print(f"Number of components based on Kaiser criterion: {n_components_kaiser}")
    print(np.cumsum(pca.explained_variance_ratio_))

    plt.figure(figsize=(15, 13))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
        np.cumsum(pca.explained_variance_ratio_), color=sns.color_palette("Blues")[4], linewidth=8)
    plt.axvline(x=27, color=sns.color_palette("Reds")[3], linewidth=4, linestyle='--')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.ylim(0, 1)
    plt.xlabel('Number of Components', fontsize=52)
    plt.ylabel('Explained Variance', fontsize=52)
    plt.title(f'{dataset}', fontsize=52)
    plt.xticks(fontsize=46)
    plt.yticks(fontsize=46)
    plt.savefig(f"plots/baseline/{dataset}_{target_column}_pca_linear_regression.png")
    plt.close()

    if not testmode:


        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        all_val_rmse = []
        all_val_spearman_rho = []
        all_train_rmse = []
        all_train_spearman_rho = []

        # Train the model with 5-fold cross-validation
        for train_index, val_index in kf.split(unique_ids):
            train_ids = unique_ids[train_index]
            val_ids = unique_ids[val_index]
            
            X_train_cv = X_train_pca[train_set['SERIAL'].isin(train_ids)]
            y_train_cv = y_train[train_set['SERIAL'].isin(train_ids)]
            X_val_cv = X_train_pca[train_set['SERIAL'].isin(val_ids)]
            y_val_cv = y_train[train_set['SERIAL'].isin(val_ids)]

            model = LinearRegression()
            model.fit(X_train_cv, y_train_cv)

            y_pred = model.predict(X_val_cv)
            all_val_rmse.append(np.sqrt(mean_squared_error(y_val_cv, y_pred)))
            all_val_spearman_rho.append(spearmanr(y_val_cv, y_pred)[0])

            y_train_pred = model.predict(X_train_cv)
            all_train_rmse.append(np.sqrt(mean_squared_error(y_train_cv, y_train_pred)))
            all_train_spearman_rho.append(spearmanr(y_train_cv, y_train_pred)[0])

        avg_rmse, lower_rmse, upper_rmse = confidence_interval(all_val_rmse)
        avg_spearman_rho, lower_rho, upper_rho = confidence_interval(all_val_spearman_rho)
        avg_train_rmse, lower_train_rmse, upper_train_rmse = confidence_interval(all_train_rmse)
        avg_train_spearman_rho, lower_train_rho, upper_train_rho = confidence_interval(all_train_spearman_rho)

        print(f"Results for {target_column}:")
        print(f"Train RMSE: {avg_train_rmse:.3f} (95% CI: [{lower_train_rmse:.3f}, {upper_train_rmse:.3f}])")
        print(f"Train Spearman’s rho: {avg_train_spearman_rho:.3f} (95% CI: [{lower_train_rho:.3f}, {upper_train_rho:.3f}])")
        print(f"Val RMSE: {avg_rmse:.3f} (95% CI: [{lower_rmse:.3f}, {upper_rmse:.3f}])")
        print(f"Val Spearman’s rho: {avg_spearman_rho:.3f} (95% CI: [{lower_rho:.3f}, {upper_rho:.3f}])")
        
    if testmode:
        X_test_pca = pca.transform(X_test)
        model = LinearRegression()
        model.fit(X_train_pca, y_train)
        test_predictions = model.predict(X_test_pca)
        all_actual_values = y_test.values
        print(f"Linear Regression Test RMSE: {np.sqrt(mean_squared_error(all_actual_values, test_predictions)):.3f}")
        print(f"Linear Regression Test Spearman’s rho: {spearmanr(all_actual_values, test_predictions)[0]:.3f}")

        results = pd.DataFrame({
            'Actual': inverse_normalize_target(pd.Series(all_actual_values), target_column).values,
            'Predicted': inverse_normalize_target(pd.Series(test_predictions), target_column).values
        })
        
        results = results.sort_values('Actual').reset_index(drop=True)
        all_actual_values = results['Actual'].values
        test_predictions = results['Predicted'].values

        plt.figure(figsize=(10, 6))
        jitter = np.random.uniform(-0.02, 0.02, size=len(all_actual_values))

        sns.scatterplot(
            x=range(len(all_actual_values)),
            y=all_actual_values + jitter,
            label="Ground Truth",
            alpha=0.6,
            s=10,
            color=sns.color_palette("Blues")[4]
        )

        plt.plot(range(len(all_actual_values)), test_predictions, label="Linear Regression Predictions", color=sns.color_palette("Reds")[3], linewidth=2)

        plt.xlabel("Sorted Sample Index", fontsize=14)
        plt.ylabel("Questionnaire Score", fontsize=14)
        plt.legend(fontsize=12)

        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        plt.tight_layout()

        plt.savefig(f"plots/baseline/{dataset}_{target_column}_linear_regression_average_sorted_ground_truth.png")
        plt.close()

def german_bert_embedding(text, model, tokenizer, max_length=128):
    """
    Encode text using BERT for German and return the embeddings.
    
    Parameters:
        text (str): The input text to encode.
        model (BertModel): The BERT model.
        tokenizer (BertTokenizer): The BERT tokenizer.
        max_length (int): The maximum length of the input text.

    Returns:
        np.array: The BERT embeddings for the input text.
    """
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  
    return embeddings



def perform_random_forest_bert_embedding(train_set, test_set, target_column, dataset='',pca_components=2, testmode=True):
    """
    Perform Random Forest regression using BERT embeddings, PCA, and GridSearchCV with 5-fold CV.
    Saves plots for PCA components and testing predictions.

    Parameters:
        train_set (pd.DataFrame): The training data.
        test_set (pd.DataFrame): The testing data.
        target_column (str): The target column to predict.
        dataset (str): The name of the dataset.
        pca_components (int): The number of PCA components to use.
        testmode (bool): Whether to run in test mode.

    Returns:
        None
    """
    model_name = "bert-base-german-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    train_set[target_column] = normalize_target(train_set[target_column], target_column).copy()
    test_set[target_column] = normalize_target(test_set[target_column], target_column).copy()

    X_train = german_bert_embedding(train_set['TEXT'].tolist(), model, tokenizer)
    X_test = german_bert_embedding(test_set['TEXT'].tolist(), model, tokenizer)

    if not os.path.exists('data/embeddings'):
        os.makedirs('data/embeddings')
    np.save(f"data/embeddings/{target_column}_X_train.npy", X_train)
    np.save(f"data/embeddings/{target_column}_X_test.npy", X_test)

    y_train = train_set[target_column]
    y_test = test_set[target_column]

    pca = PCA(n_components=pca_components) 
    X_train_pca = pca.fit_transform(X_train)
    eigenvalues = pca.explained_variance_
    n_components_kaiser = np.sum(eigenvalues > 1)
    print(f"Number of components based on Kaiser criterion: {n_components_kaiser}")
    print(np.cumsum(pca.explained_variance_ratio_))
    plt.figure(figsize=(15, 13))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
        np.cumsum(pca.explained_variance_ratio_), color=sns.color_palette("Blues")[4], linewidth=8)
    plt.axvline(x=85, color=sns.color_palette("Reds")[3], linewidth=4, linestyle='--')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.ylim(0, 1)
    plt.xlabel('Number of Components', fontsize=52)
    plt.ylabel('Explained Variance', fontsize=52)
    plt.title(f'Type', fontsize=52)
    plt.xticks(fontsize=46)
    plt.yticks(fontsize=46)
    plt.savefig(f"plots/baseline/{dataset}_{target_column}_pca_random_forest.png")
    plt.close()

    if not testmode:
        unique_ids = train_set['SERIAL'].unique()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        all_val_rmse = []
        all_val_spearman_rho = []
        all_train_rmse = []
        all_train_spearman_rho = []

        print('...training model with cross-validation...')

        for train_index, val_index in kf.split(unique_ids):
            print('...next fold...')
            train_ids = unique_ids[train_index]
            val_ids = unique_ids[val_index]
            
            X_train_cv = X_train_pca[train_set['SERIAL'].isin(train_ids)]
            y_train_cv = y_train[train_set['SERIAL'].isin(train_ids)]
            X_val_cv = X_train_pca[train_set['SERIAL'].isin(val_ids)]
            y_val_cv = y_train[train_set['SERIAL'].isin(val_ids)]
            
            param_grid = {
                'n_estimators': [100, 300, 500, 1000],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [5, 10, 20],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
            
            grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), 
                                    param_grid=param_grid, 
                                    cv=3, 
                                    n_jobs=-1, 
                                    scoring='neg_mean_squared_error')
            
            grid_search.fit(X_train_cv, y_train_cv)
            best_rf_model = grid_search.best_estimator_

            print(f"Best hyperparameters: {grid_search.best_params_}")
            best_rf_model = RandomForestRegressor(random_state=42, n_estimators=500, max_depth=10, min_samples_split=5, min_samples_leaf=10, max_features=None, bootstrap=True)
            best_rf_model.fit(X_train_cv, y_train_cv)
            y_pred = best_rf_model.predict(X_val_cv)
            all_val_rmse.append(np.sqrt(mean_squared_error(y_val_cv, y_pred)))
            all_val_spearman_rho.append(spearmanr(y_val_cv, y_pred)[0])

            y_train_pred = best_rf_model.predict(X_train_cv)
            all_train_rmse.append(np.sqrt(mean_squared_error(y_train_cv, y_train_pred)))
            all_train_spearman_rho.append(spearmanr(y_train_cv, y_train_pred)[0])

        avg_rmse, lower_rmse, upper_rmse = confidence_interval(all_val_rmse)
        avg_spearman_rho, lower_rho, upper_rho = confidence_interval(all_val_spearman_rho)
        avg_train_rmse, lower_train_rmse, upper_train_rmse = confidence_interval(all_train_rmse)
        avg_train_spearman_rho, lower_train_rho, upper_train_rho = confidence_interval(all_train_spearman_rho)

        print(f"Results for {target_column}:")
        print(f"Train RMSE: {avg_train_rmse:.3f} (95% CI: [{lower_train_rmse:.3f}, {upper_train_rmse:.3f}])")
        print(f"Train Spearman’s rho: {avg_train_spearman_rho:.3f} (95% CI: [{lower_train_rho:.3f}, {upper_train_rho:.3f}])")
        print(f"Val RMSE: {avg_rmse:.3f} (95% CI: [{lower_rmse:.3f}, {upper_rmse:.3f}])")
        print(f"Val Spearman’s rho: {avg_spearman_rho:.3f} (95% CI: [{lower_rho:.3f}, {upper_rho:.3f}])")

    if testmode:
        X_test_pca = pca.transform(X_test)


        best_rf_model = RandomForestRegressor(random_state=42, n_estimators=500, max_depth=10, min_samples_split=5, min_samples_leaf=10, max_features=None, bootstrap=True)
        best_rf_model.fit(X_train_pca, y_train)

        test_predictions = best_rf_model.predict(X_test_pca)
        all_actual_values = y_test.values
        print(f"Random Forest Test RMSE: {np.sqrt(mean_squared_error(all_actual_values, test_predictions)):.3f}")
        print(f"Random Forest Test Spearman’s rho: {spearmanr(all_actual_values, test_predictions)[0]:.3f}")

        results = pd.DataFrame({
            'Actual': inverse_normalize_target(pd.Series(all_actual_values), target_column).values,
            'Predicted': inverse_normalize_target(pd.Series(test_predictions), target_column).values
        })

        results = results.sort_values('Actual').reset_index(drop=True)
        all_actual_values = results['Actual'].values
        test_predictions = results['Predicted'].values

        plt.figure(figsize=(10, 6))
        jitter = np.random.uniform(-0.02, 0.02, size=len(all_actual_values))

        sns.scatterplot(
            x=range(len(all_actual_values)),
            y=all_actual_values + jitter,
            label="Ground Truth",
            alpha=0.6,
            s=10,
            color=sns.color_palette("Blues")[4]
        )

        plt.plot(range(len(all_actual_values)), test_predictions, label="Random Forest Predictions", color=sns.color_palette("Reds")[3], linewidth=2)

        plt.xlabel("Sorted Sample Index", fontsize=14)
        plt.ylabel("Questionnaire Score", fontsize=14)
        plt.legend(fontsize=12)

        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        plt.tight_layout()

        plt.savefig(f"plots/baseline/{dataset}_{target_column}_random_forest_average_sorted_ground_truth.png")
        plt.close()

def calculate_baseline_rmse(data, target_columns=target_columns, dataset=''):
    """
    Calculate and print the RMSE for a baseline predictor that predicts the mean
    for each target column in the global variable `target_columns`.
    
    Parameters:
        dataframe (pd.DataFrame): The input data containing the target columns.

    Returns:
        None
    """
    data = data.copy()
    
    for column in target_columns:
        if column not in data:
            print(f"Column {column} not found in the DataFrame.")
            continue

        data[column] = normalize_target(data[column], column)
        mean_value = data[column].mean()

        rmse = np.sqrt(np.mean((data[column] - mean_value) ** 2))
        print(f"Baseline RMSE for {column}: {rmse:.3f}")

        data['Baseline_Prediction'] = mean_value
        data = data.sort_values(column)

        plt.figure(figsize=(10, 6))
        jitter = np.random.uniform(-0.02, 0.02, size=len(data[column]))
        
        sns.scatterplot(
            x=range(len(data[column])), 
            y=data[column] + jitter, 
            label="Ground Truth", 
            alpha=0.6, 
            s=10, 
            color=sns.color_palette("Blues")[4]
        )

        plt.plot(range(len(data[column])), [mean_value] * len(data), label="Mean", color=sns.color_palette("Reds")[3], linewidth=2)

        plt.xlabel("Sorted Sample Index", fontsize=14)
        plt.ylabel("Normalized Score", fontsize=14)
        plt.legend(fontsize=12)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        plt.tight_layout()

        plt.savefig(f"plots/baseline/{dataset}_{column}_mean.png")
        plt.close()


class JAMSTEPDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length=128):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {key: encoding[key].squeeze(0) for key in encoding}
        item['labels'] = torch.tensor(self.targets[idx], dtype=torch.float)
        return item

def fine_tune_bert_for_regression(train_set, test_set, target_column, model_name="bert-base-german-cased", epochs=3, batch_size=8, learning_rate=5e-5, max_length=128, dataset_name='', testmode=False):
    """
    Fine-tune BERT for regression and make predictions using 5-fold cross-validation.
    
    Parameters:
        train_set (pd.DataFrame): The training data.
        test_set (pd.DataFrame): The testing data.
        target_column (str): The target column to predict.
        model_name (str): The name of the BERT model.
        epochs (int): The number of epochs to train for.
        batch_size (int): The batch size.
        learning_rate (float): The learning rate.
        max_length (int): The maximum length of the input text.
        dataset_name (str): The name of the dataset.
        testmode (bool): Whether to run in test mode.

    Returns:
        None
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)  # For regression, num_labels = 1
    if not testmode:
        
        
        print('Preprocessing...')
        train_losses = []
        val_losses = []

        best_val_loss = float('inf')  
        patience_counter = 0 

        train_data, val_data = split_by_serial(train_set, target_column)

        train_texts = train_data['TEXT'].tolist()
        train_labels = train_data[target_column].values
        val_texts = val_data['TEXT'].tolist()
        val_labels = val_data[target_column].values

        train_labels = normalize_target(train_labels, target_column)
        val_labels = normalize_target(val_labels, target_column)
        

        train_dataset = JAMSTEPDataset(train_texts, train_labels, tokenizer, max_length)
        val_dataset = JAMSTEPDataset(val_texts, val_labels, tokenizer, max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = torch.nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            train_loss = 0

            train_loss = 0  
            total_samples = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                inputs = {key: batch[key].to(model.device) for key in batch if key != 'labels'}
                labels = batch['labels'].to(model.device)
                
                outputs = model(**inputs)
                logits = outputs.logits.squeeze(-1) 
                loss = criterion(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item() * len(labels)  
                total_samples += len(labels)

            mse_epoch = train_loss / total_samples
            rmse_epoch = np.sqrt(mse_epoch)
            train_losses.append(rmse_epoch)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = {key: batch[key].to(model.device) for key in batch if key != 'labels'}
                    labels = batch['labels'].to(model.device)
                    
                    outputs = model(**inputs)
                    logits = outputs.logits.squeeze(-1)
                    
                    loss = criterion(logits, labels)  
                    val_loss += loss.item() * len(labels)  
                    total_samples += len(labels)

            mse_epoch = val_loss / total_samples
            rmse_epoch = np.sqrt(mse_epoch)
            val_losses.append(rmse_epoch)


            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]} | Val Loss: {val_losses[-1]}")

            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                patience_counter = 0
                best_state = model.state_dict()
                print(f'Best epoch: {epoch+1}')
            else:
                patience_counter += 1

            if patience_counter >= 20:
                print("Early stopping...")
                num_epochs = epoch + 1
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), train_losses, label='Average Train Loss', color=sns.color_palette("Blues")[4], marker='o')
        plt.plot(range(1, epochs + 1), val_losses, label='Average Val Loss', color=sns.color_palette("Reds")[3], marker='x')
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.legend()
        plt.xticks(range(1, epoch + 1))
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.tight_layout()
        
        plt.savefig(f"plots/baseline/{dataset_name}_{target_column}_bert_regression_loss_plot.png")
        plt.close()
        
        if not os.path.exists('models/baseline/models'):
            os.makedirs('models/baseline/models')
        torch.save(best_state, f"models/baseline/models/{dataset_name}_{target_column}_bert_regression_best_model.pth")

    if testmode:
        model.load_state_dict(torch.load(f"models/baseline/models/{dataset_name}_{target_column}_bert_regression_best_model.pth", weights_only=True))
    else:
        model.load_state_dict(best_state)

    if testmode:
        test_texts = test_set['TEXT'].tolist()
        test_labels = test_set[target_column].values
        test_labels = normalize_target(test_labels, target_column)
        

        model.eval()
        all_preds = []
        all_actual = []
        with torch.no_grad():
            test_dataset = JAMSTEPDataset(test_texts, test_labels, tokenizer, max_length)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            for batch in test_loader:
                inputs = {key: batch[key].to(model.device) for key in batch if key != 'labels'}
                labels = batch['labels'].to(model.device)
                
                outputs = model(**inputs)
                logits = outputs.logits.squeeze(-1)
                all_preds.extend(logits.cpu().numpy())
                all_actual.extend(labels.cpu().numpy())

        rmse = np.sqrt(mean_squared_error(all_actual, all_preds))
        spearman_corr = spearmanr(all_actual, all_preds)[0]
        print(f"Test RMSE: {rmse:.3f}")
        print(f"Test Spearman’s rho: {spearman_corr:.3f}")
        
        results = pd.DataFrame({
            'Actual': all_actual,
            'Predicted': all_preds
        })
        
        results = results.sort_values('Actual').reset_index(drop=True)
        all_actual_values = inverse_normalize_target(pd.Series(results['Actual']), target_column).values
        test_predictions = inverse_normalize_target(pd.Series(results['Predicted']), target_column).values

        if dataset_name == 'expert':
            expert_predictions = {
                "CASE": [
                    701, 1787, 707, 1087, 1414, 2351, 1398, 1769, 1606, 2388,
                    2013, 1683, 1328, 2912, 1506, 2840, 1205, 1047, 861, 1735,
                    1229, 2466, 2157, 1277
                ],
                "MDBF_Valence_Score": [
                    3.5, 3.0, 2.8, 3.2, 2.5, 3.3, 2.7, 3.5, 3.9, 5.4,
                    6.1, 5.1, 5.2, 5.4, 4.9, 4.9, 5.3, 3.0, 5.3, 3.6,
                    5.5, 5.8, 5.7, 5.6
                ]
            }

            expert_data = pd.DataFrame(expert_predictions)

            model_errors = np.abs(np.array(test_predictions) - np.array(all_actual_values))
            expert_errors = np.abs(np.array(expert_data['MDBF_Valence_Score']) - np.array(all_actual_values))
            closeness_model_expert = abs(np.array(test_predictions) - np.array(expert_data['MDBF_Valence_Score']))

            model_normality = shapiro(model_errors)
            expert_normality = shapiro(expert_errors)
            closeness_normality = shapiro(closeness_model_expert)

            print("Model Errors Normality Test:", model_normality)
            print("Expert Errors Normality Test:", expert_normality)
            print("Closeness Model-Expert Normality Test:", closeness_normality)
            model_error_mean = np.mean(model_errors)
            expert_error_mean = np.mean(expert_errors)
            closeness_mean = np.mean(closeness_model_expert)

            print(f"Model Mean Error: {model_error_mean:.4f}")
            print(f"Expert Mean Error: {expert_error_mean:.4f}")
            print(f"Closeness Mean: {closeness_mean:.4f}")

            w_stat, p_value = wilcoxon(model_errors, expert_errors)
            print(f"Wilcoxon test Model/Expert Error: w-statistic={w_stat}, p-value={p_value}")

            w_stat, p_value = wilcoxon(model_errors, closeness_model_expert)
            print(f"Wilcoxon test Model Error / Closeness: w-statistic={w_stat}, p-value={p_value}")

            w_stat, p_value = wilcoxon(expert_errors, closeness_model_expert)
            print(f"Wilcoxon test Expert Error / Closeness: w-statistic={w_stat}, p-value={p_value}")
            

        plt.figure(figsize=(10, 6))
        jitter = np.random.uniform(-0.02, 0.02, size=len(all_actual_values))

        sns.scatterplot(
            x=range(len(all_actual_values)),
            y=all_actual_values + jitter,
            label="Ground Truth",
            alpha=0.6,
            s=50,
            color=sns.color_palette("Blues")[4]
        )

        plt.plot(range(len(all_actual_values)), test_predictions, label="BERT Predictions", color=sns.color_palette("Reds")[3], linewidth=2)

        if dataset_name == 'expert':
            plt.plot(
                range(len(expert_data)),
                expert_data['MDBF_Valence_Score'],
                label="Expert Predictions",
                color="gray",
                linewidth=2,
                linestyle='--'
            )

        plt.xlabel("Sorted Sample Index", fontsize=14)
        plt.ylabel("Questionnaire Score", fontsize=14)
        plt.legend(fontsize=12)

        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        plt.tight_layout()

        plt.savefig(f"plots/baseline/{dataset_name}_{target_column}_bert_regression_predictions.png")
        plt.close()


if not os.path.exists('plots/baseline'):
    os.makedirs('plots/baseline')

def main(target_column = 'MDBF_Valence_Score', dataset='all', models=['mean', 'linear', 'random_forest', 'bert'], testmode=False,):
    data = load_data()
    if dataset == 'speak':
        print("Results for Speak:")
        speak_data = data[data['GROUP'] == 'speak']
        train_data, test_data = split_by_serial(speak_data, target_column=target_column)
    elif dataset == 'type':
        print("Results for Type:")
        type_data = data[data['GROUP'] == 'type']
        train_data, test_data = split_by_serial(type_data, target_column=target_column)
    elif dataset == 'expert':
        print("Results for Expert:")
        expert_ids = [1277,2351,701,1506,1229,1769,1787,1087,2912,1414,861,1606,2840,1735,2466,1205,1398,1683,707,2013,2157,2388,1047,1328]
        test_data = data[data['CASE'].isin(expert_ids)]
        ex_teset_ids = test_data['SERIAL'].unique()
        train_data = data[~data['SERIAL'].isin(ex_teset_ids)]
    else:
        print("Results for whole data")
        train_data, test_data = split_by_serial(data, target_column=target_column)
        
    if 'mean' in models:
        calculate_baseline_rmse(test_data,target_columns=[target_column], dataset=dataset)
    if 'linear' in models:
        perform_linear_regression(train_data, test_data, target_column, dataset=dataset, pca_components=150, testmode=testmode)
    if 'random_forest' in models:
        perform_random_forest_bert_embedding(train_data, test_data, target_column, dataset=dataset, pca_components=150, testmode=testmode)
    if 'bert' in models:
        fine_tune_bert_for_regression(train_data, 
                                    test_data, 
                                    target_column=target_column, 
                                    model_name="bert-base-german-cased", 
                                    epochs=15, 
                                    batch_size=16, 
                                    max_length=128, 
                                    dataset_name=dataset, 
                                    learning_rate=1e-5, 
                                    testmode=testmode)
        
# possible datasets: 'all', 'speak', 'type', 'expert'        
main(target_column = 'MDBF_Valence_Score', dataset='type', models=['random_forest'], testmode=True)


