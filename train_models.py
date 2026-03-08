import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

df_games = pd.read_csv('video_games_cleaned.csv')

X1 = df_games[['Platform', 'Genre', 'Critic_Score', 'User_Score']]
y1 = df_games['Global_Sales']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['Critic_Score', 'User_Score']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Platform', 'Genre'])
])

model_lr = LinearRegression()
model_dt = DecisionTreeRegressor(random_state=42)
model_rf = RandomForestRegressor(n_estimators=50, random_state=42)

ensemble = VotingRegressor(estimators=[('lr', model_lr), ('dt', model_dt), ('rf', model_rf)])

games_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', ensemble)])

games_pipeline.fit(X1_train, y1_train)
score1 = games_pipeline.score(X1_test, y1_test)
joblib.dump(games_pipeline, 'games_model.pkl')

df_reviews = pd.read_csv('clothing_reviews_cleaned.csv')

df_reviews_sampled = df_reviews.sample(n=10000, random_state=42)
X2 = df_reviews_sampled['Review_Cleaned']
y2 = df_reviews_sampled['Recommended IND']

vectorizer = TfidfVectorizer(max_features=5000)
X2_vec = vectorizer.fit_transform(X2)

X2_train, X2_test, y2_train, y2_test = train_test_split(X2_vec, y2, test_size=0.2, random_state=42)

nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)

nn_model.fit(X2_train, y2_train)
score2 = nn_model.score(X2_test, y2_test)
joblib.dump(nn_model, 'clothing_nn_model.pkl')
joblib.dump(vectorizer, 'clothing_vectorizer.pkl')