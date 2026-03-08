import pandas as pd
import numpy as np
import re

try:
    df_games = pd.read_csv('video_games.csv')

    if 'User_Score' in df_games.columns:
        df_games['User_Score'] = df_games['User_Score'].replace('tbd', np.nan).astype(float)

    if 'Critic_Score' in df_games.columns:
        df_games['Critic_Score'] = df_games['Critic_Score'].fillna(df_games['Critic_Score'].median())
    if 'User_Score' in df_games.columns:
        df_games['User_Score'] = df_games['User_Score'].fillna(df_games['User_Score'].median())

    df_games = df_games.dropna(subset=['Name', 'Genre', 'Platform', 'Global_Sales'])

    df_games.to_csv('video_games_cleaned.csv', index=False)
    print("ทำความสะอาด Dataset 1 เสร็จสิ้น ได้ไฟล์ 'video_games_cleaned.csv'")
except FileNotFoundError:
    print("ไม่พบไฟล์ video_games.csv กรุณาตรวจสอบชื่อไฟล์อีกครั้ง")

try:
    df_reviews = pd.read_csv('clothing_reviews.csv')

    df_reviews = df_reviews.dropna(subset=['Review Text', 'Recommended IND'])

    def clean_text(text):
        text = str(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = ' '.join(text.split())
        return text.lower()

    df_reviews['Review_Cleaned'] = df_reviews['Review Text'].apply(clean_text)

    df_reviews = df_reviews[df_reviews['Review_Cleaned'] != ""]

    df_reviews.to_csv('clothing_reviews_cleaned.csv', index=False)
    print("ทำความสะอาด Dataset 2 เสร็จสิ้น ได้ไฟล์ 'clothing_reviews_cleaned.csv'")
except FileNotFoundError:
    print("ไม่พบไฟล์ clothing_reviews.csv กรุณาตรวจสอบชื่อไฟล์อีกครั้ง")