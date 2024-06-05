import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from translate import Translator
import joblib
import random

# Modeli yükle
pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

# Duygular ve ilgili emojiler
duygular_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😱", "joy": "😂", "neutral": "😐",
    "sadness": "😔", "shame": "😳", "surprise": "😮"
}


duygular_listesi = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "shame", "surprise"]

# Duyguların Türkçe karşılıkları
duygular_tr_dict = {
    "anger": "Kızgınlık", "disgust": "Tiksinme", "fear": "Korkmak", "joy": "Neşe", "neutral": "Etkisiz",
    "sadness": "Üzüntü", "shame": "Utanç", "surprise": "Sürpriz"
}

# Oturum durumu değişkenlerini başlat
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = random.choice(duygular_listesi)
if 'attempts' not in st.session_state:
    st.session_state.attempts = 3
if 'scores' not in st.session_state:
    st.session_state.scores = {emotion: None for emotion in duygular_listesi}
if 'completed' not in st.session_state:
    st.session_state.completed = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_olasilik' not in st.session_state:
    st.session_state.last_olasilik = None
if 'last_cevrilen_metin' not in st.session_state:
    st.session_state.last_cevrilen_metin = None
if 'rerun' not in st.session_state:
    st.session_state.rerun = False

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def sifirla_oyun():
    st.session_state.current_emotion = random.choice(duygular_listesi)
    st.session_state.attempts = 3
    st.session_state.scores = {emotion: None for emotion in duygular_listesi}
    st.session_state.completed = []
    st.session_state.last_prediction = None
    st.session_state.last_olasilik = None
    st.session_state.last_cevrilen_metin = None
    st.session_state.rerun = True

def translate(input):
    translator = Translator(to_lang="tr", from_lang="en")
    cevrilen_metin = translator.translate(input)
    return cevrilen_metin

def main():
    st.set_page_config(page_title="Metin Duygu Tespiti", page_icon="🌟")
    
    st.markdown(
        """
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            color: white;
            background-color: #007bff;
        }
        .stTextArea>label {
            color: #007bff;
        }
        .stDataFrameContainer {
            max-height: 200px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("Metin Duygu Tespiti 🌟")
    st.subheader("Metinlerdeki Duyguları Tespit Et")

    if st.session_state.rerun:
        st.session_state.rerun = False
        st.experimental_rerun()

    if len(st.session_state.completed) < len(duygular_listesi):
        ceviri = duygular_tr_dict[st.session_state.current_emotion]
        st.write(f"Lütfen şu duyguyu ifade eden bir metin yazın: {ceviri}")
        st.write(f"Kalan haklar: {st.session_state.attempts}")

        with st.form(key='my_form'):
            ceviri_metin = st.text_area("Buraya Yazın", key='textarea', height=150)
            metin_gonder = st.form_submit_button(label='Gönder')

        if metin_gonder:
            if not ceviri_metin.strip():
                st.error("Metin alanı boş olamaz. Lütfen bir metin girin.")
            else:
                translator = Translator(to_lang="en", from_lang="tr")
                cevrilen_metin = translator.translate(ceviri_metin)

                tahmin = predict_emotions(cevrilen_metin)
                olasilik = get_prediction_proba(cevrilen_metin)

                st.session_state.last_prediction = tahmin
                st.session_state.last_olasilik = olasilik
                st.session_state.last_cevrilen_metin = cevrilen_metin

                if tahmin == st.session_state.current_emotion:
                    st.session_state.scores[st.session_state.current_emotion] = np.max(olasilik) * 10
                    st.success(f"Doğru! {duygular_tr_dict[st.session_state.current_emotion]} için skorunuz: {st.session_state.scores[st.session_state.current_emotion]:.2f}")
                    st.session_state.completed.append(st.session_state.current_emotion)
                    if len(st.session_state.completed) < len(duygular_listesi):
                        st.session_state.current_emotion = random.choice([e for e in duygular_listesi if e not in st.session_state.completed])
                    st.session_state.attempts = 3
                else:
                    st.session_state.attempts -= 1
                    if st.session_state.attempts == 0:
                        st.session_state.scores[st.session_state.current_emotion] = 0
                        st.session_state.completed.append(st.session_state.current_emotion)
                        if len(st.session_state.completed) < len(duygular_listesi):
                            st.session_state.current_emotion = random.choice([e for e in duygular_listesi if e not in st.session_state.completed])
                        st.session_state.attempts = 3

                col1, col2 = st.columns(2)
                with col1:
                    translator = Translator(to_lang="tr", from_lang="en")
                    cevrilen_metin_tahmin = translator.translate(tahmin)
                    st.success("Tahmin")
                    emoji_icon = duygular_emoji_dict[tahmin]
                    st.write(f"{cevrilen_metin_tahmin}: {emoji_icon}")
                    st.write(f"Güven: {np.max(olasilik):.2f}")
                    olasilik_df = pd.DataFrame(olasilik, columns=pipe_lr.classes_)
                    olasilik_df_clean = olasilik_df.T.reset_index()
                    olasilik_df_clean.columns = ["duygular", "olasilik"]

                with col2:
                    st.success("Tahmin Olasılığı")
                  
                if tahmin != st.session_state.current_emotion and st.session_state.attempts > 0:
                    st.experimental_rerun()

    if len(st.session_state.completed) == len(duygular_listesi):

        st.success("Tüm duyguları tamamladınız!")
        st.write("Son skorlarınız:")
        translated_scores = {duygular_tr_dict[emotion]: score for emotion, score in st.session_state.scores.items()}
        scores_df = pd.DataFrame(translated_scores.items(), columns=['Duygular', 'Skor']).set_index('Duygular')
        scores_df = scores_df.reset_index()
        scores_df.columns = ["Duygular", "Skor"]

        st.write(scores_df.to_html(index=False, escape=False), unsafe_allow_html=True)
        st.button("Yeniden Başlat", on_click=sifirla_oyun)
        
    else:
        if st.session_state.last_prediction:
            col1, col2 = st.columns(2)
            with col1:
                translator = Translator(to_lang="tr", from_lang="en")
                cevrilen_metin_1 = translator.translate(st.session_state.last_prediction)
                st.success("Tahmin")
                emoji_icon = duygular_emoji_dict[st.session_state.last_prediction]
                st.write(f"{cevrilen_metin_1}: {emoji_icon}")
                st.write(f"Güven: {np.max(st.session_state.last_olasilik):.2f}")
                olasilik_df = pd.DataFrame(st.session_state.last_olasilik, columns=pipe_lr.classes_)
                olasilik_df_clean = olasilik_df.T.reset_index()
                olasilik_df_clean.columns = ["duygular", "olasilik"]

            with col2:
                st.success("Tahmin Olasılığı")
                
                anger_probability = olasilik_df_clean[olasilik_df_clean['duygular'] == 'anger']['olasilik'].values[0]
                disgust_probability = olasilik_df_clean[olasilik_df_clean['duygular'] == 'disgust']['olasilik'].values[0]
                fear_probability = olasilik_df_clean[olasilik_df_clean['duygular'] == 'fear']['olasilik'].values[0]
                joy_probability = olasilik_df_clean[olasilik_df_clean['duygular'] == 'joy']['olasilik'].values[0]
                neutral_probability = olasilik_df_clean[olasilik_df_clean['duygular'] == 'neutral']['olasilik'].values[0]
                sadness_probability = olasilik_df_clean[olasilik_df_clean['duygular'] == 'sadness']['olasilik'].values[0]
                shame_probability = olasilik_df_clean[olasilik_df_clean['duygular'] == 'shame']['olasilik'].values[0]
                surprise_probability = olasilik_df_clean[olasilik_df_clean['duygular'] == 'surprise']['olasilik'].values[0]

                # Oranları tablo içinde yazdır
                result_df = pd.DataFrame({
                    'duygular': ['Kızgınlık', 'Tiksinme','Korkmak', 'Neşe', 'Etkisiz', 'Üzüntü', 'Utanç' ,'Sürpriz'],
                    'olasilik': [anger_probability, disgust_probability, fear_probability, joy_probability, neutral_probability, sadness_probability, shame_probability, surprise_probability]
                })
                st.table(result_df)

if __name__ == '__main__':
    main()
