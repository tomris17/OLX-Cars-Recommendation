import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="OLX Cars Recommendation App", layout="centered")

st.title("OLX Cars Content-Based Recommendation App")
st.write(
    "Bu uygulama, araç özellikleri ve açıklamalarını TF-IDF ve kosinüs benzerliği ile birleştirerek benzer araçlar önerir."
)

@st.cache_data
def load_data():
    df = pd.read_csv("OLX_cars_dataset00.csv", low_memory=False)
    df["Car Features"] = df["Car Features"].fillna("")
    df["Description"] = df["Description"].fillna("")
    df["Make"] = df["Make"].fillna("")
    df["Model"] = df["Model"].fillna("")
    return df

try:
    df = load_data()
    
    def create_car_soup(x):
        return (
            str(x["Make"])
            + " "
            + str(x["Model"])
            + " "
            + str(x["Car Name"])
            + " "
            + str(x["Fuel"])
            + " "
            + str(x["Transmission"])
            + " "
            + str(x["Car Features"])
        )

    df["soup"] = df.apply(create_car_soup, axis=1)

    @st.cache_data
    def compute_similarity(data):
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(data["soup"])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        return cosine_sim

    cosine_sim = compute_similarity(df)
    indices = pd.Series(df.index, index=df["Car Name"]).drop_duplicates()

    def get_car_recommendations(car_name):
        if car_name not in indices:
            return None
        idx = indices[car_name]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        car_indices = [i[0] for i in sim_scores]
        return df[["Car Name", "Make", "Model", "Price", "Year"]].iloc[car_indices]

    st.subheader("Araç Öneri Paneli")
    unique_cars = df["Car Name"].dropna().unique()
    selected_car = st.selectbox("Bir Araç Seçin", unique_cars)

    if st.button("Öneri Al"):
        recommendations = get_car_recommendations(selected_car)
        if recommendations is not None:
            st.write(f"'{selected_car}' için önerilen benzer araçlar:")
            st.dataframe(recommendations)
        else:
            st.warning("Seçilen araç için öneri bulunamadı.")

except Exception as e:
    st.error(f"Veri yüklenirken veya işlenirken bir hata oluştu: {e}")