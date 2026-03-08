import streamlit as st
import pandas as pd
import joblib
import re

st.set_page_config(page_title="IS Project: AI Models", page_icon="🍆", layout="wide")

st.sidebar.title("🧭 เมนูนำทาง")
page = st.sidebar.radio("เลือกหน้าการใช้งาน:",
    ["1. 📖 อธิบายโมเดลพยากรณ์ยอดขาย (ML)", 
     "2. 🎮 ทดสอบประเมินยอดขายเกม", 
     "3. 📖 อธิบายโมเดลวิเคราะห์รีวิว (NN)", 
     "4. 👗 ทดสอบวิเคราะห์รีวิวเสื้อผ้า"]
)

st.sidebar.markdown("---")
st.sidebar.info("โปรเจคสำหรับวิชา Intelligent System")

# หน้าที่ 1: อธิบายโมเดล ML (Ensemble)

if page == "1. 📖 อธิบายโมเดลพยากรณ์ยอดขาย (ML)":
    st.title("✨ อธิบายโมเดล Machine Learning (Ensemble Model)")
    st.header("โปรเจค: พยากรณ์ยอดขายวิดีโอเกมทั่วโลก (Video Game Sales)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧹 1. การเตรียมข้อมูล (Data Preprocessing)")
        st.write("- **Dataset:** ข้อมูลยอดขายและคะแนนรีวิววิดีโอเกมจาก Kaggle")
        st.write("- **จัดการค่าผิดปกติ:** แปลงค่า 'tbd' (To Be Determined) ให้เป็นค่าว่าง (NaN) เพื่อให้คำนวณเป็นตัวเลขได้")
        st.write("- **จัดการค่าว่าง (Missing Values):** เติมคะแนนวิจารณ์ (Critic Score) และคะแนนผู้เล่น (User Score) ที่หายไปด้วยค่ามัธยฐาน (Median)")
        st.write("- **การทำความสะอาด:** ลบแถวที่ไม่มีข้อมูลชื่อเกม แพลตฟอร์ม หรือยอดขายทิ้งไป")
        
        st.subheader("⚙️ 3. ขั้นตอนการพัฒนาโมเดล")
        st.write("1. นำข้อมูลที่คลีนแล้วมาแบ่ง Train 80% / Test 20%")
        st.write("2. ใช้ `ColumnTransformer` แปลงชื่อแพลตฟอร์มและแนวเกมเป็นตัวเลข (OneHotEncoding) และปรับสเกลคะแนน (StandardScaler)")
        st.write("3. สร้าง Pipeline นำข้อมูลที่แปลงแล้วส่งเข้าโมเดล Ensemble")

    with col2:
        st.subheader("🧠 2. ทฤษฎีของอัลกอริทึม")
        st.write("ใช้เทคนิค **Ensemble Learning (Voting Regressor)** รวมพลังโมเดล 3 ตัว:")
        st.info("1. **Linear Regression:** หาระนาบความสัมพันธ์เชิงเส้นตรง\n2. **Decision Tree Regressor:** สร้างเงื่อนไขการแตกกิ่งเพื่อประเมินยอดขาย\n3. **Random Forest Regressor:** สร้างต้นไม้ตัดสินใจหลายๆ ต้นแล้วนำค่าเฉลี่ยมาใช้ เพื่อลดความผิดพลาด")
        
        st.subheader("📚 4. แหล่งอ้างอิงข้อมูล")
        st.write("- Dataset: 'Video Game Sales with Ratings' จากเว็บไซต์ Kaggle (อัปโหลดโดย Rush Kirubi)")
        st.write("- อ้างอิงทฤษฎี Ensemble จาก Scikit-Learn Documentation")

# หน้าที่ 2: ทดสอบโมเดล ML (Ensemble)

elif page == "2. 🎮 ทดสอบประเมินยอดขายเกม":
    st.title("🎮 ระบบ AI ประเมินยอดขายวิดีโอเกม")
    st.markdown("จำลองการเป็นนักพัฒนาเกม! กรอกข้อมูลโปรเจคเกมของคุณเพื่อให้ AI ประเมินยอดขายทั่วโลก (หน่วย: ล้านชุด)")
    
    col1, col2 = st.columns(2)
    with col1:
        platform = st.selectbox("🖥️ แพลตฟอร์ม (Platform)", ['PS4', 'XOne', 'PC', 'WiiU', 'Switch', 'PS3', 'X360'])
        genre = st.selectbox("🎯 แนวเกม (Genre)", ['Action', 'Shooter', 'Role-Playing', 'Sports', 'Misc', 'Racing', 'Platform'])
    with col2:
        critic_score = st.slider("⭐ คาดการณ์คะแนนจากนักวิจารณ์ (Critic Score: 1-100)", 1, 100, 75)
        user_score = st.slider("🍆💦 คาดการณ์คะแนนจากผู้เล่น (User Score: 1-10)", 1.0, 10.0, 7.5, step=0.1)
    
    st.markdown("---")
    if st.button("✨ ให้ AI ประเมินยอดขาย", type="primary"):
        try:
            model = joblib.load('games_model.pkl')
            input_data = pd.DataFrame([[platform, genre, critic_score, user_score]], 
                                      columns=['Platform', 'Genre', 'Critic_Score', 'User_Score'])
            prediction = model.predict(input_data)[0]
            
            st.success("การประเมินเสร็จสิ้น!")

            final_pred = max(0, prediction) 
            st.metric(label="💰 คาดการณ์ยอดขายทั่วโลก (ล้านชุด)", value=f"{final_pred:,.2f} Million Copies")
        except FileNotFoundError:
            st.error("⚠️ ไม่พบไฟล์โมเดล 'games_model.pkl' กรุณารัน train_models.py ก่อน")

# หน้าที่ 3: อธิบายโมเดล Neural Network

elif page == "3. 📖 อธิบายโมเดลวิเคราะห์รีวิว (NN)":
    st.title("✨ อธิบายโมเดล Neural Network")
    st.header("โปรเจค: วิเคราะห์รีวิวเสื้อผ้าสตรีใน E-Commerce")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧹 1. การเตรียมข้อมูล (Data Preprocessing)")
        st.write("- **Dataset:** ข้อมูลรีวิวเสื้อผ้าสตรีของจริงจาก Kaggle")
        st.write("- **จัดการค่าว่าง:** ลบแถวที่ลูกค้าให้ดาวแต่ไม่พิมพ์ข้อความรีวิวทิ้งไป")
        st.write("- **Text Cleaning (RegEx):** ลบ HTML Tags, ลบ Emoji, ลบเครื่องหมายวรรคตอน และเก็บไว้เพียงตัวอักษรภาษาอังกฤษกับตัวเลข จากนั้นแปลงเป็นตัวพิมพ์เล็ก (Lowercase)")
        
        st.subheader("⚙️ 3. ขั้นตอนการพัฒนาโมเดล")
        st.write("1. นำข้อความมารันผ่านฟังก์ชันทำความสะอาด (`clean_text`)")
        st.write("2. ใช้ `TfidfVectorizer` (จำกัด 5,000 คำศัพท์ยอดฮิต) แปลงข้อความเป็นเวกเตอร์ตัวเลข")
        st.write("3. สร้างโมเดล `MLPClassifier` โดยออกแบบให้มี 2 Hidden Layers (100 โหนด และ 50 โหนด)")

    with col2:
        st.subheader("🧠 2. ทฤษฎีของอัลกอริทึม")
        st.write("**การแปลงข้อความ (TF-IDF):**")
        st.info("เทคนิคให้คะแนนความสำคัญของคำ (Term Frequency-Inverse Document Frequency) ทำให้ AI รู้ว่าคำไหนเป็นคีย์เวิร์ดสำคัญของประโยค")
        st.write("**โครงข่ายประสาทเทียม (MLP Classifier):**")
        st.info("ใช้สถาปัตยกรรมแบบ Feedforward นำตัวเลข TF-IDF เข้าสู่ Input Layer ส่งผ่าน Hidden Layer เพื่อสกัดคุณลักษณะที่ซับซ้อน และทายผลความน่าจะเป็นออกทาง Output Layer")
        
        st.subheader("📚 4. แหล่งอ้างอิงข้อมูล")
        st.write("- Dataset: 'Women's E-Commerce Clothing Reviews' จาก Kaggle (อัปโหลดโดย nicapotato)")


# หน้าที่ 4: ทดสอบโมเดล Neural Network

elif page == "4. 👗 ทดสอบวิเคราะห์รีวิวเสื้อผ้า":
    st.title("👗 ระบบ AI วิเคราะห์ความพึงพอใจลูกค้า")
    st.markdown("พิมพ์รีวิวสินค้า (ภาษาอังกฤษ) เพื่อให้ AI ประเมินว่าลูกค้ารายนี้ **แนะนำ (Recommended)** หรือ **ไม่แนะนำ (Not Recommended)** สินค้าชิ้นนี้")

    def clean_text(text):
        if pd.isna(text): return text
        text = str(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = ' '.join(text.split())
        return text.lower()

    user_input = st.text_area("พิมพ์ข้อความรีวิวของลูกค้าที่นี่ :", height=150, placeholder="e.g. The fabric is very soft and it fits perfectly! I love it.")
    
    if st.button("วิเคราะห์การแนะนำสินค้า", type="primary"):
        if user_input.strip() == "":
            st.warning("กรุณาพิมพ์ข้อความรีวิวก่อนครับ")
        else:
            try:
                nn_model = joblib.load('clothing_nn_model.pkl')
                vectorizer = joblib.load('clothing_vectorizer.pkl')

                cleaned_input = clean_text(user_input)
                input_vec = vectorizer.transform([cleaned_input])

                prediction = nn_model.predict(input_vec)[0]
                
                st.markdown("---")
                if prediction == 1:
                    st.success("🌟 AI วิเคราะห์ว่า: **ลูกค้าแนะนำสินค้านี้ (Recommended)**")
                else:
                    st.error("💔 AI วิเคราะห์ว่า: **ลูกค้าไม่แนะนำสินค้านี้ (Not Recommended)**")
                    
            except FileNotFoundError:
                st.error("!!! ไม่พบไฟล์โมเดล กรุณารัน train_models.py ก่อน")
                
# streamlit run app.py