import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from utils import set_img_as_background, load_trained_model, predict_model

def home():
    """
    Display the home page content with a background image and a title.

    Returns:
        None
    """
    set_img_as_background("asset/dashboard.png")
    st.markdown("""
        ## <span style='color:#F79BD3'>Ketahui jenis kulit anda hanya dengan selfie</span>
    """, unsafe_allow_html = True)

def classification(model):
    """
    Perform skin type classification based on user-uploaded or captured images.

    Parameters:
        model (tensorflow.keras.models.Model): Trained Keras model for classification.

    Returns:
        None
    """
    list_class = ["Kulit Berminyak", "Kulit Kering", "Kulit Kombinasi", "Kulit Normal", "Kulit Sensitif"]
    def _inference(upload_file):
        image = Image.open(upload_file)
        image = image.resize((150, 150))
        image = np.array(image)
        image = np.reshape(image, (-1, 150, 150, 3))
        result = predict_model(image, model)
        section_img, section_table = st.columns(2)
        section_img.image(upload_file)
        section_table.dataframe({
            "Jenis Kulit" : list_class, 
            "Probabilitas (%)": [round(i * 100, 2) for i in result[0]]
        })
    
    options = st.selectbox(
        "Ketahui jenis kulit kamu 📸", 
        ["", "Pilih Gambar", "Ambil Gambar"]
    )
    if options == "Pilih Gambar":
        upload_file = st.file_uploader("")
        if upload_file is not None: 
            _inference(upload_file)

    if options == "Ambil Gambar":
        upload_file = st.camera_input("")
        if upload_file is not None: 
            _inference(upload_file)

def about():
    """
    Display information about different skin types based on user selection.

    Returns:
        None
    """
    options = st.selectbox(
        "Ketahui deskripsi jenis kulitmu lebih lanjut disini👇", 
        ["", "Kulit Normal", "Kulit Berminyak", "Kulit Kering", "Kulit Kombinasi", "Kulit Sensitif"]
    )
    if options == "Kulit Normal":
        desc = \
        """
        # Kulit Normal
        
        Jenis kulit ini cenderung memiliki keseimbangan antara jumlah kandungan air dan minyak, sehingga tidak terlalu kering tapi juga tidak terlalu berminyak.Jenis kulit wajah seperti ini biasanya jarang memiliki masalah kulit, tidak terlalu sensitif, terlihat bercahaya, dan pori-pori pun hampir tak terlihat. Jenis kulit normal juga lebih mudah dirawat.

        **Solusi**:
         
        Membersihkan wajah cukup dengan air, ketika kulit wajah dalam keadaan tanpa make up. Jika kulit wajah dalam keadaan bermakeup, bisa dibersihkan menggunakan milk cleanser, face tonic dan facial foam. Bisa menggunakan face tonic dan krim pelembab, ketika musim panas. Karena di musim panas kulit normal akan terasa agak kering. Perawatan facial di klinik kecantikan diperlukan sewaktu-waktu saja, cukup 1 kali dalam 3 bulan. Menggunakan krim tabir surya untuk melindungi dari panas sinar matahari
        """
        st.markdown(desc)
        set_img_as_background("asset/dashboard.png") 
    if options == "Kulit Berminyak":
        desc = \
        """
        # Kulit Berminyak
        
        Jenis kulit wajah berminyak cenderung licin dan mengkilap karena produksi minyak atau sebum yang berlebih. Sebum dihasilkan secara alami oleh kelenjar minyak atau kelenjar sebaceous di bawah permukaan kulit.

        **Solusi**:
        
        Membersihkan wajah menggunakan facial foam, kemudian dibilas sampai bersih. Setelah mencuci wajah, gunakan face tonic.

        """
        st.markdown(desc)
        set_img_as_background("asset/berminyak.jpg")
    if options == "Kulit Kering":
        desc = \
        """
        # Kulit Kering
        
        Kulit wajah kering umumnya terjadi akibat rendahnya tingkat kelembapan pada lapisan kulit terluar. Hal ini mengakibatkan kulit kering mudah pecah-pecah dan mengalami keretakan pada permukaan kulit. Pemilik kulit wajah kering biasanya memiliki pori-pori kulit yang hampir tak terlihat, permukaan luar kulit terlihat kasar dan kusam, serta kulit kurang elastis. Jenis kulit ini juga lebih mudah memerah, gatal, bersisik, dan meradang.

        **Solusi**:
         
        Gunakan krim pelembap sesering mungkin, baik pada siang maupun malam hari. Gunakan tabir surya pada siang hari, karena kulit kering ini sangat mudah terkena flek kecokelatan. Jangan terlalu sering menggunakan sabun wajah.

        """
        st.markdown(desc)
        set_img_as_background("asset/kering.jpg")
    if options == "Kulit Kombinasi":
        desc = \
        """
        # Kulit Kombinasi
        
        Jenis kulit wajah kombinasi adalah perpaduan antara kulit berminyak dan kulit kering. Seseorang dengan jenis kulit wajah ini memiliki kulit berminyak di zona T, yaitu area dagu, hidung, dan dahi, serta kulit kering di area pipi.

        **Solusi**:
        
        Gunakan selalu facial foam, milk cleanser dan face tonic. Lakukan perawatan facial di salon kecantikan sebulan sekali. Oleskan tipis-tipis krim atau lotion pencegah komedo pada malam hari.

        """
        st.markdown(desc)
        set_img_as_background("asset/kombinasi.jpg")
    if options == "Kulit Sensitif":
        desc = \
        """
        # Kulit Sensitif
        
        Jenis kulit sensitif umumnya sangat peka dan mudah sekali mengalami alergi atau iritasi dan ruam sebagai reaksi terhadap faktor tertentu, seperti lingkungan, makanan, atau penggunaan produk kosmetik.

        **Solusi**:
        
        Berdasarkan gejalanya, perawatan kulit sensitif ditujukan untuk melindungi kulit serta mengurangi dan menanggulangi iritasi. Kulit sensitif tidak dapat diamati secara langsung, diperlukan bantuan dokter kulit atau dermatolog untuk memeriksanya dalam tes alergi- imunologi. Apabila dideteksi alergi, maka biasanya pasien akan diberi beberapa allergen untuk mengetahui kadar sensitivitas kulit

        """
        st.markdown(desc)
        set_img_as_background("asset/sensitif.jpg")
    

def interface():
    """
    Create the Streamlit app interface with tabs for Home, Klasifikasi, and Tentang.

    Returns:
        None
    """
    model = load_trained_model()

    tab_home, tab_clf, tab_about = st.tabs([
        "Home", "Klasifikasi", "Tentang"
    ])
    with tab_home:
        home()

    with tab_clf:
        classification(model)

    with tab_about:
        about()

if __name__ == "__main__":
    interface()