import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
import sqlite3
import base64
import hashlib
import os

st.set_page_config(page_title="Ratio", page_icon="icon.png")

# === Connexion à la base de données ===
conn = sqlite3.connect("data_app.db", check_same_thread=False)
cursor = conn.cursor()

# === Création des tables si elles n'existent pas ===
cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS glucose_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    day INTEGER,
    gly_before REAL,
    gly_after REAL,
    timestamp TEXT
)''')
conn.commit()

# === Fonctions ===
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password):
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
    conn.commit()

def authenticate_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hash_password(password)))
    return cursor.fetchone()

def save_glucose_data(username, day, before, after):
    cursor.execute("INSERT INTO glucose_data (username, day, gly_before, gly_after, timestamp) VALUES (?, ?, ?, ?, ?)",
                   (username, day, before, after, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()

def load_glucose_data(username):
    cursor.execute("""
        SELECT day, gly_before, gly_after, timestamp 
        FROM glucose_data 
        WHERE username=? 
        ORDER BY datetime(timestamp) DESC
    """, (username,))
    return cursor.fetchall()

def get_last_saved_for_day(username, day):
    cursor.execute("""
        SELECT gly_before, gly_after 
        FROM glucose_data 
        WHERE username=? AND day=? 
        ORDER BY datetime(timestamp) DESC LIMIT 1
    """, (username, day))
    row = cursor.fetchone()
    if row:
        return row[0], row[1]
    else:
        return 0.0, 0.0

def save_analysis_to_txt(username, variations_detail, conclusion):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder = "sauvegardes"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{username}_glycemia_analysis.txt")
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"=== Results - {now} ===\n")
        for i, (before, after, var) in enumerate(variations_detail, 1):
            f.write(f"Day {i}: Before = {before} mmol/L, After = {after} mmol/L, Variation = {var:.2f} mmol/L\n")
        f.write(f"{conclusion}\n")
        f.write("="*50 + "\n\n")
    return filename

def analyze_variations(variations):
    stables = sum(abs(v) < 1 for v in variations)
    if stables >= 2:
        return True, "✅ Glycémie stable au moins 2 jours sur 3. La dose d’insuline à injecter est appropriée.."
    else:
        return False, "⚠️ Glycémie variable sur au moins 2 jours. Ajustement nécessaire."


# === Initialisation session_state ===
for key in ["stable", "variations_detail", "logged_in", "user", "last_saved_day", "last_saved_data", "conclusion"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "logged_in" else False

# === Sidebar Connexion ===
with st.sidebar:
    st.title("🔐 Espace Professionnel")

    if not st.session_state.logged_in:
        auth_mode = st.radio("Choisissez une option", ["Login", "Sign Up"])
        username = st.text_input("Nom d'utilisateur", key="username_input")
        password = st.text_input("Mot de passe", type="password", key="password_input")

        if st.button("Valider"):
            if username and password:
                if auth_mode == "Sign Up":
                    try:
                        add_user(username, password)
                        st.success("Compte créé ! Connectez-vous.")
                    except sqlite3.IntegrityError:
                        st.error("Ce nom d'utilisateur existe déjà.")
                else:
                    if authenticate_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.user = username
                        st.success(f"Connecté en tant que {username}")
                    else:
                        st.error("Identifiants incorrects.")
            else:
                st.error("Veuillez remplir tous les champs.")
    else:
        st.markdown(f"👤 Connecté en tant que **{st.session_state.user}**")
        if st.button("Déconnexion"):
            for key in ["logged_in", "user", "last_saved_day", "last_saved_data", "stable", "variations_detail", "conclusion"]:
                st.session_state[key] = None if key != "logged_in" else False

# === Stop si pas connecté ===
if not st.session_state.logged_in:
    st.warning("Veuillez vous connecter via le menu latéral.")
    st.stop()

# === Interface principale ===
st.markdown(f"### 👋 Bonjour, **{st.session_state.user}** !")
st.title("Insulin-to-Carbohydrate Ratio Calculator")
st.markdown("## 📘 Ratio User Guide")

# PDF affichage
if st.button("📄 Show Ratio PDF file"):
    try:
        with open("Ratio.pdf", "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="900px"></iframe>',
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        st.error("❌ Le fichier 'Ratio.pdf' est introuvable.")

# Convertisseur
st.header("Grams → Millimoles (mmol) Converter")
MOLAR_MASS = 180.16
grams = st.number_input("Amount in grams (g)", min_value=0.0, step=0.1)
if st.button("Convert to mmol"):
    mmol = (grams * 1000) / MOLAR_MASS
    st.success(f"{grams} g = **{mmol:.2f} mmol**")

# Entrée glycémie
st.markdown("## Entrée des glycémies pour 3 jours")
variations = []
variations_detail = []

for i in range(1, 4):
    gly_before_default, gly_after_default = get_last_saved_for_day(st.session_state.user, i)
    st.subheader(f"Day {i}")
    col1, col2, col3 = st.columns([3, 3, 2])
    with col1:
        gly_before = st.number_input(f"Glycémie avant repas - Jour {i}", key=f"before_{i}", step=0.1, value=gly_before_default)
    with col2:
        gly_after = st.number_input(f"Glycémie avant prochain repas - Jour {i}", key=f"after_{i}", step=0.1, value=gly_after_default)
    with col3:
        if st.button(f"Sauvegarder jour {i}", key=f"save_{i}"):
            save_glucose_data(st.session_state.user, i, gly_before, gly_after)
            st.success(f"Jour {i} sauvegardé ✅")
            st.session_state.last_saved_day = i
            st.session_state.last_saved_data = (gly_before, gly_after)

    variation = gly_after - gly_before
    variations.append(variation)
    variations_detail.append((gly_before, gly_after, variation))

# Affichage dernière sauvegarde
if st.session_state.last_saved_day and st.session_state.last_saved_data:
    day = st.session_state.last_saved_day
    before, after = st.session_state.last_saved_data
    st.info(f"**Données sauvegardées du Jour {day} :** Avant = {before} mmol/L, Après = {after} mmol/L")

# Analyse
if st.button("Analyser les glycémies"):
    stable, conclusion = analyze_variations(variations)
    st.session_state.stable = stable
    st.session_state.variations_detail = variations_detail
    st.session_state.conclusion = conclusion

    saved_file = save_analysis_to_txt(st.session_state.user, variations_detail, conclusion)
    st.success(conclusion)

    days = [f"Day {i+1}" for i in range(3)]
    fig, ax = plt.subplots()
    ax.bar(days, variations, color="skyblue")
    ax.axhline(1, color="red", linestyle="--", label="1 mmol/L")
    ax.axhline(-1, color="red", linestyle="--")
    ax.set_ylabel("Variation (mmol/L)")
    ax.set_title("Variation de glycémie")
    ax.legend()
    st.pyplot(fig)

    with open(saved_file, "r", encoding="utf-8") as file:
        txt_content = file.read()
        st.download_button(
            label="📥 Télécharger les résultats",
            data=txt_content,
            file_name=os.path.basename(saved_file),
            mime="text/plain"
        )

# Historique
st.markdown("### 🗂️ Historique des données sauvegardées")
hist = load_glucose_data(st.session_state.user)
if hist:
    for day, before, after, timestamp in hist:
        st.write(f"📅 **{timestamp}** | 🗓️ Jour {day} : Avant = {before} mmol/L | Après = {after} mmol/L")
else:
    st.info("Aucune donnée sauvegardée.")

# Résumé
st.markdown("---")
st.markdown("### 📊 Résumé des valeurs saisies aujourd'hui")
if st.session_state.variations_detail:
    for i, (before, after, var) in enumerate(st.session_state.variations_detail, 1):
        st.write(f"Jour {i} : Avant = {before} mmol/L, Après = {after} mmol/L, Variation = {var:.2f} mmol/L")
    if st.session_state.conclusion:
        st.markdown(f"**Conclusion :** {st.session_state.conclusion}")
else:
    st.info("Aucune donnée saisie ou analysée pour le moment.")
