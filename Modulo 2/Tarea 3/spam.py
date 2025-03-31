# Integrantes: Grande Espinoza Victor Ramon, Montero Lopez Yahel Alejandro.
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np

nltk.download('punkt')

# Metodo para el preprocesamiento de datos
def limpiar_texto(texto):
    texto = texto.lower()  
    texto = re.sub(r'\W', ' ', texto)  
    tokens = word_tokenize(texto)  
    return tokens

# Datos de ejemplo
data = {'email': ['Gana dinero fácil $$$ desde casa!!!',
                  'Reunión de trabajo a las 3 PM.',
                  'Gana dinero fácil $$$ desde casa!!!',
                  'Haz clic aquí para reclamar tu premio!'],
        'spam': [1, 0, 1, 1]}
df = pd.DataFrame(data)

# Eliminar correos electrónicos duplicados
df = df.drop_duplicates()

# Aplicar limpieza y preprocesamiento
df['tokens'] = df['email'].apply(limpiar_texto)

# Cálculo de la probabilidad previa P(Spam)
num_spam = df['spam'].sum()
total_correos = len(df)
P_spam = num_spam / total_correos
P_no_spam = 1 - P_spam 

# Separar en spam y no spam
spam_tokens = [word for tokens in df[df['spam'] == 1]['tokens'] for word in tokens]
no_spam_tokens = [word for tokens in df[df['spam'] == 0]['tokens'] for word in tokens]

# Contar frecuencia de palabras en spam y no spam
spam_counts = Counter(spam_tokens)
no_spam_counts = Counter(no_spam_tokens)

# Calcular P(Palabra | Spam) con suavizado de Laplace
V = len(set(spam_tokens + no_spam_tokens))
total_spam_words = sum(spam_counts.values())
total_no_spam_words = sum(no_spam_counts.values())

prob_spam_words = {word: (spam_counts[word] + 1) / (total_spam_words + V) for word in spam_counts}
prob_no_spam_words = {word: (no_spam_counts[word] + 1) / (total_no_spam_words + V) for word in no_spam_counts}

# Función para calcular P(Spam | Características) con Bayes
def calcular_probabilidad_spam(texto):
    palabras = limpiar_texto(texto)
    
    # P(Características | Spam) = Producto de P(palabra | Spam)
    P_caracteristicas_spam = np.prod([prob_spam_words.get(word, 1 / (total_spam_words + V)) for word in palabras])
    
    # P(Características | NoSpam) = Producto de P(palabra | NoSpam)
    P_caracteristicas_no_spam = np.prod([prob_no_spam_words.get(word, 1 / (total_no_spam_words + V)) for word in palabras])
    
    # Aplicar fórmula de Bayes
    P_spam_dado_caracteristicas = (P_spam * P_caracteristicas_spam) / (
        (P_spam * P_caracteristicas_spam) + (P_no_spam * P_caracteristicas_no_spam)
    )
    
    return P_spam_dado_caracteristicas

# 6. Función para clasificar correos electrónicos
def clasificar_correo(texto):
    palabras = limpiar_texto(texto)
    
    # P(Características | Spam) = Producto de P(palabra | Spam)
    P_caracteristicas_spam = np.prod([prob_spam_words.get(word, 1 / (total_spam_words + V)) for word in palabras])
    
    # P(Características | NoSpam) = Producto de P(palabra | NoSpam)
    P_caracteristicas_no_spam = np.prod([prob_no_spam_words.get(word, 1 / (total_no_spam_words + V)) for word in palabras])
    
    # Aplicar fórmula de Bayes
    P_spam_dado_caracteristicas = (P_spam * P_caracteristicas_spam) / (
        (P_spam * P_caracteristicas_spam) + (P_no_spam * P_caracteristicas_no_spam)
    )
    
    # Clasificación
    if P_spam_dado_caracteristicas > 0.5:
        return "Spam"
    else:
        return "No Spam"

# Calcular precisión y recuperación
clasificaciones = np.array([clasificar_correo(correo) for correo in df['email']])
precision = np.sum(clasificaciones == df["spam"]) / len(clasificaciones)
recuperacion = np.sum(clasificaciones == df["spam"]) / df["spam"].sum()

# Resultado:
print(df)
print(f"\nProbabilidad previa de Spam (P(Spam)): {P_spam:.2f}")
print("\nProbabilidades P(Palabra | Spam):")
for word, prob in list(prob_spam_words.items())[:10]:
    print(f"{word}: {prob:.4f}")
print("\nProbabilidades P(Palabra | NoSpam):")
for word, prob in list(prob_no_spam_words.items())[:10]:
    print(f"{word}: {prob:.4f}")

# Probabilidad posterior de que el correo sea Spam con ejemplo nuevo:
nuevo_email = "Gana dinero ahora con esta increíble oferta."
probabilidad_spam = calcular_probabilidad_spam(nuevo_email)
resultado = clasificar_correo(nuevo_email)
print(f"\nNuevo Correo: {nuevo_email} ➡ Clasificación: {resultado}")
print(f"Probabilidad de que el correo sea spam: {probabilidad_spam:.4f}")

print(f"Precisión: {precision:.4f}")
print(f"Recuperación: {recuperacion:.4f}")
