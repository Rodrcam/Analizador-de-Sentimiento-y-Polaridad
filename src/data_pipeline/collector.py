import tweepy
import json
import os
import time
from datetime import datetime, timedelta

# --- ⚠️ CONFIGURACIÓN (Asegúrate de que el BEARER_TOKEN es correcto) ⚠️ ---
BEARER_TOKEN = "TU BEARER_TOKEN"
# ---------------------------------------------------------------------

INPUT_FILENAME = 'tweets_raw_ES.json'
OUTPUT_PATH = os.path.join('data', 'raw', INPUT_FILENAME)
RETRY_TIME = 900 # 15 minutos de espera forzada (en segundos)

def load_existing_tweets():
    """Carga los tuits ya guardados para continuar la recolección."""
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                print(f"✅ Se han cargado {len(data)} tuits preexistentes para continuar.")
                return data
            except json.JSONDecodeError:
                print("Advertencia: El archivo JSON está corrupto o vacío. Empezando de nuevo.")
                return []
    return []

def save_tweets(tweets_list):
    """Guarda la lista de tuits en el archivo JSON."""
    os.makedirs(os.path.join('data', 'raw'), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(tweets_list, f, ensure_ascii=False, indent=4)
    print(f"-> Datos guardados en {OUTPUT_PATH}. Total: {len(tweets_list)}")

def collect_tweets_v2(query, max_results_per_call=100, total_limit=1000):
    
    if not BEARER_TOKEN or BEARER_TOKEN == "PEGA_AQUI_TU_BEARER_TOKEN_COMPLETO":
        print("ERROR: El BEARER_TOKEN no está configurado.")
        return

    client = tweepy.Client(BEARER_TOKEN)
    all_tweets = load_existing_tweets()
    tweets_collected = len(all_tweets)
    next_token = None # Usarás esto si tienes un mecanismo para guardar/cargar el token, pero por ahora se omite.
    
    print(f"Buscando tuits V2 con la consulta: '{query}'...")
    print(f"Límite total: {total_limit}. Empezando desde {tweets_collected}.")

    while tweets_collected < total_limit:
        
        current_max_results = min(max_results_per_call, total_limit - tweets_collected)
        if current_max_results <= 0:
            break
        
        try:
            print(f"\nIntentando recolectar {current_max_results} tuits (Total actual: {tweets_collected})...")
            
            response = client.search_recent_tweets(
                query=query,
                max_results=current_max_results,
                tweet_fields=['created_at', 'lang'],
                next_token=next_token 
            )
            
            if response.data:
                # Filtrar y acumular tuits
                new_tweets = []
                for tweet in response.data:
                    if tweet.lang == 'es':
                        tweet_data = {
                            'id_tuit': str(tweet.id),
                            'texto_original': tweet.text, 
                            'fecha': str(tweet.created_at)
                        }
                        new_tweets.append(tweet_data)
                        
                all_tweets.extend(new_tweets)
                tweets_collected = len(all_tweets)
                
                # Guardar los datos recolectados hasta ahora (Guardado Progresivo)
                save_tweets(all_tweets)
                
                # Actualizar el token para la próxima llamada
                next_token = response.meta.get('next_token')
                
                if next_token:
                    print(f"Pausa de 5 segundos antes de la siguiente llamada...")
                    time.sleep(5) 
                else:
                    print("Fin de los resultados disponibles para esta consulta.")
                    break
            else:
                print("No se encontraron tuits nuevos o no hay más resultados.")
                break

        except tweepy.TweepyException as e:
            if e.response and e.response.status_code == 429:
                # 429: Rate Limit excedido
                retry_until = datetime.now() + timedelta(seconds=RETRY_TIME)
                print(f"\n🛑 ERROR 429 (Too Many Requests). Límite de la API excedido.")
                print(f"Se reanudará automáticamente la recolección aproximadamente a las: {retry_until.strftime('%H:%M:%S')}")
                
                # Pausa larga
                time.sleep(RETRY_TIME)
                
            else:
                # Otros errores (ej: 503 Service Unavailable)
                print(f"Error inesperado de la API de X (V2): {e}. Intentando de nuevo en 30 segundos.")
                time.sleep(30)
                
        except Exception as e:
            print(f"Ocurrió un error general: {e}")
            break

    print(f"\n--- RECOLECCIÓN FINALIZADA ---")
    print(f"Total de tuits en el corpus: {len(all_tweets)}")


if __name__ == '__main__':
    
    # Consulta de búsqueda para España, mezclando crítica y elogio
    QUERY_ES = "(vaya tela OR chapuza OR timo OR cutre OR vergüenza OR flipa OR brutal OR máquina OR de locos) lang:es"
    
    collect_tweets_v2(
        query=QUERY_ES,
        total_limit=1000 
    )
