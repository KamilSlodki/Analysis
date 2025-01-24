from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from kucoin.client import Market
from ta.momentum import RSIIndicator
from ta.trend import MACD
from pmdarima import auto_arima
import os
from dotenv import load_dotenv  # Necesario para pruebas locales

# === Configuración inicial ===

# Cargar .env solo para pruebas locales
if not os.getenv("KUCOIN_API_KEY"):
    load_dotenv()

# Obtener claves de las variables de entorno
KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_SECRET_KEY = os.getenv("KUCOIN_SECRET_KEY")
KUCOIN_PASSPHRASE = os.getenv("KUCOIN_PASSPHRASE")

# Verificar que las claves están configuradas
if not all([KUCOIN_API_KEY, KUCOIN_SECRET_KEY, KUCOIN_PASSPHRASE]):
    raise ValueError("Faltan claves de API. Verifica las variables de entorno en Render o el archivo .env.")

# Configurar el cliente de KuCoin
market_client = Market(KUCOIN_API_KEY, KUCOIN_SECRET_KEY, KUCOIN_PASSPHRASE)

# === Configurar FastAPI ===
app = FastAPI()

# === Modelos de datos ===
class SymbolRequest(BaseModel):
    symbol: str
    interval: str

class PriceData(BaseModel):
    prices: list[float]

# === Funciones principales ===
def obtener_velas(symbol: str, interval: str):
    try:
        velas = market_client.get_kline(symbol, interval)
        if not velas:
            raise ValueError("Datos de velas vacíos.")
        df = pd.DataFrame(velas, columns=['time', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
        df = df.drop(columns=['turnover'])  # Eliminar columnas innecesarias
        return df
    except Exception as e:
        raise ValueError(f"Error al obtener datos de velas: {e}")

def calcular_indicadores(df: pd.DataFrame):
    df['close'] = pd.to_numeric(df['close'])
    df['RSI'] = RSIIndicator(df['close']).rsi()
    macd = MACD(df['close'])
    df['MACD'], df['MACD_Signal'] = macd.macd(), macd.macd_signal()
    return df

def modelo_arima_auto(precios):
    try:
        modelo = auto_arima(precios, seasonal=False, trace=False, error_action="ignore", suppress_warnings=True)
        predicciones = modelo.predict(n_periods=5)
        return predicciones
    except Exception as e:
        raise ValueError(f"Error al ajustar ARIMA: {e}")

# === Endpoints ===
@app.post("/get_data/")
async def get_data(request: SymbolRequest):
    try:
        df = obtener_velas(request.symbol, request.interval)
        df = calcular_indicadores(df)
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

@app.post("/predictions/arima/")
async def arima_predictions(data: PriceData):
    try:
        predicciones = modelo_arima_auto(data.prices)
        return {"predictions": predicciones.tolist()}
    except Exception as e:
        return {"error": str(e)}
