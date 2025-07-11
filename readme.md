import logging
from ib_insync import IB, Option, Stock, Contract, Position, Ticker, IBConnectionError, util
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set

# Configuración básica de logging (INFO para producción, DEBUG para desarrollo)
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
# Silenciar logs detallados de ib_insync si es necesario
# logging.getLogger('ib_insync').setLevel(logging.WARNING)

# --- Clases y Funciones Auxiliares ---

class MarketCache:
    """
    Gestiona la obtención y el cacheo de datos de mercado (tickers y griegas)
    desde Interactive Brokers con un tiempo de vida (TTL) configurable.
    Utiliza suscripciones batch para mayor eficiencia.
    """
    def __init__(self, ib: IB, ttl_seconds: int = 5):
        if not ib or not ib.isConnected():
            raise ValueError("Se requiere una instancia de IB conectada.")
        self.ib: IB = ib
        self.ttl: timedelta = timedelta(seconds=ttl_seconds)
        self.ticker_cache: Dict[int, Ticker] = {}       # Cache de tickers por conId
        self.greeks_cache: Dict[int, Tuple[Optional[float], ...]] = {} # Cache de griegas por conId
        self.ticker_times: Dict[int, datetime] = {}       # Timestamps para ticker_cache
        self.greeks_times: Dict[int, datetime] = {}       # Timestamps para greeks_cache
        self._subscribed_conIds: Set[int] = set()         # Para evitar suscripciones duplicadas

    def _is_fresh(self, key: int, time_store: Dict[int, datetime]) -> bool:
        """Comprueba si la entrada en cache 'key' está dentro del TTL."""
        last_update = time_store.get(key)
        if last_update is None:
            return False
        return (datetime.now() - last_update) < self.ttl

    def subscribe_batch(self, contracts: List[Contract]):
        """
        Suscribe de forma agrupada a tickers (que incluyen precios y griegas para opciones)
        para una lista de contratos, evitando duplicados ya suscritos.
        """
        new_contracts_to_subscribe = [
            c for c in contracts if c.conId not in self._subscribed_conIds and c.conId # Asegura conId válido
        ]

        if not new_contracts_to_subscribe:
            logging.debug("No new contracts to subscribe in this batch.")
            return

        logging.info(f"Batch subscribing to {len(new_contracts_to_subscribe)} new contracts for market data and Greeks...")
        try:
            # reqTickers devuelve una lista de tickers actualizada en tiempo real
            tickers = self.ib.reqTickers(*new_contracts_to_subscribe, regulatorySnapshot=False)
            # Es buena idea esperar un poco para que lleguen los datos iniciales
            self.ib.sleep(0.5 + len(new_contracts_to_subscribe) * 0.05) # Ajustar si es necesario

            now = datetime.now()
            for t in tickers:
                if not isinstance(t, Ticker) or not t.contract: # Comprobar ticker válido
                    logging.warning(f"Received invalid ticker data: {t}")
                    continue

                cid = t.contract.conId
                if cid is None:
                    logging.warning(f"Ticker received without conId for contract: {t.contract}")
                    continue

                # Actualizar cache de tickers
                self.ticker_cache[cid] = t
                self.ticker_times[cid] = now
                self._subscribed_conIds.add(cid) # Marcar como suscrito

                # Actualizar cache de griegas si están disponibles (normalmente para opciones)
                mg = t.modelGreeks
                if mg:
                    greeks_tuple = (mg.delta, mg.gamma, mg.theta, mg.vega, mg.iv) # Añadimos IV
                    self.greeks_cache[cid] = greeks_tuple
                    self.greeks_times[cid] = now
                    logging.debug(f"Cached Greeks for conId={cid}: {greeks_tuple}")
                elif cid in self.greeks_cache:
                    # Si previamente tenía griegas y ahora no, invalidar
                    del self.greeks_cache[cid]
                    if cid in self.greeks_times: del self.greeks_times[cid]

            logging.info(f"Batch subscription processing complete for {len(tickers)} tickers.")

        except Exception as e:
            logging.error(f"Error during batch subscription (reqTickers): {e}", exc_info=True)
            # Podríamos intentar cancelar suscripciones fallidas si tuviéramos sus tickers
            st.error(f"Error subscribing to market data batch: {e}")

    def get_ticker(self, contract: Contract) -> Optional[Ticker]:
        """
        Obtiene el ticker de un contrato desde la cache si está fresco.
        No realiza una nueva suscripción aquí; se asume que se hizo con subscribe_batch.
        """
        cid = contract.conId
        if cid is None:
            logging.warning(f"Contract {contract} has no conId.")
            return None

        if cid not in self._subscribed_conIds:
             logging.warning(f"Attempted to get ticker for conId={cid} which was not subscribed via batch.")
             # Opcional: Podríamos intentar una suscripción individual aquí si es necesario
             # self.subscribe_batch([contract])
             # self.ib.sleep(1) # Esperar datos
             # return self.ticker_cache.get(cid)
             return None # Por ahora, devolvemos None si no se suscribió en batch

        if cid in self.ticker_cache and self._is_fresh(cid, self.ticker_times):
            return self.ticker_cache[cid]
        elif cid in self.ticker_cache:
            # Está suscrito pero el dato en caché no está fresco (raro con reqTickers activo)
             logging.debug(f"Ticker data for conId={cid} is stale, but should be updating live.")
             return self.ticker_cache[cid] # Devolver el último conocido, se actualizará solo
        else:
            # Está suscrito pero aún no hay datos en la caché (puede tardar un instante)
            logging.warning(f"Ticker for conId={cid} subscribed but not yet in cache.")
            return None

    def get_mid_price(self, contract: Contract) -> Optional[float]:
        """
        Calcula el precio medio (bid+ask)/2 desde la cache.
        Usa 'last' o 'close' como fallback si bid/ask no están disponibles.
        Devuelve None si no hay precio disponible.
        """
        t = self.get_ticker(contract)
        if not t:
            return None

        bid = t.bid
        ask = t.ask
        last = t.last
        close = t.close # Precio de cierre del día anterior

        # Priorizar mid-price si bid y ask son válidos y el spread no es irreal
        if bid and ask and bid > 0 and ask > 0 and (ask - bid) < (ask * 0.5): # Spread < 50% del ask
            price = (bid + ask) / 2.0
            logging.debug(f"Mid price for {contract.localSymbol}: {price:.2f} (Bid: {bid}, Ask: {ask})")
            return price
        # Usar 'last' si es válido
        elif last and not pd.isna(last) and last > 0:
            price = last
            logging.debug(f"Mid price for {contract.localSymbol}: {price:.2f} (Using Last)")
            return price
        # Usar 'close' como último recurso
        elif close and not pd.isna(close) and close > 0:
            price = close
            logging.debug(f"Mid price for {contract.localSymbol}: {price:.2f} (Using Close)")
            return price
        else:
            logging.warning(f"Could not determine a valid price for {contract.localSymbol} (conId={contract.conId}). Ticker: {t}")
            return None

    def get_greeks(self, contract: Contract) -> Optional[Tuple[Optional[float], ...]]:
        """
        Obtiene las griegas (delta, gamma, theta, vega, iv) desde la cache si están frescas.
        Devuelve None si no están disponibles.
        """
        cid = contract.conId
        if cid is None: return None

        # Asegura que los datos del ticker (que contienen las griegas) estén relativamente frescos
        if not self._is_fresh(cid, self.ticker_times):
             logging.warning(f"Greeks data for conId={cid} might be stale (ticker time).")
             # No refrescamos activamente aquí, confiamos en la suscripción batch

        if cid in self.greeks_cache and self._is_fresh(cid, self.greeks_times):
            greeks = self.greeks_cache[cid]
            logging.debug(f"Greeks for {contract.localSymbol}: Delta={greeks[0]}, Gamma={greeks[1]}, Theta={greeks[2]}, Vega={greeks[3]}, IV={greeks[4]}")
            return greeks
        else:
            # Puede que aún no hayan llegado o que el contrato no sea una opción
             logging.debug(f"No fresh Greeks found in cache for {contract.localSymbol} (conId={cid}).")
             # Intentar obtener el ticker por si acaso actualiza las griegas
             ticker = self.get_ticker(contract)
             if ticker and ticker.modelGreeks:
                  mg = ticker.modelGreeks
                  greeks_tuple = (mg.delta, mg.gamma, mg.theta, mg.vega, mg.iv)
                  self.greeks_cache[cid] = greeks_tuple
                  self.greeks_times[cid] = datetime.now() # Actualizar timestamp
                  return greeks_tuple
             return None

    def close(self):
        """Cancela todas las suscripciones de datos de mercado activas."""
        if not self.ib or not self.ib.isConnected():
            return
        logging.info(f"Cancelling {len(self._subscribed_conIds)} market data subscriptions...")
        contracts_to_cancel = []
        for cid in self.subscribed_conIds:
             if cid in self.ticker_cache:
                 contracts_to_cancel.append(self.ticker_cache[cid].contract)
        # Intenta cancelar en batch si es posible (ib_insync no lo soporta directamente)
        # Cancelar una por una
        for contract in contracts_to_cancel:
             try:
                 self.ib.cancelMktData(contract)
                 logging.debug(f"Cancelled market data for {contract.localSymbol}")
             except Exception as e:
                 logging.warning(f"Error cancelling market data for {contract.localSymbol}: {e}")
        self._subscribed_conIds.clear()
        self.ticker_cache.clear()
        self.greeks_cache.clear()
        self.ticker_times.clear()
        self.greeks_times.clear()


@st.cache_data # Decorador moderno de Streamlit para cachear datos
def compute_expiries(positions_tuple: Tuple[Position, ...]) -> Dict[int, int]:
    """Calcula los días hasta la expiración para cada posición."""
    # Usar tupla como input para que sea hasheable por st.cache_data
    logging.info(f"Computing expiries for {len(positions_tuple)} positions...")
    now_date = datetime.now().date()
    expiries: Dict[int, int] = {}
    for p in positions_tuple:
        contract = p.contract
        cid = contract.conId
        ds = contract.lastTradeDateOrContractMonth
        if not ds or not cid: continue # Saltar si falta información

        try:
            # Intentar parsear la fecha, manejar diferentes formatos si es necesario
            if len(ds) == 8: # YYYYMMDD
                exp_date = datetime.strptime(ds, '%Y%m%d').date()
            else: # Añadir más formatos si IB los usa
                logging.warning(f"Formato de fecha desconocido para conId={cid}: {ds}")
                continue
            # Calcular días restantes, asegurando que no sea negativo
            days_left = max(0, (exp_date - now_date).days)
            expiries[cid] = days_left
        except ValueError:
            logging.error(f"Formato de fecha inválido para conId={cid}: {ds}")
        except Exception as e:
             logging.error(f"Error calculando expiración para conId={cid}: {e}")

    logging.info(f"Expiries computed: {len(expiries)} entries.")
    return expiries

def connect_ib(host: str, port: int, client_id: int) -> Optional[IB]:
    """Establece conexión con IB TWS/Gateway."""
    logging.info(f"Attempting to connect to IB: {host}:{port} (ClientID: {client_id})")
    ib = IB()
    try:
        # Aumentar timeout si la conexión es lenta
        ib.connect(host, port, clientId=client_id, timeout=10)
        if ib.isConnected():
            logging.info(f"Successfully connected to IB. Server version: {ib.serverVersion()}")
            return ib
        else:
            logging.error("IB connection attempt returned, but not connected.")
            return None
    except IBConnectionError as e:
        logging.error(f"IB Connection Error: {e}")
        st.error(f"Connection Error: {e}")
        return None
    except Exception as e:
        logging.error(f"Failed to connect to IB: {e}", exc_info=True)
        st.error(f"Failed to connect: {e}")
        return None

def get_option_positions(ib: IB) -> Optional[List[Position]]:
    """Obtiene las posiciones de opciones de la cuenta."""
    logging.info("Fetching account positions...")
    try:
        all_positions = ib.positions()
        option_positions = [p for p in all_positions if p.contract.secType == 'OPT']
        logging.info(f"Found {len(option_positions)} option positions out of {len(all_positions)} total.")
        return option_positions
    except Exception as e:
        logging.error(f"Failed to retrieve positions: {e}", exc_info=True)
        st.error(f"Error fetching positions: {e}")
        return None

def get_underlying_contract(option_contract: Contract) -> Optional[Contract]:
    """
    Determina el contrato subyacente para una opción dada.
    Asume que es una Acción (Stock) por simplicidad. Requiere ajuste para Futuros etc.
    """
    # TODO: Implementar una lógica más robusta si se manejan Futuros u otros.
    # Se podría usar ib.reqContractDetails(option_contract) para obtener underlyingConId
    # y luego buscar detalles de ese conId, pero aumenta la latencia.
    if option_contract.secType != 'OPT':
        return None
    # Usar primaryExchange si está disponible, si no, exchange.
    exchange = option_contract.primaryExchange or option_contract.exchange
    if not exchange:
         logging.warning(f"No exchange found for option {option_contract.localSymbol}, cannot determine underlying exchange.")
         # Podríamos intentar con SMART si es común para el usuario
         exchange = "SMART"

    if option_contract.symbol and exchange and option_contract.currency:
         # Asume Stock. Cambiar a Future(...) si es necesario.
        underlying = Stock(option_contract.symbol, exchange, option_contract.currency)
        logging.debug(f"Determined underlying for {option_contract.localSymbol} as {underlying}")
        return underlying
    else:
        logging.warning(f"Missing symbol/exchange/currency for option {option_contract.localSymbol}, cannot create underlying contract.")
        return None


# --- Lógica Principal Refactorizada ---

def prepare_contracts_for_analysis(
    positions: List[Position],
    underlying_map: Dict[int, Contract], # Mapa: option_conId -> underlying_contract
    underlying_prices: Dict[int, float], # Mapa: option_conId -> underlying_price
    width: float
) -> Tuple[Dict[int, Option], Dict[int, Option], List[Contract]]:
    """
    Construye los contratos Put y Call para el collar y reúne todos los contratos
    necesarios (originales, subyacentes, puts, calls) para la suscripción batch.
    """
    put_map: Dict[int, Option] = {}
    call_map: Dict[int, Option] = {}
    all_contracts_to_subscribe: Set[Contract] = set() # Usar set para evitar duplicados

    logging.info(f"Preparing collar contracts based on {len(underlying_prices)} underlying prices.")

    for p in positions:
        option_contract = p.contract
        option_conId = option_contract.conId

        if option_conId not in underlying_prices or option_conId not in underlying_map:
            logging.warning(f"Skipping position {option_contract.localSymbol} (conId={option_conId}): Missing underlying price or contract.")
            continue

        underlying_price = underlying_prices[option_conId]
        underlying_contract = underlying_map[option_conId]

        # Añadir contrato original y subyacente a la lista de suscripción
        all_contracts_to_subscribe.add(option_contract)
        all_contracts_to_subscribe.add(underlying_contract)

        # Calcular strikes para Put (OTM) y Call (OTM)
        # Redondear a 2 decimales o según las reglas del mercado/activo
        put_strike = round(underlying_price - width, 2)
        call_strike = round(underlying_price + width, 2)

        # Validar strikes (deben ser positivos)
        if put_strike <= 0 or call_strike <= 0:
             logging.warning(f"Calculated invalid strike(s) for {option_contract.localSymbol} (Underlying: {underlying_price:.2f}, Width: {width}). Skipping collar.")
             continue

        expiry_date = option_contract.lastTradeDateOrContractMonth
        symbol = option_contract.symbol
        exchange = option_contract.exchange # Usar el mismo exchange que la opción original? O el del subyacente? O SMART?
        currency = option_contract.currency

        # Crear contratos Put y Call del Collar
        # Asegurarse que el exchange es correcto para las nuevas opciones (puede ser diferente al original)
        # Usar 'SMART' puede ser una opción general, pero verificar requerimientos de IB
        collar_exchange = underlying_contract.exchange # Usar exchange del subyacente suele ser más seguro

        try:
            put_option = Option(symbol, expiry_date, put_strike, 'P', collar_exchange, currency=currency)
            call_option = Option(symbol, expiry_date, call_strike, 'C', collar_exchange, currency=currency)

             # Añadir contratos del collar a la lista y los mapas
            put_map[option_conId] = put_option
            call_map[option_conId] = call_option
            all_contracts_to_subscribe.add(put_option)
            all_contracts_to_subscribe.add(call_option)
            logging.debug(f"Collar for {symbol} {expiry_date}: PUT @{put_strike}, CALL @{call_strike}")

        except Exception as e:
             logging.error(f"Error creating Put/Call contract for {symbol} {expiry_date}: {e}")


    # Convertir set a lista para la suscripción
    contract_list = list(all_contracts_to_subscribe)
    logging.info(f"Total unique contracts prepared for batch subscription: {len(contract_list)}")
    return put_map, call_map, contract_list

def calculate_strategy_metrics(
    positions: List[Position],
    cache: MarketCache,
    put_map: Dict[int, Option], # Mapa: option_conId -> Put Option contract
    call_map: Dict[int, Option], # Mapa: option_conId -> Call Option contract
    expiries: Dict[int, int],    # Mapa: option_conId -> days to expiry
    width: float,
) -> pd.DataFrame:
    """
    Calcula el coste neto, protección máxima y eficiencia para cada estrategia de collar.
    Utiliza los precios cacheados por MarketCache.
    """
    rows = []
    logging.info("Calculating strategy metrics...")

    for p in positions:
        option_contract = p.contract
        option_conId = option_contract.conId

        # Verificar si tenemos los contratos del collar y la expiración para esta posición
        if option_conId not in put_map or option_conId not in call_map or option_conId not in expiries:
            logging.debug(f"Skipping metric calculation for {option_contract.localSymbol}: Missing collar contracts or expiry.")
            continue

        put_contract = put_map[option_conId]
        call_contract = call_map[option_conId]

        # Obtener precios mid (o fallback) desde la cache
        put_price = cache.get_mid_price(put_contract)
        call_price = cache.get_mid_price(call_contract)

        # Obtener griegas (opcionalmente, para posible uso futuro)
        # put_greeks = cache.get_greeks(put_contract)
        # call_greeks = cache.get_greeks(call_contract)
        # original_greeks = cache.get_greeks(option_contract)

        # Validar que obtuvimos precios válidos
        if put_price is None or call_price is None:
            logging.warning(f"Skipping {option_contract.localSymbol}: Could not get valid prices for Put ({put_price}) or Call ({call_price}).")
            continue

        # Calcular coste neto del collar (Coste de la Put - Prima de la Call)
        net_cost = put_price - call_price

        # Protección máxima (diferencia de strikes * multiplicador * num_contratos)
        # Asumiendo multiplicador estándar de 100 para opciones sobre acciones
        multiplier = 100 # TODO: Hacer configurable o obtener de contractDetails si es necesario
        # La protección es relevante para posiciones largas en el subyacente.
        # Si la posición original es una Call larga o Put corta (alcista), el collar protege.
        # Si es una Put larga o Call corta (bajista), el collar protege si se invierte (compra Call, vende Put).
        # Este script asume que se quiere proteger la posición original, sea cual sea.
        # La protección viene dada por la Put comprada.
        # El valor protegido es (Strike Put - Precio Actual + Coste Neto)?? No, es más simple.
        # La protección máxima es la diferencia entre el strike de la Put y donde estaría el precio si baja mucho.
        # Más simple: el valor máximo que la Put puede alcanzar es su strike (menos la prima pagada).
        # O, si el subyacente va a cero, el valor es strike*multiplier.
        # Pero la "protección" del collar se refiere a limitar la pérdida MÁXIMA.
        # Para una acción larga + collar: Pérdida Max = Precio Accion - Strike Put + Coste Neto Collar
        # Aquí no tenemos la acción, solo la opción. El concepto de "protección" es diferente.
        # Quizás la intención es "valor salvado"? O el beneficio si el precio cae bajo el strike Put?
        # Reinterpretando "MaxProtection" como el ancho del collar en euros:
        max_protection_per_share = width # Diferencia entre strike Put y precio actual aprox.
        # Si width es 5€, y el multiplicador 100, protege 500€ por contrato (si el precio baja 5€ o más)
        # Multiplicar por el número de contratos de la posición original y el multiplicador.
        position_size = abs(p.position) # Usar valor absoluto
        max_protection_euros = max_protection_per_share * multiplier * position_size

        # Ignorar estrategias con coste cero o negativo (o muy pequeño) ya que la eficiencia sería infinita o inválida
        # Un coste negativo significa que recibes crédito por abrir el collar.
        if net_cost <= 0.01: # Umbral pequeño para evitar división por cero o eficiencias enormes
             logging.debug(f"Skipping {option_contract.localSymbol}: Net cost is zero or negative ({net_cost:.2f}).")
             continue

        # Calcular Eficiencia (Protección (€) / Coste Neto (€))
        efficiency = max_protection_euros / net_cost

        rows.append({
            'Symbol': option_contract.symbol,
            'Expiry': option_contract.lastTradeDateOrContractMonth,
            'DaysLeft': expiries[option_conId],
            'Position': p.position, # Mostrar si es Long/Short
            'Orig. Strike': option_contract.strike,
            'Orig. Type': option_contract.right,
            'Collar Width': width,
            'Put Strike': put_contract.strike,
            'Call Strike': call_contract.strike,
            'Net Cost': net_cost,
            'Max Protection (€)': max_protection_euros,
            'Efficiency': efficiency
            # Añadir griegas aquí si se quieren mostrar
            # 'Put Delta': put_greeks[0] if put_greeks else None,
            # ... etc ...
        })

    if not rows:
        logging.info("No valid strategy metrics could be calculated.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logging.info(f"Calculated metrics for {len(df)} strategies.")
    return df


# --- Función Principal de Streamlit ---

def main():
    st.set_page_config(layout="wide")
    st.title("IB Option Protective Collar Evaluator")

    # --- Sidebar para Parámetros ---
    with st.sidebar.form(key='ib_params_form'):
        st.header("IB Connection & Parameters")
        host = st.text_input('Host', '127.0.0.1')
        port = st.number_input('Port', min_value=1, max_value=65535, value=7497) # Puerto TWS por defecto
        client_id = st.number_input('Client ID', min_value=0, value=1)
        width = st.number_input('Collar Width (€/$)', min_value=0.1, value=5.0, step=0.1, format="%.2f")
        # ttl_cache = st.number_input('Cache TTL (seconds)', min_value=1, value=5) # Hacer TTL configurable?
        run_button = st.form_submit_button('Evaluate Strategies')

    if not run_button:
        st.info("Configure connection details and click 'Evaluate Strategies'.")
        return

    # --- Lógica de Ejecución ---
    ib: Optional[IB] = None
    cache: Optional[MarketCache] = None
    try:
        # 1. Conectar a IB
        with st.spinner("Connecting to Interactive Brokers..."):
            ib = connect_ib(host, port, client_id)

        if not ib:
            # connect_ib ya muestra el error en st.error
            return # Detener si la conexión falla

        # Inicializar MarketCache
        cache = MarketCache(ib, ttl_seconds=5) # Usar TTL fijo por ahora

        # 2. Obtener Posiciones de Opciones
        with st.spinner("Fetching option positions..."):
            positions = get_option_positions(ib)

        if positions is None:
            # get_option_positions ya muestra el error en st.error
            return # Detener si falla la obtención
        if not positions:
            st.info("No option positions found in the account.")
            return # Terminar si no hay posiciones

        st.success(f"Found {len(positions)} option positions.")

        # 3. Calcular Expiraciones (usando cache de Streamlit)
        with st.spinner("Calculating expirations..."):
            # Pasar una tupla a la función cacheada
            expiries = compute_expiries(tuple(positions))

        # 4. Determinar Contratos Subyacentes y Obtener sus Precios
        underlying_map: Dict[int, Contract] = {}
        underlying_contracts_to_fetch: List[Contract] = []
        option_conId_to_underlying: Dict[int, Contract] = {} # Mapa: option_conId -> underlying_contract

        with st.spinner("Determining underlying contracts..."):
            for p in positions:
                option_conId = p.contract.conId
                if option_conId not in underlying_map:
                     underlying_contract = get_underlying_contract(p.contract)
                     if underlying_contract:
                         # Calificar el contrato para asegurar que IB lo encuentra
                         try:
                             qual_contract = ib.qualifyContracts(underlying_contract)[0]
                             underlying_map[option_conId] = qual_contract
                             option_conId_to_underlying[option_conId] = qual_contract
                             if qual_contract not in underlying_contracts_to_fetch:
                                  underlying_contracts_to_fetch.append(qual_contract)
                         except Exception as e:
                              logging.warning(f"Could not qualify underlying contract for {p.contract.localSymbol}: {e}")

        if not underlying_contracts_to_fetch:
             st.warning("Could not determine any valid underlying contracts.")
             return

        # 5. Suscribir a Subyacentes para obtener precios iniciales
        with st.spinner(f"Fetching initial prices for {len(underlying_contracts_to_fetch)} underlyings..."):
            cache.subscribe_batch(underlying_contracts_to_fetch)
            # Dar tiempo a que lleguen los precios
            ib.sleep(1) # Ajustar si es necesario

        underlying_prices: Dict[int, float] = {}
        for option_conId, underlying_contract in option_conId_to_underlying.items():
            price = cache.get_mid_price(underlying_contract)
            if price is not None:
                underlying_prices[option_conId] = price
                logging.info(f"Got underlying price for {underlying_contract.symbol}: {price:.2f} (linked to option {option_conId})")
            else:
                logging.warning(f"Could not get initial price for underlying {underlying_contract.symbol} (conId={underlying_contract.conId}).")

        if not underlying_prices:
            st.error("Failed to retrieve market prices for any underlying asset. Cannot calculate collars.")
            return

        # 6. Preparar Contratos del Collar y Lista Batch Completa
        with st.spinner("Preparing collar contracts..."):
            put_map, call_map, all_contracts = prepare_contracts_for_analysis(
                positions, option_conId_to_underlying, underlying_prices, width
            )

        if not all_contracts:
            st.warning("No valid collar contracts could be generated.")
            return

        # 7. Suscribir a Todos los Contratos (Originales, Subyacentes, Puts, Calls)
        with st.spinner(f"Subscribing to market data for {len(all_contracts)} total contracts..."):
            cache.subscribe_batch(all_contracts) # El cache manejará duplicados
            # Espera adicional para asegurar que los datos fluyan
            ib.sleep(1.5) # Ajustar según sea necesario

        # 8. Calcular Métricas de la Estrategia
        with st.spinner("Calculating strategy metrics..."):
            df_results = calculate_strategy_metrics(
                positions, cache, put_map, call_map, expiries, width
            )

        # --- Mostrar Resultados ---
        if not df_results.empty:
            st.subheader("Protective Collar Opportunities")
            # Ordenar y formatear para visualización
            df_display = df_results.sort_values('Efficiency', ascending=False).copy()
            df_display['Net Cost'] = df_display['Net Cost'].map('{:,.2f}'.format)
            df_display['Max Protection (€)'] = df_display['Max Protection (€)'].map('{:,.0f}'.format)
            df_display['Efficiency'] = df_display['Efficiency'].map('{:,.1f}x'.format) # Mostrar como 'ratio x'

            # Seleccionar y renombrar columnas para claridad
            df_display = df_display[[
                'Symbol', 'Expiry', 'DaysLeft', 'Position', 'Orig. Strike', 'Orig. Type',
                'Put Strike', 'Call Strike', 'Net Cost', 'Max Protection (€)', 'Efficiency'
            ]]
            st.dataframe(df_display, use_container_width=True) # Usar todo el ancho
        else:
            st.info("No suitable protective collar strategies found or calculated based on current data and parameters.")

    except ConnectionRefusedError:
         st.error(f"Connection Refused: Ensure TWS/Gateway is running on {host}:{port}, API connections are enabled, and the Client ID {client_id} is not in use.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in main execution: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {e}")

    finally:
        # --- Desconectar ---
        if cache:
             with st.spinner("Cancelling market data subscriptions..."):
                 cache.close() # Intenta cancelar suscripciones
        if ib and ib.isConnected():
            logging.info("Disconnecting from IB.")
            ib.disconnect()
            st.info("Disconnected from IB.")


if __name__ == '__main__':
    # util.patchAsyncio() # Necesario en algunos entornos, especialmente si usas notebooks
    util.startLoop() # Requerido por ib_insync en scripts
    main()

Cambios Clave y Mejoras Implementadas:
Corrección Precio Base Collar:
Se añade get_underlying_contract para determinar el subyacente (asumiendo Stock, ajustable).
Se califican (ib.qualifyContracts) los subyacentes para asegurar que IB los reconoce.
Se obtienen los precios de estos subyacentes (cache.get_mid_price).
Estos precios se usan en prepare_contracts_for_analysis para calcular los strikes put_strike y call_strike.
Uso de Griegas:
MarketCache sigue obteniendo y cacheando las griegas (delta, gamma, theta, vega) y también la Volatilidad Implícita (iv) cuando están disponibles en el Ticker (principalmente para opciones).
La función calculate_strategy_metrics tiene comentarios indicando dónde se podrían añadir las griegas al DataFrame de resultados si se necesitan mostrar o usar.
Cache Streamlit:
compute_expiries ahora usa @st.cache_data y acepta una tupla de posiciones para ser hasheable.
Manejo de Errores:
Se han añadido bloques try...except más específicos alrededor de las llamadas críticas a IB (connect, positions, qualifyContracts, reqTickers).
MarketCache maneja errores internos en suscripciones y cancelaciones.
Las funciones devuelven Optional[...] (ej. Optional[IB], Optional[float]) para indicar posibles fallos.
Se usan st.spinner, st.error, st.warning, st.info para dar feedback al usuario en la interfaz.
Se incluye un bloque finally para asegurar la desconexión de IB y la cancelación de suscripciones.
Refactorización main:
La lógica se ha dividido en funciones más claras: connect_ib, get_option_positions, compute_expiries, get_underlying_contract, prepare_contracts_for_analysis, calculate_strategy_metrics.
MarketCache encapsula toda la lógica de obtención y cacheo de datos de mercado.
Type Hints:
Se han añadido anotaciones de tipo en la mayoría de las funciones y métodos para mejorar la legibilidad y permitir análisis estático.
Limpieza y Optimizaciones:
MarketCache usa timedelta para el TTL.
Se usa set para evitar duplicados al preparar la lista de contratos para subscribe_batch.
subscribe_batch ahora evita suscribir contratos ya suscritos.
Se ha mejorado la lógica de get_mid_price con fallbacks a last y close.
Se usa ib.sleep() estratégicamente para dar tiempo a que lleguen los datos después de las suscripciones.
El DataFrame final se formatea para una mejor presentación en Streamlit.
Se incluye util.startLoop() necesario para ib_insync en modo script.
Este código está mucho más cercano a una versión de producción. Recuerda probarlo exhaustivamente con tu cuenta de Paper Trading de IB antes de usarlo con dinero real. Es posible que necesites ajustar la lógica de get_underlying_contract si operas con Futuros u otros instrumentos complejos, y afinar los tiempos de ib.sleep() según la latencia de tu conexión.