import streamlit as st
import yfinance as yf
from google import genai
import pandas as pd

# ------------------------------------------------------------
# Macro News AI Robot Toolkit
# Run with: streamlit run app.py
# Install dependencies:
# pip install streamlit yfinance google-genai pandas
# ------------------------------------------------------------

st.set_page_config(
    page_title="Macro News AI Toolkit",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Macro News AI Robot Toolkit")
st.caption(
    "Analyze NFP, CPI, PPI, and FOMC releases alongside recent market data. "
    "This is an educational analysis aid, not a guaranteed trading signal."
)

# Sidebar configuration
st.sidebar.header("🔧 Toolkit Configuration")
api_key = st.sidebar.text_input("Gemini API Key", type="password")

event_type = st.sidebar.selectbox(
    "Select Economic Event",
    [
        "Non-Farm Payrolls (NFP)",
        "Consumer Price Index (CPI)",
        "Producer Price Index (PPI)",
        "FOMC Rate Decision",
    ],
)

actual_val = st.sidebar.number_input("Actual Release Value", value=0.0, step=0.1)
forecast_val = st.sidebar.number_input("Consensus Forecast Value", value=0.0, step=0.1)
previous_val = st.sidebar.number_input("Previous Value", value=0.0, step=0.1)

st.sidebar.info(
    "Enter values using the same units as the published economic release. "
    "For FOMC, enter the rate or target midpoint consistently."
)

# Market data
st.subheader("📈 Recent Market Context")

@st.cache_data(ttl=60, show_spinner=False)
def get_market_data():
    tickers = {
        "EUR/USD": "EURUSD=X",
        "Gold Futures": "GC=F",
        "S&P 500": "^GSPC",
        "US 10-Year Yield": "^TNX",
    }
    results = {}

    for name, ticker in tickers.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
            closes = hist["Close"].dropna() if not hist.empty and "Close" in hist else pd.Series(dtype=float)

            if len(closes) >= 1:
                current_price = float(closes.iloc[-1])
                previous_price = float(closes.iloc[-2]) if len(closes) >= 2 else None
                change = current_price - previous_price if previous_price is not None else None
                pct_change = (change / previous_price) * 100 if previous_price not in (None, 0) else None

                results[name] = {
                    "Price": current_price,
                    "Change": change,
                    "Pct": pct_change,
                    "Ticker": ticker,
                }
            else:
                results[name] = {"Price": None, "Change": None, "Pct": None, "Ticker": ticker}
        except Exception:
            results[name] = {"Price": None, "Change": None, "Pct": None, "Ticker": ticker}

    return results

with st.spinner("Fetching recent market data..."):
    market_data = get_market_data()

cols = st.columns(4)
for col, (asset, stats) in zip(cols, market_data.items()):
    with col:
        if stats["Price"] is None:
            st.metric(label=asset, value="Unavailable")
        else:
            delta = f'{stats["Pct"]:+.2f}%' if stats["Pct"] is not None else None
            st.metric(label=asset, value=f'{stats["Price"]:,.3f}', delta=delta)

st.caption(
    "Market quotes may be delayed, unavailable outside trading hours, or affected by data-provider limitations."
)

# AI analysis
st.subheader("🧠 Robot AI Macro Analysis")
st.write(
    "Compare the release with the forecast and previous reading, then review possible "
    "market scenarios and risks."
)

if st.button("🚀 Run AI Macro Analysis", type="primary", use_container_width=True):
    if not api_key.strip():
        st.error("Enter your Gemini API key in the sidebar.")
    else:
        with st.spinner("Analyzing the release and recent market context..."):
            try:
                client = genai.Client(api_key=api_key.strip())

                market_lines = []
                for name, stats in market_data.items():
                    if stats["Price"] is None:
                        market_lines.append(f"- {name}: unavailable")
                    else:
                        pct = f'{stats["Pct"]:+.2f}%' if stats["Pct"] is not None else "daily change unavailable"
                        market_lines.append(f'- {name}: {stats["Price"]:.4f} ({pct})')
                market_context = "\n".join(market_lines)

                prompt = f"""
You are a careful macroeconomic analyst. Provide scenario-based market education,
not personalized financial advice and not a guaranteed prediction.

Economic release:
- Event: {event_type}
- Actual: {actual_val}
- Consensus forecast: {forecast_val}
- Previous: {previous_val}

Recent market context from Yahoo Finance:
{market_context}

Write a structured report with:
1. Release surprise: compare actual with forecast and previous. Explain whether the result
   is stronger or weaker than expected, while noting that the meaning depends on the indicator.
2. Macro interpretation: explain possible hawkish/dovish implications. For CPI/PPI, consider
   inflation pressure; for NFP, consider employment strength and wage/inflation context;
   for FOMC, consider the rate decision versus expectations.
3. Scenario map: discuss possible reactions in EUR/USD, gold, and US yields over the next
   1–4 hours. Use conditional language and explain that the initial move can reverse.
4. Confirmation checklist: mention price action, revisions, core readings, press conference/
   forward guidance where relevant, and whether the move holds after the first spike.
5. Risk warning: highlight whipsaws, liquidity, spread widening, and the limits of the
   market data provided. Do not claim certainty or give an instruction to buy or sell.

Clearly distinguish observed data from interpretation. If the input values are ambiguous
or the units are not clear, state that limitation.
"""

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )

                if getattr(response, "text", None):
                    st.markdown(response.text)
                else:
                    st.warning("The AI returned no text. Check your API access and try again.")

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.info(
                    "Check that your Gemini API key is valid, the google-genai package is installed, "
                    "and the selected model is available to your account."
                )

with st.expander("ℹ️ About this toolkit"):
    st.markdown(
        """
        - **Market data:** retrieved through `yfinance`; it may be delayed or incomplete.
        - **AI analysis:** generated by the Gemini API using the values you enter.
        - **No guaranteed signals:** economic releases can cause rapid reversals, and markets may
          already have priced in the result.
        - **API key safety:** enter your own key locally and do not share it publicly or commit it
          to a public code repository.
        """
    )
