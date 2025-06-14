import asyncio
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from autogen import ConversableAgent, register_function
import re
import shutil
import yfinance as yf
import numpy as np
import ta
from test import advanced_technical_analysis, generate_comprehensive_signal, identify_candlestick_patterns, calculate_fibonacci_retracement
import threading

def fetch_gainers_list() -> str:
    """
    Fetches and parses the top stock gainers list from Webull's website.
    Also updates stocks_to_analyze.csv with the tickers only (no company names).
    """
    url = "https://www.webull.com/quote/us/gainers"
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    # Use asyncio.sleep instead of time.sleep for async compatibility
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.sleep(5))
    gainers = []
    tickers = set()
    rows = driver.find_elements(By.CSS_SELECTOR, 'div.table-row')
    for row in rows:
        cells = row.find_elements(By.CSS_SELECTOR, 'div.table-cell')
        if len(cells) >= 6:
            ticker_candidate = cells[1].text.strip()
            # Split by newlines and take the last non-empty line as the ticker
            ticker_lines = [line.strip() for line in ticker_candidate.split('\n') if line.strip()]
            if ticker_lines:
                ticker = ticker_lines[-1]
                print(f"DEBUG: extracted ticker='{ticker}' from ticker_candidate='{ticker_candidate}'")
                if re.match(r'^[A-Z]{1,5}(-[A-Z]{1,5})?$', ticker):
                    tickers.add(ticker)
                # For gainers_str, just use the full ticker_candidate as the name for now
                price = cells[3].text
                change = cells[4].text
                percent = cells[5].text
                gainers_str = f"{ticker}: {ticker_candidate} | Price: {price} | Change: {change} | %: {percent}"
                gainers.append(gainers_str)
    driver.quit()
    # Update stocks_to_analyze.csv with only tickers
    if tickers:
        with open("stocks_to_analyze.csv", "w", newline="") as f:
            f.write("ticker\n")
            for t in sorted(tickers):
                f.write(f"{t}\n")
    return "\n".join(gainers[:10]) if gainers else "No gainers found."

def analyze_bullish_score_changes(csv_path="stock_analysis.csv") -> str:
    """
    Analyzes the CSV for stocks with rising, falling, or unchanged bullish scores.
    Returns a summary string.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error reading CSV: {e}"
    rising = []
    falling = []
    unchanged = []
    for _, row in df.iterrows():
        try:
            prev = float(row.get("Previous_Bullish_Score", 0))
            curr = float(row.get("Bullish_Score", 0))
        except Exception:
            continue
        if curr > prev:
            rising.append(f"{row['ticker']}: {prev} → {curr} (▲{curr-prev})")
        elif curr < prev:
            falling.append(f"{row['ticker']}: {prev} → {curr} (▼{prev-curr})")
        else:
            unchanged.append(f"{row['ticker']}: {curr} (no change)")
    summary = []
    if rising:
        summary.append("Stocks with rising bullish score:\n" + "\n".join(rising))
    if falling:
        summary.append("Stocks with falling bullish score:\n" + "\n".join(falling))
    if unchanged:
        summary.append("Stocks with unchanged bullish score:\n" + "\n".join(unchanged))
    return "\n\n".join(summary) if summary else "No data or no changes detected."

def update_indicators_and_scores(csv_tickers_path, csv_out_path):
    """
    For each ticker in csv_tickers_path, fetch price data, calculate indicators, and write to csv_out_path.
    Adds debug output for DataFrame shape/columns to help debug alignment errors.
    """
    tickers = pd.read_csv(csv_tickers_path)['ticker'].tolist()
    results = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="3mo", interval="1d", progress=False)
            print(f"{ticker}: initial df shape: {df.shape}, columns: {df.columns.tolist()}")
            if len(df) < 50:
                continue  # Not enough data for indicators
            df = df.reset_index()
            print(f"{ticker}: after reset_index: {df.shape}, columns: {df.columns.tolist()}")
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            print(f"{ticker}: after flattening columns: {df.shape}, columns: {df.columns.tolist()}")
            df = advanced_technical_analysis(df)
            print(f"{ticker}: after advanced_technical_analysis: {df.shape}, columns: {df.columns.tolist()}")
            patterns = identify_candlestick_patterns(df)
            fib = calculate_fibonacci_retracement(df)
            latest = df.iloc[-1]
            signal = generate_comprehensive_signal(df, latest, patterns, fib)
            results.append({
                'ticker': ticker,
                'RSI': latest['RSI'],
                'EMA_9': latest['EMA_9'],
                'EMA_20': latest['EMA_20'],
                'EMA_33': latest['EMA_33'],
                'EMA_100': latest['EMA_100'],
                'EMA_200': latest['EMA_200'],
                'Bullish_Score': signal['bullish_score'] if isinstance(signal, dict) and 'bullish_score' in signal else None
            })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    pd.DataFrame(results).to_csv(csv_out_path, index=False)

def refresh_analysis_loop():
    """
    Every minute, copy new->old, update new, and let the agent analyze changes.
    This function is now thread-safe and non-blocking for asyncio.
    """
    def loop_body():
        while True:
            try:
                shutil.copy('stock_analysis_new.csv', 'stock_analysis_old.csv')
            except Exception:
                pass  # First run, file may not exist
            update_indicators_and_scores('stocks_to_analyze.csv', 'stock_analysis_new.csv')
            print("Refreshed analysis. Waiting 60 seconds...")
            time.sleep(60)
    t = threading.Thread(target=loop_body, daemon=True)
    t.start()

async def main() -> None:
    assistant = ConversableAgent(
        name="Assistant",
        system_message="You are a helpful AI assistant. You can fetch the daily top gainers from Webull. Return 'TERMINATE' when the task is done.",
        llm_config={
            "config_list": [
                {
                    "model": "llama3-groq-tool-use:latest",
                    "base_url": "http://localhost:11434",
                    "api_type": "ollama",
                    "api_key": "ollama"
                }
            ]
        },
    )
    user_proxy = ConversableAgent(
        name="User",
        llm_config=False,
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
        human_input_mode="NEVER",
    )
    register_function(
        fetch_gainers_list,
        caller=assistant,
        executor=user_proxy,
        name="fetch_gainers_list",
        description="Fetches the top stock gainers list from Webull's website."
    )
    register_function(
        analyze_bullish_score_changes,
        caller=assistant,
        executor=user_proxy,
        name="analyze_bullish_score_changes",
        description="Analyzes the CSV for stocks with rising, falling, or unchanged bullish scores."
    )
    register_function(
        analyze_indicator_changes,
        caller=assistant,
        executor=user_proxy,
        name="analyze_indicator_changes",
        description="Compares the old and new CSVs for each ticker and summarizes which indicators (RSI, EMA, Bullish_Score, etc.) are rising, falling, or unchanged."
    )
    chat_result = user_proxy.initiate_chat(
        assistant,
        message="Can you get me the daily top gainers from Webull?"
    )
    print(chat_result)

def analyze_indicator_changes(old_csv='stock_analysis_old.csv', new_csv='stock_analysis_new.csv') -> str:
    """
    Compares the old and new CSVs for each ticker and summarizes which indicators (RSI, EMA, Bullish_Score, etc.) are rising, falling, or unchanged.
    """
    try:
        old_df = pd.read_csv(old_csv)
        new_df = pd.read_csv(new_csv)
    except Exception as e:
        return f"Error reading CSVs: {e}"
    summary = []
    indicators = ['RSI', 'EMA_9', 'EMA_20', 'EMA_33', 'EMA_100', 'EMA_200', 'Bullish_Score']
    for _, new_row in new_df.iterrows():
        ticker = new_row['ticker']
        old_row = old_df[old_df['ticker'] == ticker]
        if old_row.empty:
            continue
        old_row = old_row.iloc[0]
        changes = []
        for ind in indicators:
            try:
                old_val = float(old_row[ind])
                new_val = float(new_row[ind])
                if new_val > old_val:
                    changes.append(f"{ind} ↑ ({old_val:.2f} → {new_val:.2f})")
                elif new_val < old_val:
                    changes.append(f"{ind} ↓ ({old_val:.2f} → {new_val:.2f})")
                else:
                    changes.append(f"{ind} = ({new_val:.2f})")
            except Exception:
                continue
        if changes:
            summary.append(f"{ticker}: " + ", ".join(changes))
    if not summary:
        return "No changes detected or insufficient data."
    return "\n".join(summary)
