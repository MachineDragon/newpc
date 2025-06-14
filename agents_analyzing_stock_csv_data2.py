# Import necessary libraries and modules
import pandas as pd
import asyncio
import os

from autogen.agentchat.groupchat import GroupChat, GroupChatManager
from autogen.agentchat.assistant_agent import AssistantAgent

# --- LLM CONFIGS ---
OLLAMA_URL = "http://localhost:11434"
OLLAMA_KEY = "ollama"

config_phi3 = [{
    "api_type": "ollama",
    "model": "phi3:3.8b",
    "base_url": OLLAMA_URL,
    "api_key": OLLAMA_KEY,
    "temperature": 0.1
}]
config_qwen3 = [{
    "api_type": "ollama",
    "model": "qwen3:4b",
    "base_url": OLLAMA_URL,
    "api_key": OLLAMA_KEY,
    "temperature": 0.1
}]
config_qwen25 = [{
    "api_type": "ollama",
    "model": "qwen2.5:7b-instruct-q4_K_M",
    "base_url": OLLAMA_URL,
    "api_key": OLLAMA_KEY,
    "temperature": 0.1
}]
config_phi4 = [{
    "api_type": "ollama",
    "model": "phi4-mini-reasoning:3.8b-q4_K_M",
    "base_url": OLLAMA_URL,
    "api_key": OLLAMA_KEY,
    "temperature": 0.1
}]
config_llama3 = [{
    "api_type": "ollama",
    "model": "llama3-groq-tool-use:latest",
    "base_url": OLLAMA_URL,
    "api_key": OLLAMA_KEY,
    "temperature": 0.1
}]
config_gemma3_4b = [{
    "api_type": "ollama",
    "model": "gemma3:4b-it-q4_K_M",
    "base_url": OLLAMA_URL,
    "api_key": OLLAMA_KEY,
    "temperature": 0.1
}]

# --- AGENTIC WORKFLOW ---
def run_round_robin_groupchat(scalper_prompt):
    agents = [
        AssistantAgent(
            "TrendAgent",
            llm_config={"config_list": config_phi3},
            system_message=(
                "You are a trend analysis expert. Focus ONLY on these columns: overall_trend, overall_trend_confidence, trend_based_on_levels, short_term_bullish_trend_confidence, long_term_bullish_trend_confidence. "
                "Summarize which tickers have the strongest short-term bullish trends and why, using only these columns. "
                "Share your findings with the group and suggest which tickers are strongest by trend."
            )
        ),
        AssistantAgent(
            "VolumeAgent",
            llm_config={"config_list": config_gemma3_4b},
            system_message=(
                "You are a volume and momentum expert. Focus ONLY on these columns: volume_trend, volume_trend_signal, VWAP, ATR, ADX. "
                "Identify which tickers have the best volume/momentum setup for a scalp trade and explain your reasoning. "
                "Share your findings with the group and suggest which tickers are strongest by volume/momentum."
            )
        ),
        AssistantAgent(
            "IndicatorAgent",
            llm_config={"config_list": config_qwen25},
            system_message=(
                "You are a technical indicator specialist. Focus ONLY on these columns: MACD, MACD_Trend, RSI, RSI_Trend, Price_Above_9_20_33_200_EMAs, Price_Above_50_150_200_SMA_and_SMA_Alignment, 20_SMA_Above_200_SMA (Long-term), Bollinger_Low, Bollinger_High, Stoch_K, Stoch_D, Candlestick_Pattern, Fibonacci_Levels. "
                "Identify which tickers have the most bullish technical indicator setup for a scalp trade and explain your reasoning. "
                "Share your findings with the group and suggest which tickers are strongest by technical indicators."
            )
        ),
        AssistantAgent(
            "ScoreAgent",
            llm_config={"config_list": config_phi4},
            system_message=(
                "You are a scoring and summary expert. Focus ONLY on these columns: Bullish_Score, Bullish_Score_Trend, latest_price. "
                "Identify which tickers have the highest bullish score and best score trend for a scalp trade. "
                "Share your findings with the group and suggest which tickers are strongest by score."
            )
        ),
        AssistantAgent(
            "FinalDecisionAgent",
            llm_config={"config_list": config_llama3},
            system_message=(
                "You are a senior scalper. Listen to the other agents' findings. Weigh their arguments and pick the single best ticker for a 1-minute scalp trade. "
                "Justify your choice based on the group discussion."
            )
        )
    ]
    groupchat = GroupChat(
        agents=agents,
        max_round=8,
        speaker_selection_method="round_robin"
    )
    manager = GroupChatManager(groupchat)
    print("\n===== ROUND ROBIN GROUP CHAT RESULT =====\n")
    async def run_team():
        await agents[0].a_initiate_chat(manager, message=scalper_prompt)
        for msg in groupchat.messages:
            print(f"---------- {msg.get('name', 'Agent')} ----------\n{msg.get('content', '')}\n")
    asyncio.run(run_team())

# --- DATA LOADING & PROMPT ---
csv_path = os.path.join(os.path.dirname(__file__), '../stock_csv_files/stock_analysis_comparison.csv')
def get_stock_prompt():
    df = pd.read_csv(csv_path)
    df['ticker'] = df['ticker'].astype(str).str.replace(r'^\d+\.\s*', '', regex=True)
    allowed_tickers = df['ticker'].tolist()
    allowed_tickers_str = ', '.join(allowed_tickers)
    all_columns = [
        'ticker', 'overall_trend', 'overall_trend_confidence', 'volume_trend', 'volume_trend_signal', 'latest_price',
        'trend_based_on_levels', 'short_term_bullish_trend_confidence', 'Price_Above_9_20_33_200_EMAs',
        'long_term_bullish_trend_confidence', '20_SMA_Above_200_SMA (Long-term)', 'Price_Above_50_150_200_SMA_and_SMA_Alignment',
        'MACD', 'MACD_Trend', 'RSI', 'RSI_Trend', 'Bollinger_Low', 'Bollinger_High', 'Aroon_Down', 'Aroon_Up',
        'Stoch_K', 'Stoch_D', 'ADX', 'ATR', 'VWAP', 'Candlestick_Pattern', 'Fibonacci_Levels', 'Bullish_Score', 'Bullish_Score_Trend'
    ]
    df_small = df[[col for col in all_columns if col in df.columns]]
    stock_data_markdown = df_small.to_markdown(index=False)
    prompt = (
        f"Allowed tickers: [{allowed_tickers_str}]. Only pick from these. "
        "You are a 1-minute scalper. Ignore long-term signals and trends. Only recommend a stock if there is a clear, immediate, high-probability upward move based on the table below. If none, say 'NO BUY: No clear scalp opportunity.' "
        "Keep your answer under 2 sentences. Do not mention long-term trends or downside protection. "
        "Analyze EVERY stock below using ONLY the provided data. "
        "The table below contains ALL the columns you need. Do not reference columns that are not present in the table. "
        "Pick the single best stock to buy for a scalper (1-minute) trade. "
        "Focus on ALL of the following (higher is better unless noted):\n"
        "- Bullish_Score_Trend (Rising is best, Unchanged is ok, Falling is bad)\n"
        "- volume_trend (High is best), volume_trend_signal (Bullish/Weak Bullish is better)\n"
        "- MACD (Bullish preferred), MACD_Trend (Rising preferred)\n"
        "- RSI (Rising preferred, but avoid if overbought/Sell Signal)\n"
        "- Price_Above_9_20_33_200_EMAs (Bullish preferred)\n"
        "- VWAP (price above VWAP is better, indicates buyers market)\n"
        "- overall_trend (Bullish preferred), overall_trend_confidence (higher is better)\n"
        "- short_term_bullish_trend_confidence (higher is better)\n"
        "- trend_based_on_levels (Bullish preferred)\n"
        "- latest_price (for context only)\n"
        "- Any other columns that indicate a strong, fast upward move.\n"
        "If multiple stocks are close, break ties by Bullish_Score, then volume, then confidence.\n"
        f"Stock data (table):\n{stock_data_markdown}\n\n"
        "Respond ONLY as: BUY <ticker>: <reasoning> or NO BUY: No clear scalp opportunity."
    )
    return prompt

if __name__ == "__main__":
    scalper_prompt = get_stock_prompt()
    run_round_robin_groupchat(scalper_prompt)











