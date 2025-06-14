"""
Group chat with auto speaker selection. User inputs a prompt, agents discuss automatically.
"""
from autogen import (
    GroupChat, GroupChatManager, UserProxyAgent, AssistantAgent, ConversableAgent,
    register_function
)
import requests
from bs4 import BeautifulSoup

# LLM configs (replace with your actual LLM config dicts as needed)
OLLAMA_URL = "http://localhost:11434"  # Update if your Ollama server is elsewhere
OLLAMA_KEY = ""  # Add your key if needed, else leave as empty string

config_deepseek = {
    "api_type": "ollama",
    "model": "deepseek-r1:1.5b-qwen-distill-q4_K_M",
    "base_url": OLLAMA_URL,
    "api_key": OLLAMA_KEY,
    "temperature": 0.1
}
config_qwen3 = {
    "api_type": "ollama",
    "model": "qwen3:4b",
    "base_url": OLLAMA_URL,
    "api_key": OLLAMA_KEY,
    "temperature": 0.1
}
config_gemma3_4b = {
    "api_type": "ollama",
    "model": "gemma3:4b-it-q4_K_M",
    "base_url": OLLAMA_URL,
    "api_key": OLLAMA_KEY,
    "temperature": 0.1
}

# Define agents
user_proxy = UserProxyAgent(
    name="User",
    system_message="You are the user. You provide the initial prompt and can ask follow-up questions.",
    human_input_mode="ALWAYS",
    code_execution_config={"use_docker": False}
)

research_agent = AssistantAgent(
    name="ResearchAgent",
    system_message="You are a research expert. Research the topic and provide relevant information. The current user prompt is: {prompt}",
    llm_config=config_deepseek
)

factcheck_agent = AssistantAgent(
    name="FactCheckAgent",
    system_message="You are a fact-checking expert. Verify the information provided and point out any inaccuracies. The current user prompt is: {prompt}",
    llm_config=config_qwen3
)

summary_agent = AssistantAgent(
    name="SummaryAgent",
    system_message="You are a summarization expert. Summarize the discussion and provide a concise conclusion. The current user prompt is: {prompt}",
    llm_config=config_gemma3_4b
)

# --- DuckDuckGo Search Tool Function and Agent ---
def duckduckgo_search(query: str, max_results: int = 3) -> list:
    """
    Search DuckDuckGo and fetch the main content of the top results.
    Args:
        query (str): The search query.
        max_results (int): Number of top results to fetch.
    Returns:
        list: A list of dicts with 'title', 'link', and 'content' for each result.
    """
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.post(url, data=params, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for a in soup.find_all('a', class_='result__a', limit=max_results):
        title = a.get_text()
        link = a['href']
        # Fetch page content
        try:
            page_resp = requests.get(link, headers=headers, timeout=10)
            page_soup = BeautifulSoup(page_resp.text, "html.parser")
            main = page_soup.find('main')
            if main:
                content = main.get_text(separator=' ', strip=True)
            else:
                content = page_soup.get_text(separator=' ', strip=True)
        except Exception as e:
            content = f"Error fetching {link}: {e}"
        results.append({
            "title": title,
            "link": link,
            "content": content
        })
    # Add the reminder line at the bottom of the last result's content
    user_prompt = query
    if results:
        results[-1]["content"] += f"\n\nRemember, the user prompt is: {user_prompt}\nIgnore any information not related to the prompt."
    return results

duckduckgo_search_agent_llm_config = {
    "api_type": "ollama",
    "model": "llama3-groq-tool-use:latest",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "temperature": 0.1
}

duckduckgo_search_agent = ConversableAgent(
    name="DuckDuckGoSearchAgent",
    system_message=(
        'You are a web search agent. You have the capability to perform real-time web searches using the DuckDuckGo search tool '
        '(by calling the "duckduckgo_search" function). For every user question, always use this tool to find the answer, regardless of the topic. '
        'Extract and return ONLY the specific information requested. Do NOT summarize or analyze unless explicitly asked. '
        'When done, say TERMINATE.'
    ),
    llm_config=duckduckgo_search_agent_llm_config,
    human_input_mode="NEVER"
)

register_function(
    duckduckgo_search,
    caller=duckduckgo_search_agent,
    executor=duckduckgo_search_agent,
    name="duckduckgo_search",
    description="Search DuckDuckGo and fetch the main content of the top N results for a query."
)

def search_duckduckgo(query: str, max_results: int = 3):
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.post(url, data=params, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for a in soup.find_all('a', class_='result__a', limit=max_results):
        title = a.get_text()
        link = a['href']
        results.append((title, link))
    return results

def fetch_page_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        main = soup.find('main')
        if main:
            text = main.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        return f"Error fetching {url}: {e}"

# Create group chat with auto speaker selection
agents = [user_proxy, duckduckgo_search_agent, research_agent, factcheck_agent, summary_agent]
groupchat = GroupChat(
    agents=agents,
    messages=[],
    max_round=10,
    speaker_selection_method="auto"  # Options: "auto" (agents decide who speaks next), "round_robin" (fixed order), "random" (random agent), or a custom function. See AutoGen docs for more.
)
manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=config_deepseek  # Manager can use any LLM
)

if __name__ == "__main__":
    print("\n--- Group Chat (auto speaker selection) ---\n")
    print("Enter your prompt to start the discussion:")
    user_input = input("Prompt: ")
    user_proxy.initiate_chat(
        manager,
        message=user_input
    )
