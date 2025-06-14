import requests
from bs4 import BeautifulSoup
from autogen import (
    GroupChat, GroupChatManager, UserProxyAgent, AssistantAgent, ConversableAgent, register_function
)
import re

# LLM config for Ollama
OLLAMA_URL = "http://localhost:11434/v1"
OLLAMA_KEY = "NO_KEY_NEEDED"

config_qwen = {
    "api_type": "ollama",
    "model": "qwen3:4b",
    "base_url": OLLAMA_URL,
    "api_key": OLLAMA_KEY,
    "temperature": 0.1
}

config_llama3 = {
    "api_type": "ollama",
    "model": "llama3-groq-tool-use:latest",
    "base_url": OLLAMA_URL,
    "api_key": OLLAMA_KEY,
    "temperature": 0.1
}

# Global variables to store the original prompt and teams
original_prompt = ""
team1_name = ""
team2_name = ""

# --- Base DuckDuckGo Search Function ---
def _base_duckduckgo_search(query: str, max_results: int = 4, search_focus: str = "", agent_task: str = "") -> str:
    """
    Base search function that handles the actual web scraping.
    """
    global original_prompt
    
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    try:
        response = requests.post(url, data=params, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        
        for i, a in enumerate(soup.find_all('a', class_='result__a', limit=max_results), 1):
            title = a.get_text().strip()
            link = a.get('href', '')
            
            # Fetch page content
            try:
                page_resp = requests.get(link, headers=headers, timeout=10)
                page_soup = BeautifulSoup(page_resp.text, "html.parser")
                
                # Remove script and style elements
                for script in page_soup(["script", "style", "nav", "header", "footer", "aside"]):
                    script.decompose()
                
                # Try to get main content
                main_content = (
                    page_soup.find('main') or 
                    page_soup.find('article') or 
                    page_soup.find('div', class_=re.compile(r'content|article|main', re.I)) or
                    page_soup.find('body')
                )
                
                if main_content:
                    content = main_content.get_text(separator=' ', strip=True)
                else:
                    content = page_soup.get_text(separator=' ', strip=True)
                
                # Limit content length to avoid token limits
                content = content[:2500] if len(content) > 2500 else content
                
            except Exception as e:
                content = f"Error fetching content from {link}: {str(e)}"
            
            results.append(f"RESULT {i}:\nTitle: {title}\nURL: {link}\nContent: {content}\n")
        
        # Format all results with specialized task reminder
        formatted_results = "\n" + "="*60 + "\n".join(results)
        formatted_results += f"\n{'='*60}\n"
        formatted_results += f"🎯 SPECIALIZED SEARCH FOCUS: {search_focus}\n"
        formatted_results += f"📋 YOUR SPECIFIC TASK: {agent_task}\n"
        formatted_results += f"🏆 ORIGINAL MATCHUP: '{original_prompt}'\n"
        formatted_results += f"{'='*60}\n"
        formatted_results += "⚠️  IMPORTANT REMINDERS:\n"
        formatted_results += "✅ ANALYZE ONLY information relevant to your specific task above\n"
        formatted_results += "❌ IGNORE: TV schedules, how to watch, advertisements, unrelated news\n"
        formatted_results += "🎯 YOUR GOAL: Determine how your findings affect which team will win\n"
        formatted_results += f"{'='*60}\n"
        
        return formatted_results
        
    except Exception as e:
        return f"Search failed: {str(e)}\n\n🎯 TASK REMINDER: Focus on {search_focus} for '{original_prompt}'"

# --- Specialized Search Functions for Each Agent ---

def search_odds_and_statistics(matchup: str) -> str:
    """Search for betting odds, lines, and statistical comparisons"""
    global team1_name, team2_name
    query = f"{matchup} betting odds lines spread over under expert picks predictions statistics"
    
    return _base_duckduckgo_search(
        query=query,
        max_results=4,
        search_focus="BETTING ODDS & STATISTICS",
        agent_task="Find betting lines, point spreads, over/under, expert picks, win probabilities, and statistical comparisons between teams"
    )

def search_injuries_and_roster(matchup: str) -> str:
    """Search for injury reports and roster updates"""
    global team1_name, team2_name
    query = f"{matchup} injury report roster updates player availability questionable out starting lineup"
    
    return _base_duckduckgo_search(
        query=query,
        max_results=4,
        search_focus="INJURIES & ROSTER STATUS",
        agent_task="Find key player injuries, questionable players, roster changes, suspensions, and how they impact team strength"
    )

def search_team_form_momentum(matchup: str) -> str:
    """Search for recent team performance and momentum"""
    global team1_name, team2_name
    query = f"{matchup} recent form last 5 games streak momentum head to head record home away performance"
    
    return _base_duckduckgo_search(
        query=query,
        max_results=4,
        search_focus="TEAM FORM & MOMENTUM",
        agent_task="Find recent win/loss records, current streaks, head-to-head history, home/away performance, and team momentum"
    )

def search_general_matchup_info(matchup: str) -> str:
    """Search for general matchup information and news"""
    global team1_name, team2_name
    query = f"{matchup} preview analysis prediction who will win matchup breakdown"
    
    return _base_duckduckgo_search(
        query=query,
        max_results=4,
        search_focus="GENERAL MATCHUP ANALYSIS",
        agent_task="Find general matchup previews, expert analysis, and overall predictions about which team will win"
    )

# --- Specialized Agents with Their Own Search Functions ---

# Search Agent (coordinates initial search)
search_coordinator = ConversableAgent(
    name="SearchCoordinator",
    system_message=(
        "You are the search coordinator. As soon as it is your turn, you MUST IMMEDIATELY call the search_general_matchup_info function as your VERY FIRST ACTION. Do NOT analyze, summarize, or respond in any way until you have called this function and received the results. After receiving the search results, briefly summarize the key findings from your search results and provide an initial assessment. Focus on general predictions and expert opinions you found."
    ),
    llm_config=config_qwen,
    human_input_mode="NEVER"
)

register_function(
    search_general_matchup_info,
    caller=search_coordinator,
    executor=search_coordinator,
    name="search_general_matchup_info",
    description="Search for general matchup information, previews, and predictions."
)

# Odds Analyst with specialized search
odds_analyst = ConversableAgent(
    name="OddsAnalyst",
    system_message=(
        "You are the odds and statistics expert. As soon as it is your turn, you MUST IMMEDIATELY call the search_odds_and_statistics function as your VERY FIRST ACTION. Do NOT analyze, summarize, or respond in any way until you have called this function and received the results. After receiving the search results, analyze which team the betting markets favor, point spreads and their implications, expert picks and win probabilities, and statistical advantages. Provide a clear assessment of which team has the betting/statistical edge."
    ),
    llm_config=config_qwen,
    human_input_mode="NEVER"
)

register_function(
    search_odds_and_statistics,
    caller=odds_analyst,
    executor=odds_analyst,
    name="search_odds_and_statistics",
    description="Search for betting odds, lines, spreads, and statistical comparisons for the matchup."
)

# Injury Analyst with specialized search
injury_analyst = ConversableAgent(
    name="InjuryAnalyst", 
    system_message=(
        "You are the injury and roster expert. As soon as it is your turn, you MUST IMMEDIATELY call the search_injuries_and_roster function as your VERY FIRST ACTION. Do NOT analyze, summarize, or respond in any way until you have called this function and received the results. After receiving the search results, analyze key players who are injured or questionable, the impact of missing players on team performance, roster advantages one team may have, and how injuries affect the matchup outcome. Determine which team has the healthier, more complete roster."
    ),
    llm_config=config_qwen,
    human_input_mode="NEVER"
)

register_function(
    search_injuries_and_roster,
    caller=injury_analyst,
    executor=injury_analyst,
    name="search_injuries_and_roster",
    description="Search for injury reports, roster updates, and player availability for the matchup."
)

# Form Analyst with specialized search
form_analyst = ConversableAgent(
    name="FormAnalyst",
    system_message=(
        "You are the team form and momentum expert. As soon as it is your turn, you MUST IMMEDIATELY call the search_team_form_momentum function as your VERY FIRST ACTION. Do NOT analyze, summarize, or respond in any way until you have called this function and received the results. After receiving the search results, analyze recent win/loss records and current streaks, team momentum and confidence levels, head-to-head historical performance, and home/away advantages. Determine which team has better recent form and momentum going into this matchup."
    ),
    llm_config=config_qwen,
    human_input_mode="NEVER"
)

register_function(
    search_team_form_momentum,
    caller=form_analyst,
    executor=form_analyst,
    name="search_team_form_momentum",
    description="Search for recent team performance, form, momentum, and head-to-head records."
)

# Final Predictor (synthesizes all information)
final_predictor = AssistantAgent(
    name="FinalPredictor",
    system_message=(
        "You are the final decision maker. After all specialists have completed their searches and analysis, "
        "synthesize their findings to make the final prediction. Structure your response as follows:\n\n"
        "1. FIRST SENTENCE: '[TEAM NAME] is more likely to win this matchup'\n"
        "2. SECOND SENTENCE: State the primary deciding factor\n"
        "3. List 2-3 key supporting points from the different analyses\n"
        "4. Mention any concerning factors for your predicted winner\n"
        "5. End with 'PREDICTION APPROVED' to conclude the analysis\n\n"
        "Base your decision on the combined insights from odds, injuries, form, and general analysis."
    ),
    llm_config=config_qwen
)

# User Proxy
user_proxy = UserProxyAgent(
    name="User",
    system_message="You coordinate the sports prediction process and facilitate discussion between analysts.",
    human_input_mode="NEVER",
    code_execution_config=False,
    is_termination_msg=lambda msg: "PREDICTION APPROVED" in msg.get("content", "")
)

# --- Group Chat Setup ---
def create_prediction_chat():
    agents = [user_proxy, search_coordinator, odds_analyst, injury_analyst, form_analyst, final_predictor]
    
    groupchat = GroupChat(
        agents=agents,
        messages=[],
        max_round=20,  # Increased to allow for search calls
        speaker_selection_method="round_robin",
        allow_repeat_speaker=False
    )
    
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=config_qwen,
        is_termination_msg=lambda msg: "PREDICTION APPROVED" in msg.get("content", "")
    )
    
    return manager

def extract_teams(prompt):
    """Extract team names from vs prompt"""
    vs_patterns = [r'\s+vs\.?\s+', r'\s+v\.?\s+', r'\s+@\s+', r'\s+at\s+', r'\s+against\s+']
    
    for pattern in vs_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            teams = re.split(pattern, prompt, flags=re.IGNORECASE)
            if len(teams) >= 2:
                team1 = teams[0].strip()
                # Extract team name, removing extra words like dates, game numbers
                team2_full = teams[1].strip()
                team2 = team2_full.split()[0].strip()  # Get first word after vs
                return team1, team2
    
    return None, None

def main():
    global original_prompt, team1_name, team2_name
    
    print("\n🏀 ENHANCED SPORTS PREDICTION SYSTEM 🏈")
    print("="*60)
    print("Each agent will perform their own specialized search:")
    print("🔍 SearchCoordinator: General matchup overview")
    print("📊 OddsAnalyst: Betting lines, spreads, expert picks")
    print("🏥 InjuryAnalyst: Player injuries, roster updates") 
    print("📈 FormAnalyst: Recent performance, momentum, head-to-head")
    print("🎯 FinalPredictor: Synthesize all findings into final prediction")
    print("="*60)
    print("Enter a matchup (e.g., 'Lakers vs Warriors', 'Chiefs at Bills')")
    print("="*60)
    
    user_input = input("Enter matchup: ").strip()
    
    if not user_input:
        print("Please enter a valid matchup!")
        return
    
    # Store original prompt and extract teams globally
    original_prompt = user_input
    team1_name, team2_name = extract_teams(user_input)
    
    if team1_name and team2_name:
        print(f"\n🏆 Analyzing: {team1_name} vs {team2_name}")
        print(f"📝 Full Query: {user_input}")
    else:
        print(f"\n🏆 Analyzing: {user_input}")
    
    # Create the group chat
    manager = create_prediction_chat()

    # Start the prediction process - each agent will search when it's their turn
    initial_message = (
        f"Analyze this matchup and predict the winner: {user_input}\n\n"
        f"PROCESS:\n"
        f"1. SearchCoordinator: Search for general matchup information\n"
        f"2. OddsAnalyst: Search for betting odds and statistics\n" 
        f"3. InjuryAnalyst: Search for player injuries and roster status\n"
        f"4. FormAnalyst: Search for recent performance and momentum\n"
        f"5. FinalPredictor: Synthesize all findings into final prediction\n\n"
        f"Each specialist must call their search function first, then provide analysis."
    )

    print(f"\n🤖 Starting Agent-Driven Analysis...")
    print("Each agent will search and analyze their specialty area...")
    print("="*60)

    try:
        user_proxy.initiate_chat(
            manager,
            message=initial_message
        )
    except Exception as e:
        print(f"Error during analysis: {e}")
        
    print("\n" + "="*60)
    print("🏆 AGENT-DRIVEN PREDICTION COMPLETE!")
    print("All specialists searched and analyzed their areas!")

if __name__ == "__main__":
    main()
