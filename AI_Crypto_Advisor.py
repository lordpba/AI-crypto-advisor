# L6: Multi-agent Collaboration for Financial Analysis (Cryptocurrency Market)

# Install necessary libraries if not already installed (uncomment the below lines if running locally)
# !pip install crewai==0.28.8 crewai_tools==0.1.6 langchain_community==0.0.29

# Warning control
import warnings
warnings.filterwarnings('ignore')

# Import libraries, APIs, and LLM
from crewai import Agent, Task, Crew, Process
import os
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

import dotenv
dotenv.load_dotenv()

groq = ChatGroq(temperature=0.5, model_name="llama3-8b-8192") # mixtral-8x7b-32768 - llama3-70b-8192 - gemma-7b-it - llama3-8b-8192
gpt35_turbo = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
llm = groq

# Initialize crewAI tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Create Agents
data_analyst_agent = Agent(
    role="Data Analyst",
    goal="Monitor and analyze cryptocurrency market data in real-time to identify trends and predict market movements.",
    backstory="Specializing in cryptocurrency markets, this agent uses statistical modeling and machine learning to provide crucial insights. With a knack for data, the Data Analyst Agent is the cornerstone for informing trading decisions.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
    llm=llm
)

trading_strategy_agent = Agent(
    role="Trading Strategy Developer",
    goal="Develop and test various trading strategies based on insights from the Data Analyst Agent.",
    backstory="Equipped with a deep understanding of cryptocurrency markets and quantitative analysis, this agent devises and refines trading strategies. It evaluates the performance of different approaches to determine the most profitable and risk-averse options.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
    llm=llm
)

execution_agent = Agent(
    role="Trade Advisor",
    goal="Suggest optimal trade execution strategies based on approved trading strategies.",
    backstory="This agent specializes in analyzing the timing, price, and logistical details of potential trades. By evaluating these factors, it provides well-founded suggestions for when and how trades should be executed to maximize efficiency and adherence to strategy.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
    llm=llm
)

risk_management_agent = Agent(
    role="Risk Advisor",
    goal="Evaluate and provide insights on the risks associated with potential trading activities.",
    backstory="Armed with a deep understanding of risk assessment models and market dynamics, this agent scrutinizes the potential risks of proposed trades. It offers a detailed analysis of risk exposure and suggests safeguards to ensure that trading activities align with the firm’s risk tolerance.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
    llm=llm
)

# Create Tasks
data_analysis_task = Task(
    description=(
        "Continuously monitor and analyze cryptocurrency market data for the selected cryptocurrency ({crypto_selection}). "
        "Use statistical modeling and machine learning to identify trends and predict market movements."
    ),
    expected_output=(
        "Insights and alerts about significant market opportunities or threats for {crypto_selection}."
    ),
    agent=data_analyst_agent,
)

strategy_development_task = Task(
    description=(
        "Develop and refine trading strategies based on the insights from the Data Analyst and "
        "user-defined risk tolerance ({risk_tolerance}). Consider trading preferences ({trading_strategy_preference})."
    ),
    expected_output=(
        "A set of potential trading strategies for {crypto_selection} that align with the user's risk tolerance."
    ),
    agent=trading_strategy_agent,
)

execution_planning_task = Task(
    description=(
        "Analyze approved trading strategies to determine the best execution methods for {crypto_selection}, "
        "considering current market conditions and optimal pricing."
    ),
    expected_output=(
        "Detailed execution plans suggesting how and when to execute trades for {crypto_selection}."
    ),
    agent=execution_agent,
)

risk_assessment_task = Task(
    description=(
        "Evaluate the risks associated with the proposed trading strategies and execution plans for {crypto_selection}. "
        "Provide a detailed analysis of potential risks and suggest mitigation strategies."
    ),
    expected_output=(
        "A comprehensive risk analysis report detailing potential risks and mitigation recommendations for {crypto_selection}."
    ),
    agent=risk_management_agent,
)

# Create the Crew
financial_trading_crew = Crew(
    agents=[data_analyst_agent, 
            trading_strategy_agent, 
            execution_agent, 
            risk_management_agent],
    
    tasks=[data_analysis_task, 
           strategy_development_task, 
           execution_planning_task, 
           risk_assessment_task],
    
    manager_llm=gpt35_turbo,
    process=Process.hierarchical,
    verbose=True
)

# Set the inputs for the execution of the crew
financial_trading_inputs = {
    'crypto_selection': 'CRO',  # BTC for Bitcoin, can be changed to any other cryptocurrency
    'initial_capital': '1000',
    'risk_tolerance': 'Medium',
    'trading_strategy_preference': 'Month Trading',
    'news_impact_consideration': True
}

# Execute the Crew (this execution will take some time to run)
result = financial_trading_crew.kickoff(inputs=financial_trading_inputs)

# save the result to a file
with open('crypto_trading_result.txt', 'w') as f:
    f.write(result)
