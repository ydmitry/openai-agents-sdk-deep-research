# Deep Research using OpenAI Agents SDK

A comprehensive suite of intelligent research agents built on the OpenAI Agents SDK. This project provides modular, reusable agents for different aspects of research and information gathering, from clarifying ambiguous requests to conducting multi-step web research.

## 🚀 Features

- **Modular Agent Architecture**: Individual agents that can work independently or together
- **Multiple Research Strategies**: From simple web searches to complex multi-step investigations
- **Intelligent Query Clarification**: Automatically identify and address ambiguities in user requests
- **Flexible Output Formats**: JSON, text files, console output, or custom callbacks
- **Citation Preservation**: Automatic source tracking and formatted citations
- **PostgreSQL Integration**: Index all search results in database for better post-research questions
- **Command-Line Interface**: Ready-to-use runners for all agents
- **Seamless Integration**: Agents designed to work together in research pipelines

## 📦 Available Agents

### 1. Clarification Agent
**Purpose**: Analyzes user requests to identify gaps, ambiguities, and missing information.

- ✨ Pure LLM analysis - no external tools required
- 📝 Generates numbered clarification questions with guidance
- 🔄 Automatic handoff to sequential search agent
- 🗣️ Interactive chat mode for real-time clarification

[📖 Full Documentation](src/clarification_agent/README.md)

### 2. Sequential Search Agent
**Purpose**: Performs sophisticated multi-step research by orchestrating multiple web searches.

- 🧠 Intelligent research planning and strategy
- 🔍 Uses WebSearch Agent as an internal tool
- 📚 Synthesizes results from multiple searches
- 🔗 Automatic citation preservation and formatting
- 📊 Comprehensive analysis and reporting

[📖 Full Documentation](src/sequential_search_agent/README.md)

### 3. WebSearch Agent
**Purpose**: Standalone web search agent with flexible collection capabilities.

- 🌐 Direct web search functionality
- 📄 Multiple collection methods (JSON, text, memory)
- ⚙️ Configurable models and settings
- 🛠️ Factory pattern for easy integration
- 📋 Structured result handling

[📖 Full Documentation](src/websearch_agent/README.md)

## 🚀 Quick Start

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ydmitry/openai-agents-sdk-deep-research.git
   cd openai-agents-sdk-deep-research
   ```

2. **Set up your environment**
   ```bash
   # Set your OpenAI API key
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Install dependencies**
   ```bash
   # Install OpenAI Agents SDK and other requirements
   pip install -r requirements.txt  # if available
   ```

### Basic Usage Examples

#### Simple Web Search
```bash
python src/websearch_agent/run.py "Latest developments in AI"
```

#### Complex Multi-Step Research
```bash
python src/sequential_search_agent/run.py \
    "Evolution of machine learning from 2020 to 2024: key developments and future trends" \
    --format text --output ml_evolution.txt
```

#### Interactive Clarification & Research
```bash
# Start with clarification, then automatically proceed to research
python src/clarification_agent/run.py --chat
```

#### Save Results to Files
```bash
# JSON output with citations
python src/sequential_search_agent/run.py \
    "Compare renewable energy vs fossil fuels in 2024" \
    --output energy_research.json

# Text format for easy reading
python src/websearch_agent/run.py \
    "Market trends in electric vehicles" \
    --format text --output ev_trends.txt
```

## 🔄 Agent Workflows

### Research Pipeline
```
User Request → Clarification Agent → Sequential Search Agent → Results
                     ↓                        ↓
             Clarification Questions    WebSearch Agent (internal)
```

### Individual Agent Usage
```
Direct Query → [Any Agent] → Formatted Results
```

## 🛠️ Programmatic Usage

### Basic Agent Creation
```python
from src.websearch_agent.agent import make_search_agent
from src.sequential_search_agent.agent import make_sequential_search_agent
from src.clarification_agent import make_clarification_agent
from agents import Runner

# Custom result handler
def handle_results(query: str, answer: str):
    print(f"Query: {query}")
    print(f"Answer: {answer}")

# Create and use agents
search_agent = make_search_agent(handle_results)
sequential_agent = make_sequential_search_agent(handle_results)
clarification_agent = make_clarification_agent(handle_results)

# Run research
result = await Runner.run(sequential_agent, "Your research query")
```

### Collection Methods
```python
# JSON file collection
from src.websearch_agent.agent import create_json_file_collector
collector = create_json_file_collector("results.json")

# Text file collection  
from src.websearch_agent.agent import create_text_file_collector
collector = create_text_file_collector("research_log.txt")

# Memory collection
from src.websearch_agent.agent import create_memory_collector
collector, retriever = create_memory_collector()
```

## 📋 Use Cases

### Research & Analysis
- **Comparative Studies**: "Compare X vs Y" → Automatic multi-perspective analysis
- **Market Research**: "Analyze the Z market" → Comprehensive market intelligence
- **Technology Trends**: "Evolution of X technology" → Historical and current analysis
- **Policy Analysis**: "Impact of Y policy" → Multi-faceted impact assessment

### Information Gathering
- **Current Events**: "What's happening with X?" → Latest news and developments
- **Academic Research**: Complex queries requiring multiple sources
- **Due Diligence**: Comprehensive background research
- **Trend Analysis**: Multi-temporal data gathering and synthesis

### Interactive Research
- **Ambiguous Requests**: Start with clarification for better targeting
- **Iterative Investigation**: Build understanding through guided questions
- **Stakeholder Research**: Different perspectives through clarifying questions

## 🔧 Configuration

All agents support common configuration options:

- **Model Selection**: `--model gpt-4o`, `gpt-4o-mini`, etc.
- **Temperature Control**: `--temperature 0.1` to `1.0`
- **Output Formats**: `--format json` or `text`
- **Logging Levels**: `--log-level DEBUG|INFO|WARNING|ERROR`
- **File Output**: `--output filename.json`

### PostgreSQL Database Integration

The project includes PostgreSQL integration for indexing search results:

- **Automatic Storage**: All search results can be automatically stored in PostgreSQL
- **Post-Research Queries**: Query your research history for follow-up questions
- **Data Persistence**: Maintain a searchable archive of all research conducted
- **Docker Setup**: Includes `docker-compose.yml` for easy PostgreSQL deployment

```bash
# Start PostgreSQL database
docker-compose up -d

# Initialize database tables
python create_search_results_table.py
```

See [POSTGRES_COLLECTOR.md](POSTGRES_COLLECTOR.md) for detailed setup instructions.

## 📁 Project Structure

```
my-deep-research/
├── README.md                          # This file
├── src/
│   ├── clarification_agent/
│   │   ├── __init__.py
│   │   ├── agent.py                   # Clarification agent logic
│   │   ├── run.py                     # CLI runner
│   │   └── README.md                  # Detailed documentation
│   ├── sequential_search_agent/
│   │   ├── __init__.py
│   │   ├── agent.py                   # Sequential search orchestration
│   │   ├── run.py                     # CLI runner
│   │   └── README.md                  # Detailed documentation
│   └── websearch_agent/
│       ├── __init__.py
│       ├── agent.py                   # Web search functionality
│       ├── run.py                     # CLI runner
│       └── README.md                  # Detailed documentation
└── utils/                             # Shared utilities (if any)
```

## 🤝 Integration Examples

### Clarification → Research Pipeline
```python
# Step 1: Clarify the request
clarification_agent = make_clarification_agent(collect_questions)
questions = await Runner.run(clarification_agent, vague_request)

# Step 2: Get user answers (in real application)
refined_request = incorporate_user_answers(vague_request, user_answers)

# Step 3: Perform comprehensive research
research_agent = make_sequential_search_agent(collect_results)
results = await Runner.run(research_agent, refined_request)
```

### Multi-Agent Research System
```python
agents = {
    'clarify': make_clarification_agent(clarification_handler),
    'search': make_search_agent(search_handler), 
    'research': make_sequential_search_agent(research_handler)
}

# Route requests to appropriate agents based on complexity
if is_ambiguous(request):
    await Runner.run(agents['clarify'], request)
elif is_complex(request):
    await Runner.run(agents['research'], request)  
else:
    await Runner.run(agents['search'], request)
```

## 📊 Output Examples

### JSON Format (with citations)
```json
{
  "timestamp": "2024-01-15T10:30:45.123456",
  "query": "AI developments 2024",
  "answer": "Key developments in AI during 2024 include...",
  "sources": [
    "1. [Article Title](https://example.com/article)",
    "2. https://research.example.com/ai-trends",
    "3. Industry Report on AI Advances"
  ]
}
```

### Text Format
```
================================================================================
TIMESTAMP: 2024-01-15 10:30:45
QUERY: Machine learning evolution 2020-2024
================================================================================
ANSWER:
The evolution of machine learning from 2020 to 2024 has been marked by...

**Sources:**
1. [Advances in Neural Networks](https://example.com/neural-networks)
2. https://ml-research.org/trends-2024
3. Industry Analysis: ML Market Report
================================================================================
```

## 🆘 Getting Help

- **Individual Agent Documentation**: See README files in each agent's directory
- **Command Line Help**: Run any agent with `--help` flag
- **Logging**: Use `--log-level DEBUG` for detailed troubleshooting
- **Issues**: Check agent-specific documentation for common problems

## 🔮 Future Enhancements

- **Multi-modal Search**: Integration with image and document search
- **Agent Chaining**: Automated workflows between multiple agents
- **Result Caching**: Avoid duplicate searches for efficiency
- **Advanced Analytics**: Statistical analysis of research patterns
- **Custom Tools**: Plugin system for specialized research tools

---

**Built with ❤️ using OpenAI Agents SDK**