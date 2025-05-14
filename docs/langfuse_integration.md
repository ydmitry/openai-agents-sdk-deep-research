# Langfuse Integration with OpenAI Agents SDK

This document explains how to use Langfuse to trace and monitor the OpenAI Agents SDK in your applications.

## Prerequisites

1. Install required packages:
   ```bash
   pip install "pydantic-ai[logfire]"
   pip install nest_asyncio
   ```

2. Create a Langfuse account and obtain API keys:
   - Visit [Langfuse](https://langfuse.com) and create an account
   - Create a new project
   - Get your Public and Secret keys from the project settings

3. Set up environment variables in your `.env` file:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
   LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
   LANGFUSE_HOST=https://cloud.langfuse.com  # or https://us.cloud.langfuse.com for US region
   ENVIRONMENT=development  # Options: development, staging, production
   ```

## Usage

### Basic Usage with Web Search Example

To run the web search example with Langfuse tracing enabled:

```bash
python examples/web_search_example.py "What is the current status of Mars missions?" --enable-langfuse
```

Or in interactive mode:

```bash
python examples/web_search_example.py --interactive --enable-langfuse
```

### Programmatic Usage

```python
from deep_research.utils import load_web_search_agent

# Enable Langfuse tracing
agent, run_agent = load_web_search_agent(enable_langfuse=True)

# Use the agent
result = await run_agent("Your query here")
```

### Advanced Usage with Custom Traces

For more advanced tracing with custom attributes:

```python
import asyncio
from deep_research.utils import load_web_search_agent
from deep_research.langfuse_integration import setup_langfuse, create_trace

# Set up Langfuse
setup_langfuse()

# Create agent without auto-tracing
agent, run_agent = load_web_search_agent(enable_langfuse=False)

# Add custom trace with user_id and session_id
async def traced_query(query, user_id, session_id):
    with create_trace(
        name="Custom-Trace",
        user_id=user_id,
        session_id=session_id,
        tags=["custom_tag"],
        environment="production"
    ) as span:
        # Set custom input (safely)
        if span is not None:
            span.set_attribute("input.value", query)
        
        # Run the agent
        result = await run_agent(query)
        
        # Set custom output (safely)
        if span is not None:
            span.set_attribute("output.value", result)
        
        return result

# Use the custom traced function
result = asyncio.run(traced_query(
    "What is the latest news?", 
    user_id="user-123", 
    session_id="session-456"
))
```

### Testing the Integration

You can run a simple test to verify that your Langfuse integration is working:

```bash
python examples/test_langfuse_integration.py
```

If successful, you'll see a message indicating the test passed and you can check your Langfuse dashboard for the trace.

## Viewing Traces in Langfuse

1. Go to your Langfuse project dashboard
2. Navigate to the "Traces" section
3. You will see all your agent interactions with:
   - Full LLM prompts and responses
   - Tool usage and parameters
   - Timing information
   - Tokens used and cost estimates

## Troubleshooting

If you encounter issues with Langfuse tracing:

1. Verify your API keys are correct
2. Check that you're using the correct Langfuse host for your region
3. Enable DEBUG logging to see more information:
   ```bash
   python examples/web_search_example.py "Your query" --enable-langfuse --log-level DEBUG
   ```
4. Ensure you have the latest version of `pydantic-ai[logfire]` installed

### Common Errors

1. `AttributeError: 'NoneType' object has no attribute 'set_attribute'`
   - This means the span is None. Check that setup_langfuse() was called and returned True.

2. `_AgnosticContextManager' object has no attribute 'set_attribute'`
   - If you see this error, make sure to use the updated `create_trace` function, not the deprecated `trace_with_user_session`.

## OpenTelemetry and Nested Asyncio

The integration uses OpenTelemetry for tracing and nest_asyncio to handle nested event loops (common in Jupyter notebooks). If you're having issues:

1. Make sure you're calling `setup_langfuse()` before any tracing functions
2. Check that you have the latest versions of the required packages
3. Try running the test script to debug basic functionality 