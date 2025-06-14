# PostgreSQL Collector for Clarification Agent

The PostgreSQL collector enables storing search results from the clarification agent with OpenAI embeddings in a PostgreSQL database using the pgvector extension.

## Features

- 🗄️ **Persistent Storage**: Store search results in PostgreSQL for later analysis
- 🧠 **Vector Embeddings**: Automatic OpenAI embedding generation using `text-embedding-3-small`
- 🔑 **Session Management**: Automatic UUID generation for each session
- 🛡️ **Graceful Degradation**: Stores results without embeddings if OpenAI API fails
- 🔍 **Semantic Search**: Enable similarity searches using pgvector

## Prerequisites

1. **PostgreSQL with pgvector extension**
2. **Python dependencies**:
   ```bash
   pip install psycopg2-binary openai python-dotenv
   ```
3. **Environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

## Database Setup

### 1. Run the Migration

Create the `search_results` table:

```bash
python create_search_results_table.py
```

This creates:
- Table: `search_results` with columns: `id`, `session_id`, `created_at`, `text`, `embedding`
- Indexes for performance: session_id, created_at, and vector similarity

### 2. Table Structure

```sql
CREATE TABLE search_results (
    id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    text TEXT NOT NULL,
    embedding vector(1536)  -- OpenAI text-embedding-3-small dimension
);
```

## Usage

### Command Line - Single Request

Store search results for a single analysis:

```bash
python src/clarification_agent/run.py "Build me a web app" --postgres
```

### Chat Mode with PostgreSQL

Start an interactive chat session with PostgreSQL storage:

```bash
python src/clarification_agent/run.py --chat --postgres
```

### Advanced Options

Combine with other features:

```bash
# With specific model and handoff enabled
python src/clarification_agent/run.py "Design ML pipeline" \
    --postgres \
    --model gpt-4o \
    --enable-handoff \
    --temperature 0.2
```

## Database Configuration

The collector uses the same database configuration as `demo_pgvector.py`:

```python
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 5433,
    'database': 'my-deep-research',
    'user': 'postgres',
    'password': 'secret'
}
```

To modify, edit the `DB_CONFIG` in the `create_postgres_collector()` function.

## Querying Stored Results

### View All Sessions

```sql
SELECT DISTINCT session_id, COUNT(*) as result_count, MIN(created_at) as started_at
FROM search_results 
GROUP BY session_id 
ORDER BY started_at DESC;
```

### View Results for a Session

```sql
SELECT id, created_at, 
       LEFT(text, 100) || '...' as text_preview,
       CASE WHEN embedding IS NOT NULL THEN 'Yes' ELSE 'No' END as has_embedding
FROM search_results 
WHERE session_id = 'your-session-uuid'
ORDER BY created_at;
```

### Semantic Similarity Search

Find similar results using cosine similarity:

```sql
SELECT id, text, 
       1 - (embedding <=> $1) as similarity
FROM search_results 
WHERE embedding IS NOT NULL
ORDER BY similarity DESC
LIMIT 10;
```

## Error Handling

The collector gracefully handles various failure scenarios:

- **Missing Dependencies**: Clear error message if `psycopg2` not installed
- **Database Connection Issues**: Logs errors and continues without storage
- **OpenAI API Failures**: Stores results without embeddings
- **Missing API Key**: Logs warning and stores without embeddings

## Session Management

Each collector instance automatically generates a unique session UUID:

```python
collector, session_id = create_postgres_collector()
print(f"Session ID: {session_id}")
```

Session IDs are displayed in chat mode for reference:

```
🗄️  Session ID: 550e8400-e29b-41d4-a716-446655440000
Search results will be stored in PostgreSQL with embeddings.
```

## Integration with Sequential Search

The PostgreSQL collector is designed to work with the sequential search agent that can be triggered via handoff. When `--enable-handoff` is used, the clarification agent may transition to search mode, and the search results will be stored in PostgreSQL.

## Example Workflow

1. **Setup Database**:
   ```bash
   python create_search_results_table.py
   ```

2. **Start Chat Session**:
   ```bash
   python src/clarification_agent/run.py --chat --postgres --enable-handoff
   ```

3. **Interact with Agent**:
   ```
   You: Build me a data pipeline for real-time analytics
   Assistant: [Asks clarifying questions, then searches and stores results]
   ```

4. **Query Results**:
   ```sql
   SELECT * FROM search_results WHERE session_id = 'your-session-id';
   ```

## Troubleshooting

### Common Issues

1. **"psycopg2 is required"**: Install with `pip install psycopg2-binary`
2. **Connection refused**: Check PostgreSQL is running on port 5433
3. **Extension not found**: Ensure pgvector extension is installed
4. **No embeddings stored**: Check `OPENAI_API_KEY` environment variable

### Debug Logging

Enable debug logging to see detailed information:

```bash
python src/clarification_agent/run.py "test" --postgres --log-level DEBUG
```

## Performance Considerations

- **Embedding Generation**: ~100-500ms per request to OpenAI API
- **Database Insertion**: <10ms for typical search results
- **Vector Index**: Automatically created for efficient similarity searches
- **Batch Processing**: Consider batching for high-volume usage

## Security Notes

- Database credentials are hardcoded for demo purposes
- In production, use environment variables or secure configuration
- OpenAI API key should be stored securely (`.env` file or environment variable) 