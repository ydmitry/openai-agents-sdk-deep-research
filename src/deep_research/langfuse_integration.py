"""
Langfuse integration for OpenAI Agents SDK.

This module provides utilities to integrate Langfuse observability with the OpenAI Agents SDK.
"""
import os
import base64
import logging
import contextlib

# Configure logging
logger = logging.getLogger(__name__)

# Global variable to track if Langfuse is set up
_langfuse_initialized = False

def setup_langfuse(public_key=None, secret_key=None, host=None):
    """
    Set up Langfuse for OpenAI Agents SDK tracing.
    
    Args:
        public_key: Langfuse public key (defaults to LANGFUSE_PUBLIC_KEY env var)
        secret_key: Langfuse secret key (defaults to LANGFUSE_SECRET_KEY env var)
        host: Langfuse host URL (defaults to LANGFUSE_HOST env var or https://cloud.langfuse.com)
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    global _langfuse_initialized
    
    try:
        # Use provided keys or get from environment
        public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        host = host or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        if not public_key or not secret_key:
            logger.warning("Langfuse keys not provided. Tracing will not be enabled.")
            return False
        
        # Build Basic Auth header
        langfuse_auth = base64.b64encode(
            f"{public_key}:{secret_key}".encode()
        ).decode()
        
        # Configure OpenTelemetry endpoint & headers
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host}/api/public/otel"
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"
        
        # Import and configure logfire
        try:
            import logfire
            import nest_asyncio
            
            # Apply nest_asyncio to allow nested event loops (needed for Jupyter notebooks)
            nest_asyncio.apply()
            
            # Configure logfire instrumentation
            logfire.configure(
                service_name='deep_research',
                send_to_logfire=False,
            )
            # This method automatically patches the OpenAI Agents SDK
            logfire.instrument_openai_agents()
            
            # Set up global tracer provider
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            
            # Set the global tracer provider
            trace_provider = TracerProvider()
            trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
            trace.set_tracer_provider(trace_provider)
            
            _langfuse_initialized = True
            logger.info("Successfully set up Langfuse tracing for OpenAI Agents SDK")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import required packages: {e}")
            logger.error("Make sure to install 'pydantic-ai[logfire]' package")
            return False
            
    except Exception as e:
        logger.error(f"Error setting up Langfuse: {e}", exc_info=True)
        return False

@contextlib.contextmanager
def create_trace(name="Deep-Research-Trace", user_id=None, session_id=None, tags=None, environment=None):
    """
    Simplified version of trace_with_user_session that handles errors better.
    
    Args:
        name: Name of the trace
        user_id: User identifier for Langfuse
        session_id: Session identifier for grouping traces
        tags: List of tags to add to the trace
        environment: Environment name (e.g., 'dev', 'prod')
    
    Yields:
        A span that can have attributes set on it, or None if tracing is not available
    """
    if not _langfuse_initialized:
        logger.warning("Langfuse not initialized. Call setup_langfuse() first.")
        yield None
        return
        
    try:
        from opentelemetry import trace
        
        # Get tracer
        tracer = trace.get_tracer(__name__)
        
        # Start span
        with tracer.start_as_current_span(name) as span:
            # Set standard attributes
            if user_id:
                span.set_attribute("langfuse.user.id", user_id)
            if session_id:
                span.set_attribute("langfuse.session.id", session_id)
            if tags:
                span.set_attribute("langfuse.tags", tags)
            if environment:
                span.set_attribute("langfuse.environment", environment)
                
            # Yield span for caller to use
            yield span
            
    except Exception as e:
        logger.error(f"Error in create_trace: {e}")
        yield None

def trace_with_user_session(user_id=None, session_id=None, tags=None, environment=None):
    """
    Create an OpenTelemetry span with Langfuse attributes.
    
    Args:
        user_id: User identifier for Langfuse
        session_id: Session identifier for grouping traces
        tags: List of tags to add to the trace
        environment: Environment name (e.g., 'dev', 'prod')
    
    Returns:
        A context manager that can be used with a 'with' statement
    """
    try:
        from opentelemetry import trace
        
        tracer = trace.get_tracer(__name__)
        
        class TraceContextManager:
            def __init__(self, name, user_id, session_id, tags, environment):
                self.name = name
                self.user_id = user_id
                self.session_id = session_id
                self.tags = tags
                self.environment = environment
                self.span_ctx = None
            
            def __enter__(self):
                # Start a new span and get the context manager
                self.span_ctx = tracer.start_as_current_span(self.name)
                # Enter the context
                self.span = self.span_ctx.__enter__()
                
                # Set Langfuse attributes
                if self.user_id:
                    self.span.set_attribute("langfuse.user.id", self.user_id)
                if self.session_id:
                    self.span.set_attribute("langfuse.session.id", self.session_id)
                if self.tags:
                    self.span.set_attribute("langfuse.tags", self.tags)
                if self.environment:
                    self.span.set_attribute("langfuse.environment", self.environment)
                
                return self.span
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.span_ctx:
                    self.span_ctx.__exit__(exc_type, exc_val, exc_tb)
        
        return TraceContextManager(
            name="Deep-Research-Trace", 
            user_id=user_id, 
            session_id=session_id, 
            tags=tags, 
            environment=environment
        )
        
    except ImportError as e:
        logger.error(f"OpenTelemetry not properly configured: {e}")
        logger.error("Make sure to call setup_langfuse() first.")
        
        # Fallback context manager that does nothing
        class DummyContextManager:
            def __enter__(self):
                logger.warning("Using dummy trace context manager - no tracing will occur")
                return None
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        
        return DummyContextManager() 