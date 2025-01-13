# WebUI Extensions

A collection of tools, functions, and utilities for extending Open WebUI capabilities.

## Extensions

### DeepSeek API Integration (v1.0.0)

A comprehensive Python implementation for integrating DeepSeek's V3 API with Open WebUI. This module provides a robust interface to DeepSeek's advanced language model capabilities while maintaining full compatibility with Open WebUI's plugin architecture.

#### Key Features

1. **Model Capabilities**
   - DeepSeek V3 Integration with 671B MoE parameters
   - High Performance: 60 tokens/second throughput
   - Large Context: 8192 token context window
   - Vision Support: Process and analyze image inputs

2. **Output Formats**
   - Text Generation: Standard text completion responses
   - JSON Mode: Validated JSON output with schema enforcement
   - Streaming: Real-time token-by-token responses
   - Function Calling: Define and execute external functions

3. **Resource Management**
   - Context Caching: Efficient token reuse with hit/miss tracking
   - Cost Optimization: Reduced costs for cached content (¥0.1/M vs ¥1/M)
   - Usage Metrics: Detailed token and cost tracking
   - Request Retries: Automatic retry for unstable operations

4. **Advanced Controls**
   - Temperature Control: Task-specific randomness settings
   - Top-K Sampling: Control output token diversity
   - Top-P Filtering: Nucleus sampling for better quality
   - Beta Features: Support for prefix and FIM capabilities

5. **Error Handling**
   - Status Codes: Specific error codes with descriptions
   - Validation: Input parameter and response validation
   - Recovery: Automatic recovery from transient errors
   - Logging: Comprehensive error and usage logging

#### Usage

The DeepSeek API integration is implemented in `deepseek_v3_1.py`. To use it:

1. Set up your API key in the options
2. Initialize the pipe function in Open WebUI
3. Make requests in the chat by selecting "deepseek-chat"

#### Configuration

The DeepSeek integration supports various configuration options through the `Valves` class:

- `temperature`: Controls randomness (0.0-2.0)
- `top_k`: Limits sampling to top-k tokens
- `top_p`: Controls diversity via nucleus sampling (0.0-1.0)
- `max_tokens`: Maximum tokens to generate (1-8192)
- `stream`: Enable streaming responses
- `response_format`: Output format (text/json_object)
- `enable_beta_features`: Enable experimental features
- `enable_context_cache`: Enable context caching
- `function_call_retry`: Number of retries for unstable function calls

## License

Free for non-commercial use.

## Author

Amyn Porbanderwala  
Email: amyn@porbanderwala.com  
Website: www.porbanderwala.com

## Contributing

More extensions and tools will be added soon. Contributions are welcome!
