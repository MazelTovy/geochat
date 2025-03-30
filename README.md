# LLaMA-3-8B Chat System with RAG Support

This project implements a complete process for deploying a LLaMA-3-8B chat system on an HPC cluster and interacting with it through a web interface. It includes enhanced Retrieval Augmented Generation (RAG) capabilities powered by FlashRAG.

## File Structure

```
geochat_gs/
├── model_server.py      # Model server (FastAPI)
├── static_server.py     # Static file server
├── static/              # Static frontend files
│   └── index.html       # Chat interface
├── rag/                 # RAG (Retrieval Augmented Generation) module
│   ├── __init__.py      # Package initialization
│   ├── retriever.py     # Document retrieval components (FlashRAG integration)
│   └── document_processor.py # Document processing utilities
├── nyc_schools_data/    # NYC Schools JSON dataset directory
│   └── processed/       # Directory for processed documents
├── init_rag.py          # RAG initialization script
├── test_torch.py        # PyTorch test script
└── README.md            # This documentation
```

## FlashRAG Integration

This project now integrates FlashRAG, a high-performance Retrieval Augmented Generation framework. FlashRAG provides:

- Faster and more accurate document retrieval
- Efficient vector indexing for semantic search
- Enhanced relevance scoring and document ranking
- Optimized context integration for better answers

The system is pre-configured to use NYC Schools data from the `nyc_schools_data/` directory as the knowledge base.

## Deployment Steps

### 1. Preparation

- Ensure model files are ready and accessible
- If you're using LLaMA-3-8B, you need a Hugging Face account with appropriate access
- Ensure required Python packages are installed (these are automatically handled by the SBATCH script)

### 2. Start the Model Server 

Simply submit the SBATCH job to start everything. The script will automatically:
- Install FlashRAG if needed
- Initialize the RAG system with NYC Schools data (if not already initialized)
- Start the model server

```bash
# Create log directory if it doesn't exist
mkdir -p /scratch/sx2490/Logs/model_server

# Submit the job
sbatch /scratch/sx2490/geochat_gs/run_model_server.SBATCH
```

After submitting the job, you can check the job status with:

```bash
squeue -u $USER
```

When the job starts running, check the output logs to get node information:

```bash
cat /scratch/sx2490/Logs/model_server/<job_id>.out
```

The logs will show which node and port the model service is running on.

### 3. Set up SSH Port Forwarding

From your local machine, set up SSH port forwarding to map the remote server's port to your local:

```bash
ssh -L 8000:<compute_node>:8000 sx2490@greene.hpc.nyu.edu
```

where `<compute_node>` is the compute node name assigned by SLURM (e.g. `g0123`).

### 4. Start the Frontend Service (Optional)

If you need to run the frontend locally rather than remotely, download `static/index.html` to your local machine and open it with a browser.

Alternatively, start the static file server on the remote server:

```bash
cd /scratch/sx2490/geochat_gs
python static_server.py -p 8080
```

Then set up another SSH port forwarding:

```bash
ssh -L 8080:<login_node>:8080 sx2490@greene.hpc.nyu.edu
```

### 5. Access the Chat Interface

Open your browser and visit:

- If running the frontend locally: simply open the HTML file
- If running the static server remotely: visit http://localhost:8080

In the settings, ensure the API URL points to the correct address: `http://localhost:8000/api/chat`

## Using RAG with NYC Schools Data

The system is now configured to use the NYC Schools data with FlashRAG's retrieval capabilities:

1. Enable RAG by toggling the "Use Retrieval (RAG)" switch in the web interface
2. Ask questions about NYC schools, such as:
   - "What schools are in district 6?"
   - "Tell me about the high schools in the Bronx"
   - "What's the contact information for P.S. 194?"
3. The system will retrieve relevant information from the NYC Schools dataset to provide accurate answers

You can customize the number of documents retrieved by adjusting the "Retrieval Top K" parameter in the interface settings.

## LLaMA-3-8B Model

This system uses Meta's LLaMA-3-8B model for generating responses. Key features:

- 8 billion parameter model with strong performance
- Text generation capabilities competitive with many commercial models
- Efficient inference with optional quantization
- Support for chat-style interactions

To adjust model parameters, you can use the following settings in the web interface:
- Temperature: Controls randomness in generation (higher = more random)
- Top P: Controls diversity of token selection
- Max Length: Maximum number of tokens to generate

## RAG (Retrieval Augmented Generation) with FlashRAG

The enhanced RAG system powered by FlashRAG works as follows:

1. **Document Processing**: NYC Schools JSON files are converted to structured text, divided into chunks, and processed for efficient retrieval
2. **Embedding & Indexing**: Document chunks are embedded using the E5 model and indexed for semantic search
3. **Retrieval**: When a user asks a question, FlashRAG's semantic search identifies the most relevant document chunks
4. **Augmented Generation**: The retrieved information is integrated into the prompt, helping the model provide accurate information about NYC schools

## Adding Custom Data

To use your own data with the FlashRAG-powered RAG system:

1. Place your documents in the `nyc_schools_data/` directory or create your own data directory
2. Modify the `PROCESSED_DIR` path in the `run_model_server.SBATCH` script to point to your data directory
3. Restart the model server by submitting the SBATCH script again

Supported file formats include: `.txt`, `.md`, `.json` (with custom formatting for NYC Schools JSON structure), with more formats planned for future updates.

## Behind the Scenes

The `run_model_server.SBATCH` script performs the following tasks:

1. Sets up the environment with required dependencies
2. Checks if FlashRAG is installed and installs it if necessary
3. Checks if RAG initialization has been performed (by looking for the index and corpus files)
4. If needed, automatically runs the initialization process to:
   - Process the NYC Schools data into document chunks
   - Build the FlashRAG retrieval index
5. Starts the model server

All these steps happen automatically when you submit the SBATCH job, eliminating the need to run separate commands.

## Common Issues

### 1. Port Already in Use

If the port is already in use, modify the `PORT` environment variable in `run_model_server.SBATCH` and choose an unused port.

### 2. FlashRAG Integration Issues

If you encounter problems with FlashRAG:
- Check the SBATCH job output logs for any error messages
- Ensure that your HPC environment has internet access to download FlashRAG if needed
- Try resubmitting the job to trigger the automatic initialization again

### 3. Model Loading Errors

If you encounter issues with model loading:
- Ensure you have access to the LLaMA-3-8B model on Hugging Face
- Check if you have enough GPU memory (A100 with at least 40GB recommended)
- Consider enabling 4-bit quantization by uncommenting the related lines in model_server.py

### 4. Performance Optimization

For production environments, consider these optimizations:

- Use model quantization to reduce memory usage (uncomment the load_in_4bit option)
- Implement batch processing to improve throughput
- Use FlashRAG's caching mechanism to avoid redundant retrievals
- Adjust chunk size and overlap in the SBATCH script for optimal retrieval performance

### 5. Security Considerations

- Avoid using `allow_origins=["*"]` in production environments; restrict to specific domains
- Consider adding authentication and authorization mechanisms
- Limit request rates per user to prevent abuse

## Custom Development

To adjust or extend functionality, refer to these files:

- `model_server.py`: Modify model loading and inference logic
- `static/index.html`: Adjust frontend interface and user experience
- `run_model_server.SBATCH`: Adjust compute resource allocation and initialization parameters
- `rag/retriever.py`: Customize FlashRAG integration
- `rag/document_processor.py`: Extend document processing capabilities

## Important Notes

- Ensure compliance with HPC cluster usage policies
- Release resources promptly when no longer in use
- Monitor GPU memory usage to avoid OOM errors
- Follow LLaMA-3 licensing terms when using the model 