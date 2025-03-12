# LLaMA-3-8B Chat System with RAG Support

This project implements a complete process for deploying a LLaMA-3-8B chat system on an HPC cluster and interacting with it through a web interface. It also includes Retrieval Augmented Generation (RAG) capabilities.

## File Structure

```
geochat_gs/
├── model_server.py      # Model server (FastAPI)
├── static_server.py     # Static file server
├── static/              # Static frontend files
│   └── index.html       # Chat interface
├── rag/                 # RAG (Retrieval Augmented Generation) module
│   ├── __init__.py      # Package initialization
│   ├── retriever.py     # Document retrieval components
│   ├── document_processor.py # Document processing utilities
│   └── rag_manager.py   # RAG integration manager
├── documents/           # Directory for knowledge base documents
│   └── processed/       # Directory for processed documents
├── init_rag.py          # RAG initialization script
├── test_torch.py        # PyTorch test script
└── README.md            # This documentation
```

## Deployment Steps

### 1. Preparation

- Ensure model files are ready and accessible
- If you're using LLaMA-3-8B, you need a Hugging Face account with appropriate access
- Add documents to the `documents/` directory if you plan to use RAG
- Ensure required Python packages are installed (if unsure, add installation commands before running the SBATCH script)

### 2. Initialize RAG (Optional)

If you plan to use the RAG feature, first initialize the document processing:

```bash
cd /home/sx2490/geochat_gs
python init_rag.py --docs-dir documents --chunk-size 500 --chunk-overlap 50
```

### 3. Start the Model Server

Submit a job to start the model server:

```bash
mkdir -p /home/sx2490/Logs/model_server  # Ensure log directory exists
sbatch /home/sx2490/scripts/run_model_server.SBATCH
```

After submitting the job, you can check the job status with:

```bash
squeue -u $USER
```

When the job starts running, check the output logs to get node information:

```bash
cat /home/sx2490/Logs/model_server/<job_id>.out
```

The logs will show which node and port the model service is running on.

### 4. Set up SSH Port Forwarding

From your local machine, set up SSH port forwarding to map the remote server's port to your local:

```bash
ssh -L 8000:<compute_node>:8000 sx2490@greene.hpc.nyu.edu
```

where `<compute_node>` is the compute node name assigned by SLURM (e.g. `g0123`).

### 5. Start the Frontend Service (Optional)

If you need to run the frontend locally rather than remotely, download `static/index.html` to your local machine and open it with a browser.

Alternatively, start the static file server on the remote server:

```bash
cd /home/sx2490/geochat_gs
python static_server.py -p 8080
```

Then set up another SSH port forwarding:

```bash
ssh -L 8080:<login_node>:8080 sx2490@greene.hpc.nyu.edu
```

### 6. Access the Chat Interface

Open your browser and visit:

- If running the frontend locally: simply open the HTML file
- If running the static server remotely: visit http://localhost:8080

In the settings, ensure the API URL points to the correct address: `http://localhost:8000/api/chat`

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

## RAG (Retrieval Augmented Generation)

The RAG system enhances the model's responses by retrieving relevant information from a document knowledge base:

1. **Document Processing**: Documents are divided into chunks and processed for efficient retrieval
2. **Embedding & Indexing**: Document chunks are embedded and indexed for semantic search
3. **Retrieval**: When a user asks a question, relevant document chunks are retrieved
4. **Augmented Generation**: The retrieved information is integrated into the prompt, producing more informed responses

To use RAG:
1. Place your reference documents in the `documents/` directory
2. Run the initialization script to process documents
3. Toggle the "Use Retrieval (RAG)" switch in the web interface

## Common Issues

### 1. Port Already in Use

If the port is already in use, modify the `PORT` environment variable in `run_model_server.SBATCH` and choose an unused port.

### 2. Job Termination

HPC cluster jobs have time limits. When the time specified by the `--time` parameter is reached, the job will be automatically terminated. For longer runs, adjust this parameter accordingly.

### 3. Model Loading Errors

If you encounter issues with model loading:
- Ensure you have access to the LLaMA-3-8B model on Hugging Face
- Check if you have enough GPU memory (A100 with at least 40GB recommended)
- Consider enabling 4-bit quantization by uncommenting the related lines in model_server.py

### 4. Performance Optimization

For production environments, consider these optimizations:

- Use model quantization to reduce memory usage (uncomment the load_in_4bit option)
- Implement batch processing to improve throughput
- Use specialized model serving frameworks (like Triton Inference Server)
- Add request queue mechanisms to handle high concurrency

### 5. Security Considerations

- Avoid using `allow_origins=["*"]` in production environments; restrict to specific domains
- Consider adding authentication and authorization mechanisms
- Limit request rates per user to prevent abuse

## Custom Development

To adjust or extend functionality, refer to these files:

- `model_server.py`: Modify model loading and inference logic
- `static/index.html`: Adjust frontend interface and user experience
- `run_model_server.SBATCH`: Adjust compute resource allocation
- `rag/` directory: Enhance the RAG implementation

## Important Notes

- Ensure compliance with HPC cluster usage policies
- Release resources promptly when no longer in use
- Monitor GPU memory usage to avoid OOM errors
- Follow LLaMA-3 licensing terms when using the model 