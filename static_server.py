#!/usr/bin/env python
# -*- coding: utf-8 -*-

import http.server
import socketserver
import os
import sys
import argparse
import logging
from urllib.parse import urlparse, parse_qs

# Configure logging
os.makedirs("/scratch/sx2490/Logs/model_server", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/scratch/sx2490/Logs/model_server/static_server.log')
    ]
)
logger = logging.getLogger("static_server")

# Root directory containing static files
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Custom handler for static files with connection monitoring
class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)
        
    def log_message(self, format, *args):
        # Override to use our logger instead
        logger.info(f"{self.client_address[0]} - {format % args}")
        
    def do_GET(self):
        """Handle GET requests - add connection status indicator to index.html"""
        logger.info(f"GET request: {self.path} from {self.client_address[0]}")
        
        # Special handling for index.html to inject connection info
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            
            # Read the index.html file
            try:
                with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
                    content = f.read()
                    
                # Add connection status information
                status_script = """
                <script>
                // Add connection status indicator
                document.addEventListener('DOMContentLoaded', function() {
                    const statusDiv = document.createElement('div');
                    statusDiv.id = 'connection-status';
                    statusDiv.style.position = 'fixed';
                    statusDiv.style.bottom = '10px';
                    statusDiv.style.right = '10px';
                    statusDiv.style.padding = '5px 10px';
                    statusDiv.style.borderRadius = '5px';
                    statusDiv.style.fontSize = '12px';
                    statusDiv.style.zIndex = '1000';
                    
                    // Initially show checking status
                    statusDiv.style.backgroundColor = '#f8d7da';
                    statusDiv.textContent = 'Checking API connection...';
                    document.body.appendChild(statusDiv);
                    
                    // Check API health
                    fetch('http://localhost:8000/api/health')
                        .then(response => {
                            if (response.ok) { return response.json(); }
                            throw new Error('API server not responding');
                        })
                        .then(data => {
                            if (data.model_loaded) {
                                statusDiv.style.backgroundColor = '#d4edda';
                                statusDiv.textContent = 'API Connected: Model Ready';
                            } else {
                                statusDiv.style.backgroundColor = '#fff3cd';
                                statusDiv.textContent = 'API Connected: Model Loading';
                            }
                            console.log('API health check:', data);
                        })
                        .catch(error => {
                            statusDiv.style.backgroundColor = '#f8d7da';
                            statusDiv.textContent = 'API Connection Failed';
                            console.error('API connection error:', error);
                            
                            // Show error details in console
                            console.error('Error details:', error.message);
                        });
                });
                
                // Add key combo to show debug info (Ctrl+Shift+D)
                document.addEventListener('keydown', function(e) {
                    if (e.ctrlKey && e.shiftKey && e.key === 'D') {
                        checkAndShowDebugInfo();
                    }
                });
                
                function checkAndShowDebugInfo() {
                    console.log('Showing debug information');
                    
                    // Create debug panel if it doesn't exist
                    let debugInfo = document.getElementById('debug-info');
                    if (!debugInfo) {
                        debugInfo = document.createElement('div');
                        debugInfo.id = 'debug-info';
                        debugInfo.style.position = 'fixed';
                        debugInfo.style.top = '10px';
                        debugInfo.style.right = '10px';
                        debugInfo.style.backgroundColor = '#f8f9fa';
                        debugInfo.style.border = '1px solid #ddd';
                        debugInfo.style.padding = '10px';
                        debugInfo.style.fontFamily = 'monospace';
                        debugInfo.style.fontSize = '12px';
                        debugInfo.style.zIndex = '1000';
                        debugInfo.style.maxWidth = '400px';
                        
                        document.body.appendChild(debugInfo);
                    }
                    
                    // Update debug info content
                    debugInfo.innerHTML = `
                        <h4>Connection Debug Info</h4>
                        <p>Static Server: localhost:8080</p>
                        <p>API Server: localhost:8000</p>
                        <p>Browser: ${navigator.userAgent}</p>
                        <p>Time: ${new Date().toLocaleString()}</p>
                        <button onclick="document.getElementById('debug-info').style.display='none'">Close</button>
                        <button onclick="checkAPIStatus()">Check API</button>
                    `;
                    
                    // Show the debug panel
                    debugInfo.style.display = 'block';
                }
                
                function checkAPIStatus() {
                    fetch('http://localhost:8000/api/health')
                        .then(response => response.json())
                        .then(data => {
                            alert('API Status: ' + JSON.stringify(data, null, 2));
                        })
                        .catch(error => {
                            alert('API Connection Error: ' + error.message);
                        });
                }
                </script>
                """
                
                # Insert the script before the closing body tag
                content = content.replace('</body>', f'{status_script}</body>')
                
                self.wfile.write(content.encode())
                logger.info(f"Served modified index.html with connection status script")
            except Exception as e:
                logger.error(f"Error serving index.html: {str(e)}")
                self.send_error(500, f"Error serving index.html: {str(e)}")
        else:
            # Use the default handler for other files
            super().do_GET()

def create_server(port, directory=STATIC_DIR):
    """Create a simple HTTP server to serve static files"""
    handler = CustomHTTPRequestHandler
    
    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True
    
    # Create server
    with socketserver.TCPServer(("", port), handler) as httpd:
        logger.info(f"Static file server started at http://localhost:{port}")
        logger.info(f"Serving directory: {directory}")
        logger.info("Press Ctrl+C to stop the server")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("\nServer stopped")
            sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start a simple HTTP static file server')
    parser.add_argument('-p', '--port', type=int, default=8080, help='Server port (default: 8080)')
    parser.add_argument('-d', '--directory', type=str, default=STATIC_DIR,
                     help='Directory to serve static files from (default: static folder in the script directory)')
    
    args = parser.parse_args()
    
    # Ensure directory exists
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
        logger.info(f"Created directory: {args.directory}")
    
    create_server(args.port, args.directory) 