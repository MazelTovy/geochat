#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/scratch/sx2490/Logs/model_server/static_server.log')
    ]
)
logger = logging.getLogger("static_server")

# CORS headers for handling cross-origin requests
class CORSRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"), **kwargs)
        
    def end_headers(self):
        # Add CORS headers to allow cross-origin requests
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Expose-Headers', '*')
        super().end_headers()
        
    def do_OPTIONS(self):
        # Handle OPTIONS request for CORS preflight
        self.send_response(200)
        self.end_headers()
        
    def log_message(self, format, *args):
        # Override to use our logger instead of stderr
        logger.info("%s - %s" % (self.address_string(), format % args))
        
def run_server(port=8080):
    """Run the static file server"""
    server_address = ('', port)
    
    # Use ThreadingHTTPServer for better performance with multiple requests
    class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
        pass
    
    httpd = ThreadingHTTPServer(server_address, CORSRequestHandler)
    
    logger.info(f"Starting static server on port {port}")
    logger.info(f"Serving files from {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')}")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
    finally:
        httpd.server_close()
        logger.info("Server stopped")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a static file server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs('/scratch/sx2490/Logs/model_server', exist_ok=True)
    
    # Run the server
    run_server(port=args.port) 