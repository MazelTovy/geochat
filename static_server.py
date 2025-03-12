#!/usr/bin/env python
# -*- coding: utf-8 -*-

import http.server
import socketserver
import os
import sys
import argparse

def create_server(port, directory):
    """Create a simple HTTP server to serve static files"""
    handler = http.server.SimpleHTTPRequestHandler
    
    # Change working directory to specified directory
    os.chdir(directory)
    
    # Create server
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Static file server started at http://localhost:{port}")
        print(f"Serving directory: {os.getcwd()}")
        print("Press Ctrl+C to stop the server")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")
            sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start a simple HTTP static file server')
    parser.add_argument('-p', '--port', type=int, default=8080, help='Server port (default: 8080)')
    parser.add_argument('-d', '--directory', type=str, default=os.path.join(os.path.dirname(__file__), 'static'),
                        help='Directory to serve static files from (default: static folder in the script directory)')
    
    args = parser.parse_args()
    
    # Ensure directory exists
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
        print(f"Created directory: {args.directory}")
    
    create_server(args.port, args.directory) 