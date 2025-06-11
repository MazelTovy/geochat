"""
Simple Document Retriever for NYC Schools Data
This module handles the retrieval of school information including available seats
"""

import os
import json
import csv
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("simple_retriever")

class SimpleRetriever:
    """
    Simple retriever for NYC schools data including seat availability
    """
    
    def __init__(self, documents_dir: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.documents_dir = documents_dir
        self.embedding_model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = None
        self.school_seats = {}  # 学校剩余席位信息
        
        logger.info(f"SimpleRetriever initialized with {embedding_model}")
    
    def load_documents(self):
        """Load all documents from the data directory"""
        # Load JSON files (school information)
        json_files = [f for f in os.listdir(self.documents_dir) if f.endswith('.json')]
        
        for json_file in json_files:
            try:
                file_path = os.path.join(self.documents_dir, json_file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Extract school information
                school_id = data.get('basic_info', {}).get('school_id', json_file.replace('.json', ''))
                school_name = data.get('basic_info', {}).get('name', 'Unknown School')
                
                # Create document content
                content = self._format_school_info(data)
                
                self.documents.append({
                    'id': school_id,
                    'name': school_name,
                    'content': content,
                    'data': data,
                    'type': 'school_info'
                })
                
            except Exception as e:
                logger.error(f"Error loading {json_file}: {str(e)}")
        
        # Load CSV file (seat availability)
        csv_file = os.path.join(self.documents_dir, 'site_level_seats_available_tracker_example.csv')
        if os.path.exists(csv_file):
            self._load_seat_availability(csv_file)
        
        # Create embeddings for all documents
        if self.documents:
            contents = [doc['content'] for doc in self.documents]
            self.embeddings = self.embedding_model.encode(contents)
            logger.info(f"Loaded {len(self.documents)} documents and created embeddings")
        else:
            logger.warning("No documents loaded")
    
    def _format_school_info(self, data: Dict) -> str:
        """Format school data into searchable text"""
        parts = []
        
        # Basic info
        basic_info = data.get('basic_info', {})
        parts.append(f"School: {basic_info.get('name', 'Unknown')}")
        parts.append(f"School ID: {basic_info.get('school_id', 'Unknown')}")
        
        # Location information
        location = basic_info.get('location', {})
        if location:
            full_address = location.get('full_address', '')
            if full_address:
                parts.append(f"Address: {full_address}")
            else:
                street = location.get('street_address', '')
                city_state_zip = location.get('city_state_zip', '')
                if street and city_state_zip:
                    parts.append(f"Address: {street}, {city_state_zip}")
        
        # General info
        gen_info = data.get('metrics', {}).get('Gen', {})
        if gen_info:
            parts.append(f"Principal: {gen_info.get('Principal', 'Unknown')}")
            parts.append(f"Grades: {gen_info.get('Grades served', 'Unknown')}")
            parts.append(f"Enrollment: {gen_info.get('Enrollment', 'Unknown')}")
            parts.append(f"Website: {gen_info.get('School website', 'Unknown')}")
            parts.append(f"Admission methods: {gen_info.get('Kindergarten admissions methods', '')} {gen_info.get('Middle school admissions methods', '')}")
        
        # Demographics
        demog = data.get('metrics', {}).get('Sch-demog', {})
        if demog:
            parts.append("Demographics:")
            for key, value in demog.items():
                if value and value != "N/A":
                    parts.append(f"  {key}: {value}")
        
        # Programs
        programs = data.get('metrics', {}).get('Program', {})
        if programs:
            program_list = []
            for key, value in programs.items():
                if value and value != "N/A":
                    program_list.append(f"{key}: {value}")
            if program_list:
                parts.append("Programs: " + ", ".join(program_list))
        
        # Performance metrics
        perf = data.get('metrics', {}).get('Perf', {})
        if perf:
            parts.append("Performance:")
            for key, value in perf.items():
                if value:
                    parts.append(f"  {key}: {value}")
        
        # Attendance
        attendance = data.get('metrics', {}).get('Attendance', {})
        if attendance:
            parts.append("Attendance:")
            for key, value in attendance.items():
                if value:
                    parts.append(f"  {key}: {value}")
        
        # School overview
        overview = data.get('metrics', {}).get('Directory-overview', {})
        if overview:
            for key, value in overview.items():
                if value and len(value) > 50:  # Only include substantial descriptions
                    parts.append(f"School Description: {value[:500]}...")  # Truncate long descriptions
        
        return "\n".join(parts)
    
    def _load_seat_availability(self, csv_path: str):
        """Load seat availability data from CSV"""
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dbn = row.get('dbn', '')
                    name = row.get('name', '')
                    seats_left = row.get('seats_left', '0')
                    program = row.get('type', '')
                    
                    # Store seat information by school ID
                    if dbn not in self.school_seats:
                        self.school_seats[dbn] = []
                    
                    self.school_seats[dbn].append({
                        'name': name,
                        'program': program,
                        'seats_left': int(seats_left) if seats_left.isdigit() else 0,
                        'address': row.get('address_final', ''),
                        'zip_code': row.get('zip_code', '')
                    })
            
            logger.info(f"Loaded seat availability for {len(self.school_seats)} schools")
            
        except Exception as e:
            logger.error(f"Error loading seat availability: {str(e)}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents based on query"""
        if not self.documents or self.embeddings is None:
            logger.warning("No documents available for retrieval")
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['score'] = float(similarities[idx])
            
            # Add seat availability (always include this information)
            school_id = doc.get('id', '')
            if school_id in self.school_seats:
                doc['seat_availability'] = self.school_seats[school_id]
                
                # Add seat info to content for better context
                seat_info = "\n\nAvailable Seats:"
                for seat in self.school_seats[school_id]:
                    seat_info += f"\n- {seat['program']}: {seat['seats_left']} seats left"
                doc['content'] += seat_info
            else:
                # No seat data available
                doc['seat_availability'] = []
                doc['content'] += "\n\nAvailable Seats: No current seat availability data found"
            
            results.append(doc)
        
        return results
    
    def get_school_seats(self, school_id: str) -> List[Dict[str, Any]]:
        """Get seat availability for a specific school"""
        return self.school_seats.get(school_id, [])
    
    def search_by_seats_available(self, min_seats: int = 1, program_type: str = None) -> List[Dict[str, Any]]:
        """Search schools with available seats"""
        results = []
        
        for school_id, seats in self.school_seats.items():
            for seat_info in seats:
                if seat_info['seats_left'] >= min_seats:
                    if program_type is None or program_type.lower() in seat_info['program'].lower():
                        # Find corresponding school document
                        school_doc = None
                        for doc in self.documents:
                            if doc['id'] == school_id:
                                school_doc = doc.copy()
                                break
                        
                        if school_doc:
                            school_doc['seat_availability'] = [seat_info]
                            results.append(school_doc)
        
        return results 