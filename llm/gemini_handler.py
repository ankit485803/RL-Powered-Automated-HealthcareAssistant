

"""
Gemini LLM Handler for Response Generation
Responsible: Shubham Kumar
"""

import os
import google.generativeai as genai
from typing import Dict, Any


class GeminiHandler:
    """
    Handles Gemini API calls for healthcare response generation
    
    Methods:
        generate_response(query, context) -> str
        evaluate_safety(response) -> bool
    """
    
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Medical system prompt for safe responses"""
        return """
        You are a medical triage assistant. Your role is to provide helpful, 
        accurate, and safe health information. Never diagnose. Always recommend 
        consulting a doctor for serious symptoms. Do not provide treatment plans.
        
        Emergency symptoms requiring immediate escalation:
        - Chest pain, difficulty breathing, severe bleeding, loss of consciousness, 
          sudden severe headache, seizure, inability to speak or move limbs.
        """
    
    def generate_response(self, query: str, context: str = "") -> str:
        """
        Generate response for patient query
        
        Args:
            query: Patient's question/symptoms
            context: Patient history or additional info
        
        Returns:
            Generated response text
        """
        # TODO: Shubham - Implement Gemini API call
        full_prompt = f"{self.system_prompt}\n\nPatient: {query}\nContext: {context}\nAssistant:"
        
        # Placeholder response
        return "Based on the symptoms described, please consult a healthcare provider for proper evaluation."
    
    def evaluate_safety(self, response: str) -> bool:
        """Check if response is safe"""
        # TODO: Shubham - Implement safety filtering
        unsafe_keywords = ["die", "death", "emergency treatment", "surgery"]
        return not any(word in response.lower() for word in unsafe_keywords)