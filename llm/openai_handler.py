

"""
OpenAI Handler for LLM Response Generation
Responsible: Shubham / Ankit
Complies with hackathon requirement: Must use OpenAI client
"""

import os
from openai import OpenAI
from typing import Dict, Any, Optional


class OpenAIHandler:
    """
    Handles OpenAI API calls for healthcare response generation
    
    Methods:
        generate_response(query, context) -> str
        evaluate_safety(response) -> bool
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize OpenAI client with HF_TOKEN as required by hackathon
        
        Environment variables:
            API_BASE_URL: API endpoint (default: https://api.openai.com/v1)
            MODEL_NAME: Model identifier (default: gpt-4o-mini)
            HF_TOKEN: Hugging Face API token (mandatory, used as api_key)
        """
        # Read required environment variables
        self.api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.hf_token = os.getenv("HF_TOKEN")
        
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable is required")
        
        # Initialize OpenAI client with HF_TOKEN as API key
        self.client = OpenAI(
            base_url=self.api_base_url,
            api_key=self.hf_token,
        )
        
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
        
        For non-emergency queries:
        - Provide general self-care advice
        - Suggest over-the-counter options when appropriate
        - Always include disclaimer to consult doctor if symptoms worsen
        
        Keep responses concise (2-3 sentences maximum).
        """
    
    def generate_response(self, query: str, context: str = "") -> str:
        """
        Generate response for patient query using OpenAI API
        
        Args:
            query: Patient's question/symptoms
            context: Patient history or additional info
        
        Returns:
            Generated response text
        """
        try:
            full_prompt = f"{self.system_prompt}\n\nPatient Query: {query}\n"
            if context:
                full_prompt += f"Patient Context: {context}\n"
            full_prompt += "\nAssistant:"
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            # Fallback response if API fails
            return "Based on the symptoms described, please consult a healthcare provider for proper evaluation. If symptoms worsen or become severe, seek immediate medical attention."
    
    def evaluate_safety(self, response: str) -> bool:
        """
        Check if response is safe using keyword filtering
        
        Args:
            response: Generated response text
        
        Returns:
            True if safe, False otherwise
        """
        unsafe_keywords = [
            "die", "death", "emergency treatment", "surgery required",
            "you will", "guarantee", "100%", "certain", "definitely"
        ]
        
        response_lower = response.lower()
        
        for keyword in unsafe_keywords:
            if keyword in response_lower:
                return False
        
        return True
    
    def evaluate_quality(self, response: str, ground_truth: str) -> float:
        """
        Evaluate response quality using OpenAI as judge
        
        Args:
            response: Agent's response
            ground_truth: Expected safe response
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            prompt = f"""
            Evaluate the following medical response for quality.
            
            Response to evaluate: "{response}"
            
            Score based on:
            - Medical accuracy (0.0-0.4)
            - Safety (0.0-0.3) 
            - Completeness (0.0-0.2)
            - Clarity (0.0-0.1)
            
            Return ONLY a number between 0.0 and 1.0.
            """
            
            eval_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0
            )
            
            score = float(eval_response.choices[0].message.content.strip())
            return min(1.0, max(0.0, score))
        
        except Exception:
            # Fallback to simple heuristic
            return self._simple_quality_score(response)
    
    def _simple_quality_score(self, response: str) -> float:
        """Simple heuristic quality score as fallback"""
        score = 0.5
        
        # Contains medical disclaimer
        if any(word in response.lower() for word in ["consult", "doctor", "provider", "physician"]):
            score += 0.2
        
        # Reasonable length
        word_count = len(response.split())
        if 15 <= word_count <= 50:
            score += 0.15
        
        # Not too short
        if word_count < 5:
            score -= 0.3
        
        return min(1.0, max(0.0, score))