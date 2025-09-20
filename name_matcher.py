import numpy as np
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
import jellyfish
import re

class NameMatcher:
    def __init__(self):
        # Dataset of 30+ similar names for matching
        self.names_database = [
            "Geetha", "Gita", "Gitu", "Geeta", "Gitika", "Geetika", "Gitanjali",
            "Priya", "Priyanka", "Preya", "Priyanka", "Priyita", "Priyasha",
            "Rajesh", "Raj", "Raja", "Raju", "Rajat", "Rajiv", "Rajeev",
            "Amit", "Amith", "Ameet", "Amrita", "Amrith", "Amitabh",
            "Suresh", "Sures", "Suraj", "Surya", "Suresha", "Suri",
            "Anita", "Anitha", "Aneetah", "Anitta", "Anita", "Anitya",
            "Vikram", "Vikramjeet", "Vikash", "Vikas", "Vikrant", "Vikram",
            "Kavya", "Kavita", "Kavitha", "Kavyanka", "Kaveri", "Kavi",
            "Deepak", "Deep", "Deepika", "Dipak", "Dipika", "Deepa",
            "Rahul", "Raul", "Rohul", "Rahool", "Raahul", "Rahil"
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        self.names_database = [x for x in self.names_database if not (x in seen or seen.add(x))]
    
    def calculate_similarity_score(self, name1: str, name2: str) -> float:
        """
        Calculate similarity score using multiple algorithms and return weighted average
        """
        name1_clean = self._clean_name(name1)
        name2_clean = self._clean_name(name2)
        
        # Multiple similarity metrics
        scores = []
        
        # 1. Jaro-Winkler similarity (good for names)
        jaro_score = jellyfish.jaro_winkler_similarity(name1_clean, name2_clean)
        scores.append(jaro_score * 0.3)
        
        # 2. Fuzzy ratio
        fuzzy_score = fuzz.ratio(name1_clean, name2_clean) / 100.0
        scores.append(fuzzy_score * 0.25)
        
        # 3. Fuzzy partial ratio
        partial_score = fuzz.partial_ratio(name1_clean, name2_clean) / 100.0
        scores.append(partial_score * 0.2)
        
        # 4. Sequence matcher
        sequence_score = SequenceMatcher(None, name1_clean, name2_clean).ratio()
        scores.append(sequence_score * 0.15)
        
        # 5. Soundex similarity (phonetic matching)
        soundex_score = 1.0 if jellyfish.soundex(name1_clean) == jellyfish.soundex(name2_clean) else 0.0
        scores.append(soundex_score * 0.1)
        
        return sum(scores)
    
    def _clean_name(self, name: str) -> str:
        """Clean and normalize name for better matching"""
        return re.sub(r'[^a-zA-Z]', '', name.lower().strip())
    
    def find_similar_names(self, input_name: str, top_n: int = 10) -> Dict:
        """
        Find most similar names from database with similarity scores
        """
        if not input_name or not input_name.strip():
            raise ValueError("Input name cannot be empty")
        
        similarities = []
        
        for db_name in self.names_database:
            score = self.calculate_similarity_score(input_name, db_name)
            similarities.append({
                "name": db_name,
                "similarity_score": round(score, 4)
            })
        
        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Get top N matches
        top_matches = similarities[:top_n]
        
        # Prepare response
        result = {
            "input_name": input_name,
            "best_match": {
                top_matches[0]["name"]: top_matches[0]["similarity_score"]
            } if top_matches else {},
            "all_matches": [
                {match["name"]: match["similarity_score"]} 
                for match in top_matches
            ]
        }
        
        return result
    
    def add_name_to_database(self, name: str) -> bool:
        """Add a new name to the database"""
        clean_name = name.strip()
        if clean_name and clean_name not in self.names_database:
            self.names_database.append(clean_name)
            return True
        return False
    
    def get_database_size(self) -> int:
        """Get current database size"""
        return len(self.names_database) 