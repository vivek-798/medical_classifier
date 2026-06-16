import os
import re
import json
import math

# Fast Levenshtein distance for fuzzy matching
def get_levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return get_levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

class MedicalKeywordClassifier:
    def __init__(self):
        # Spelling normalizing map for common OCR mistakes / variants
        self.ocr_typo_map = {
            "heeemoglobin": "hemoglobin",
            "heamoglobin": "haemoglobin",
            "haemetology": "haematology",
            "hemetology": "hematology",
            "b1ood": "blood",
            "comp1ete": "complete",
            "ur1ne": "urine",
            "creatin1ne": "creatinine",
            "hemglobin": "hemoglobin",
            "platelets": "platelet",
            "lymphocyte": "lymphocytes",
            "neutrophil": "neutrophils",
            "monocyte": "monocytes",
            "eosinophil": "eosinophils",
            "basophil": "basophils",
            "organisms": "organism"
        }
        
        # Load rules dynamically from compiled knowledge base JSON
        self.rules = {}
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        KNOWLEDGE_PATH = os.path.join(BASE_DIR, "src", "medical_knowledge.json")
        
        if os.path.exists(KNOWLEDGE_PATH):
            try:
                with open(KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
                    knowledge = json.load(f)
                    for cat, data in knowledge.items():
                        keywords = data.get("keywords", [])
                        
                        titles = []
                        parameters = {}
                        
                        norm_cat = cat.lower().replace('_', ' ').strip()
                        for kw in keywords:
                            kw_norm = kw.lower().strip()
                            
                            is_title = False
                            # Categorize keyword as title if it matches category name or known abbreviation
                            if kw_norm == norm_cat or kw_norm == cat.lower():
                                is_title = True
                            elif kw_norm in ["cbc", "lft", "kft", "rft", "crp", "cue", "abg", "esr", "rbs", "widal", "abo"]:
                                is_title = True
                            elif "profile" in kw_norm and "profile" in norm_cat:
                                is_title = True
                            elif "function test" in kw_norm and "function test" in norm_cat:
                                is_title = True
                                
                            if is_title:
                                titles.append(kw_norm)
                            else:
                                # Weight: multi-word parameters = 5, single-word = 3
                                weight = 5 if " " in kw_norm else 3
                                parameters[kw_norm] = weight
                        
                        self.rules[cat] = {
                            "titles": titles if titles else [norm_cat],
                            "parameters": parameters
                        }
                print(f"Loaded rules for {len(self.rules)} medical categories dynamically.")
            except Exception as e:
                print(f"Error loading knowledge JSON: {e}")
        else:
            print(f"Warning: medical_knowledge.json not found at {KNOWLEDGE_PATH}. Classifications will fail.")

    def clean_text(self, text: str) -> str:
        """
        OCR Text Cleaning Pipeline:
        - Convert to lowercase
        - Remove unnecessary symbols and punctuation (except alphanumeric and spaces)
        - Correct common OCR spelling mistakes/variants
        - Remove duplicate words (preserving order)
        - Normalize extra spaces
        """
        if not text:
            return ""
            
        # 1. Convert to lowercase
        text = text.lower()
        
        # 2. Clean symbols (retain alphanumeric, spaces, and useful chars like '+' or '/')
        text = re.sub(r'[^a-z0-9\s+/]', ' ', text)
        
        # 3. Normalize extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 4. Correct typos & normalize duplicate words while maintaining sequence order
        words = text.split()
        seen = set()
        cleaned_words = []
        for w in words:
            w_norm = self.ocr_typo_map.get(w, w)
            if w_norm not in seen:
                seen.add(w_norm)
                cleaned_words.append(w_norm)
                
        return " ".join(cleaned_words)

    def fuzzy_match(self, word: str, keyword: str) -> bool:
        """
        Determines if a word matches a keyword fuzzy-wise.
        - Exact match returns True.
        - Short words (length <= 3) must be exact.
        - Longer words permit an edit distance of 1 or 2.
        """
        if word == keyword:
            return True
            
        len_w = len(word)
        len_k = len(keyword)
        
        # Short words require exact matches
        if len_k <= 3 or len_w <= 3:
            return False
            
        max_dist = 1 if len_k <= 6 else 2
        if abs(len_w - len_k) > max_dist:
            return False
            
        dist = get_levenshtein_distance(word, keyword)
        return dist <= max_dist

    def classify(self, raw_text: str) -> dict:
        """
        Matches keywords and calculates scoring-based categories and confidence.
        Supports dynamic rule mappings.
        """
        cleaned_text = self.clean_text(raw_text)
        tokens = cleaned_text.split()
        
        scores = {}
        matched_by_category = {}
        
        # In case rules didn't load
        if not self.rules:
            return {
                "category": "Unknown",
                "confidence": 0.0,
                "matched_keywords": []
            }
            
        for category, rules in self.rules.items():
            scores[category] = 0.0
            matched_by_category[category] = set()
            
            # 1. Title matching: check exact phrases
            for title in rules["titles"]:
                if title in cleaned_text:
                    scores[category] += 10.0
                    matched_by_category[category].add(title)
            
            # 2. Parameter matching
            for param, weight in rules["parameters"].items():
                if " " in param:
                    # Multi-word parameter: check exact substring match
                    if param in cleaned_text:
                        scores[category] += weight
                        matched_by_category[category].add(param)
                else:
                    # Single-word parameter: check exact or fuzzy token match
                    matched = False
                    if param in tokens:
                        matched = True
                    else:
                        for token in tokens:
                            if self.fuzzy_match(token, param):
                                matched = True
                                break
                    if matched:
                        scores[category] += weight
                        matched_by_category[category].add(param)

        # Find top categories
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_cat, top_score = sorted_scores[0]
        
        if len(sorted_scores) >= 2:
            second_cat, second_score = sorted_scores[1]
        else:
            second_cat, second_score = "None", 0.0
        
        # Handle zero scores (completely unknown report)
        if top_score == 0:
            return {
                "category": "Unknown",
                "confidence": 0.0,
                "matched_keywords": []
            }

        # Matched keywords for output
        matched_keywords = sorted(list(matched_by_category[top_cat]))
        
        # Determine confidence score
        has_title_match = any(title in matched_keywords for title in self.rules[top_cat]["titles"])
        
        if has_title_match:
            # base confidence is 0.90, scaled by parameter support
            param_score = sum(self.rules[top_cat]["parameters"].get(kw, 0) for kw in matched_keywords)
            max_possible_param_score = sum(self.rules[top_cat]["parameters"].values())
            param_ratio = (param_score / max_possible_param_score) if max_possible_param_score > 0 else 0
            confidence = 0.90 + (0.09 * param_ratio)
        else:
            # Probability-based confidence
            if top_score + second_score > 0:
                confidence = top_score / (top_score + second_score)
                confidence = min(confidence * 0.90, 0.85)
            else:
                confidence = 0.50
                
        # Clamp confidence range
        confidence = max(0.10, min(0.99, confidence))
        
        return {
            "category": top_cat,
            "confidence": round(confidence, 2),
            "matched_keywords": matched_keywords
        }
