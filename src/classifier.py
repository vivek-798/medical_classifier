import re

class ReportClassifier:
    def __init__(self):
        # Configuration of categories, titles, and parameters.
        # Categorization uses lowercase normalized strings for comparison.
        self.rules = {
            "CBC": {
                "titles": [
                    "complete blood count", "hemogram", "haemogram", "cbc", 
                    "full blood count", "blood count", "complete blood picture",
                    "automated haematology report", "haematology report"
                ],
                "parameters": [
                    "hemoglobin", "haemoglobin", "rbc", "wbc", "platelet", "pcv", "mcv", "mch", "mchc",
                    "red blood cell", "white blood cell", "granulocytes", "lymphocytes", "monocytes",
                    "eosinophils", "basophils", "hematocrit", "neutrophils", "leucocytes", "packed cell volume",
                    "mean corpuscular volume", "mean corpuscular hemoglobin"
                ]
            },
            "CRP": {
                "titles": [
                    "c-reactive protein", "crp", "c reactive protein", "hs-crp", "hscrp"
                ],
                "parameters": [
                    "c-reactive protein", "crp", "c reactive protein", "hs-crp"
                ]
            },
            "LFT": {
                "titles": [
                    "liver function test", "lft", "liver profile", "hepatic profile", 
                    "liver function", "liver profile (lft)"
                ],
                "parameters": [
                    "alt", "ast", "bilirubin", "albumin", "sgot", "sgpt", "alkaline phosphatase",
                    "alp", "globulin", "total protein", "direct bilirubin", "indirect bilirubin",
                    "a/g ratio", "alanine aminotransferase", "aspartate aminotransferase", "gamma glutamyl"
                ]
            },
            "KFT": {
                "titles": [
                    "kidney function test", "kft", "kidney profile", "renal function test", "rft", 
                    "renal profile", "renal function", "kidney profile (kft)"
                ],
                "parameters": [
                    "creatinine", "urea", "bun", "egfr", "blood urea nitrogen", "kidney function",
                    "serum creatinine", "blood urea", "estimated glomerular filtration rate"
                ]
            },
            "RBS": {
                "titles": [
                    "random blood sugar", "rbs", "glucose", "fasting blood sugar", "fbs", 
                    "post prandial blood sugar", "ppbs", "blood sugar", "oral glucose tolerance",
                    "glucose profile", "blood glucose"
                ],
                "parameters": [
                    "glucose", "random blood sugar", "fasting blood sugar", "post prandial glucose",
                    "fasting blood glucose", "post prandial blood glucose"
                ]
            },
            "Haematology": {
                "titles": [
                    "haematology", "hematology", "esr", "erythrocyte sedimentation rate", "esr report"
                ],
                "parameters": [
                    "esr", "erythrocyte sedimentation rate", "blood counts", "hematology parameters",
                    "wintrobe"
                ]
            },
            "Coagulation": {
                "titles": [
                    "coagulation", "pt", "inr", "aptt", "prothrombin time", "activated partial thromboplastin"
                ],
                "parameters": [
                    "pt", "inr", "aptt", "prothrombin time", "control sec", "test sec", "clotting time", "bleeding time"
                ]
            },
            "Electrolytes": {
                "titles": [
                    "electrolytes", "serum electrolytes", "electrolyte profile"
                ],
                "parameters": [
                    "sodium", "potassium", "chloride", "na+", "k+", "cl-", "serum sodium", "serum potassium", "serum chloride"
                ]
            },
            "Serology": {
                "titles": [
                    "serology", "serological report", "serology report", "immunology & serology"
                ],
                "parameters": [
                    "dengue", "hiv", "hbsag", "hcv", "vdrl", "syphilis", "ra factor", 
                    "rheumatoid factor", "aso titre", "serology test"
                ]
            },
            "Microbiology": {
                "titles": [
                    "microbiology", "culture report", "culture & sensitivity", "bacteriology", 
                    "sensitivity report", "culture and sensitivity", "urine culture"
                ],
                "parameters": [
                    "culture", "sensitivity", "gram stain", "organism", "growth", "colony count",
                    "microbial", "microbiology report", "aerobic culture"
                ]
            },
            "Widal Test": {
                "titles": [
                    "widal test", "widal", "typhoid antibody", "widal slide test", "widal report"
                ],
                "parameters": [
                    "salmonella typhi", "s. typhi", "to", "th", "ao", "ah", "bo", "bh", "widal test"
                ]
            },
            "Thyroid Profile": {
                "titles": [
                    "thyroid profile", "thyroid function test", "tft", "thyroid panel", "thyroid report"
                ],
                "parameters": [
                    "t3", "t4", "tsh", "triiodothyronine", "thyroxine", "thyroid stimulating hormone",
                    "free t3", "free t4"
                ]
            },
            "Lipid Profile": {
                "titles": [
                    "lipid profile", "lipid", "cholesterol profile", "lipid panel", "lipid report"
                ],
                "parameters": [
                    "cholesterol", "triglycerides", "hdl", "ldl", "vldl", "total cholesterol",
                    "hdl cholesterol", "ldl cholesterol", "vldl cholesterol", "triglyceride"
                ]
            },
            "Urine Analysis": {
                "titles": [
                    "urine analysis", "urine routine", "urine examination", "routine urine analysis", 
                    "urine chemical", "urine report"
                ],
                "parameters": [
                    "specific gravity", "ph", "leukocytes", "nitrite", "urobilinogen", "protein", 
                    "ketone", "bilirubin", "glucose"
                ]
            },
            "Stool Examination": {
                "titles": [
                    "stool examination", "stool routine", "stool analysis", "stool report"
                ],
                "parameters": [
                    "stool occult blood", "ova", "cysts", "stool colour", "occult blood"
                ]
            },
            "Vitamin D": {
                "titles": [
                    "vitamin d", "25-hydroxy vitamin d", "vitamin d3", "25 hydroxy vitamin d"
                ],
                "parameters": [
                    "25-hydroxy vitamin d", "vitamin d", "25-oh vitamin d", "vitamin d3"
                ]
            },
            "Vitamin B12": {
                "titles": [
                    "vitamin b12", "cyanocobalamin", "vit b12", "serum vitamin b12"
                ],
                "parameters": [
                    "vitamin b12", "b12", "cyanocobalamin", "cobalamin"
                ]
            },
            "HbA1c": {
                "titles": [
                    "hba1c", "glycated hemoglobin", "glycosylated hemoglobin"
                ],
                "parameters": [
                    "hba1c", "glycated haemoglobin", "estimated average glucose", "eag", "glycosylated hemoglobin"
                ]
            },
            "Blood Grouping & Rh": {
                "titles": [
                    "blood grouping", "blood group", "rh grouping", "blood grouping & rh", "blood grouping and rh"
                ],
                "parameters": [
                    "blood group", "rh factor", "rh type", "a positive", "o positive", "b positive", 
                    "blood grouping & rh typing"
                ]
            },
            "ABG": {
                "titles": [
                    "abg", "arterial blood gas", "blood gas analysis", "blood gas"
                ],
                "parameters": [
                    "pco2", "po2", "hco3", "ph (blood)", "base excess", "so2", "arterial blood gas"
                ]
            },
            "CUE": {
                "titles": [
                    "complete urine examination", "cue", "urine cue"
                ],
                "parameters": [
                    "pus cells", "epithelial cells", "casts", "crystals", "urine routine & microscopy"
                ]
            },
            "EPE": {
                "titles": [
                    "epe", "examination of peripheral smear", "peripheral smear", "peripheral blood smear"
                ],
                "parameters": [
                    "rbc morphology", "wbc morphology", "platelets morphology", "peripheral smear", "smear study"
                ]
            },
            "Serum Uric Acid": {
                "titles": [
                    "serum uric acid", "uric acid report"
                ],
                "parameters": [
                    "uric acid", "serum uric acid"
                ]
            },
            "Immunology Reports": {
                "titles": [
                    "immunology report", "immunology", "immunoglobulin profile"
                ],
                "parameters": [
                    "immunoglobulins", "igg", "igm", "iga", "ige", "immunology"
                ]
            },
            "Pathology Reports": {
                "titles": [
                    "pathology report", "histopathology", "biopsy", "surgical pathology"
                ],
                "parameters": [
                    "gross examination", "microscopic examination", "impression", "clinical diagnosis"
                ]
            }
        }

    def _clean_text(self, text):
        # Normalize: convert to lowercase and strip excess spaces
        return text.lower().strip()

    def _is_keyword_in_text(self, keyword, text_lower):
        # Escape the keyword for regex safety
        escaped = re.escape(keyword)
        # Use custom word boundaries that work with special chars like +, -
        if len(keyword) <= 3:
            pattern = rf"(?<![a-zA-Z0-9]){escaped}(?![a-zA-Z0-9])"
        else:
            pattern = escaped
        return bool(re.search(pattern, text_lower))

    def classify(self, text_lines):
        """
        Classifies the report text lines and returns (category, confidence).
        """
        if not text_lines:
            return "Unknown", 0.0

        # Create a joined full text representation
        full_text = " ".join([self._clean_text(line) for line in text_lines])

        # We inspect the first 15 lines as candidates for the title
        title_candidates = [self._clean_text(line) for line in text_lines[:15]]

        scores = {}
        title_matched_categories = []

        for category, rules in self.rules.items():
            title_score = 0.0
            param_score = 0.0

            # 1. Title matching: search for category title keywords in first 15 lines
            for title in rules["titles"]:
                for line in title_candidates:
                    if self._is_keyword_in_text(title, line):
                        title_score += 15.0 # High score for title match
                        title_matched_categories.append(category)
                        break # Count title match once per category

            # 2. Parameter matching: search for parameters in full text
            for param in rules["parameters"]:
                if self._is_keyword_in_text(param, full_text):
                    param_score += 1.0

            # Combined score
            scores[category] = title_score + param_score

        # Sort the scores in descending order
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_category, top_score = sorted_scores[0]
        second_category, second_score = sorted_scores[1]

        # Minimum score threshold to classify
        if top_score < 1.0:
            return "Unknown", 0.0

        # Confidence calculation
        # If there is a title match for the top category, set high confidence
        if top_category in title_matched_categories:
            confidence = 0.95
        else:
            # Infer confidence based on top_score vs second_score
            if top_score + second_score > 0:
                confidence = top_score / (top_score + second_score)
                # Scale it down slightly since it's parameter-inferred only
                confidence = round(min(confidence * 0.90, 0.85), 2)
            else:
                confidence = 0.50

        # If it's a complete tie between two categories with the same score and no title matches, 
        # classify as Unknown or return top one with low confidence
        if top_score == second_score and top_score > 0 and top_category not in title_matched_categories:
            # Tie breaker: let's select the one with more specific parameters if possible, 
            # otherwise mark confidence low
            confidence = 0.50

        return top_category, confidence
