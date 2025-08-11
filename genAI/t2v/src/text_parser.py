"""
Forensic Text-to-Video Feature Extractor and Prompt Generator
Extracts forensic features from structured text and generates DeeVid AI prompts
"""

import json
import re
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import openai
import spacy
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class ForensicFeatures:
    """Data class to store extracted forensic features"""
    person_behavior: List[str]
    background_setting: List[str]
    actions: List[str]
    lighting: List[str]
    clothing: List[str]
    facial_expressions: List[str]
    body_language: List[str]
    timeline: List[str]
    objects_involved: List[str]
    emotions_detected: List[str]

class ForensicTextExtractor:
    """Extracts forensic features from structured text files"""
    
    def __init__(self):
        """Initialize NLP models and components"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model loaded successfully")
            
            # Load HuggingFace models
            self.emotion_classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-emotion",
                return_all_scores=False
            )
            
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            logger.info("✅ HuggingFace models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise
    
    def load_structured_text(self, file_path: str) -> Dict:
        """Load structured text data from file"""
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                # Assume plain text file with structured content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Convert to structured format
                    data = self._parse_plain_text_to_structure(content)
            
            logger.info(f"✅ Loaded text data from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error loading file {file_path}: {e}")
            return {}
    
    def _parse_plain_text_to_structure(self, text: str) -> Dict:
        """Parse plain text into structured format"""
        # Basic text parsing - you can enhance this based on your actual format
        return {
            "content": text,
            "transcribed_audio": "",
            "facial_expressions": "",
            "body_language": "",
            "visual_observations": ""
        }
    
    def extract_person_behavior(self, text: str) -> List[str]:
        """Extract person behavior patterns"""
        behaviors = []
        
        # Behavior keywords and patterns
        behavior_patterns = [
            r'\b(?:appeared|seemed|looked)\s+(\w+(?:\s+\w+)?)',  # appeared nervous
            r'\b(?:was|were)\s+(\w+ing)\b',  # was running, were hiding
            r'\b(?:acting|behaving)\s+(\w+(?:\s+\w+)?)',  # acting suspicious
            r'\bmoved\s+(\w+(?:\s+\w+)?)',  # moved quickly
            r'\b(?:walking|running|standing|sitting|lying)\s+(\w+(?:\s+\w+)?)',  # walking slowly
        ]
        
        for pattern in behavior_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            behaviors.extend([match.strip() for match in matches if match.strip()])
        
        # Use spaCy for additional extraction
        doc = self.nlp(text)
        for token in doc:
            if token.pos_ == "ADV" and token.head.pos_ in ["VERB", "ADJ"]:
                behaviors.append(f"{token.head.text} {token.text}")
        
        return list(set(behaviors))  # Remove duplicates
    
    def extract_background_setting(self, text: str) -> List[str]:
        """Extract background and setting information"""
        settings = []
        
        # Location patterns
        location_patterns = [
            r'\b(?:in|at|inside|outside)\s+(?:the\s+)?([a-zA-Z\s]+?)(?:\s+(?:room|area|building|house|apartment))',
            r'\b(?:kitchen|bedroom|living room|bathroom|garage|basement|attic|office|hallway)\b',
            r'\b(?:restaurant|store|park|street|alley|parking lot|mall|school)\b',
            r'\benvironment\s*:\s*([^\n]+)',  # environment: description
            r'\bsetting\s*:\s*([^\n]+)',     # setting: description
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            settings.extend([match.strip() for match in matches if match.strip()])
        
        # Extract named entities (locations)
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC", "FAC"]:  # Geopolitical entity, location, facility
                settings.append(ent.text)
        
        return list(set(settings))
    
    def extract_actions(self, text: str) -> List[str]:
        """Extract specific actions performed"""
        actions = []
        
        # Action patterns
        action_patterns = [
            r'\b(?:he|she|they|suspect|person)\s+(\w+ed)\s+',  # past tense verbs
            r'\b(?:was|were)\s+(\w+ing)\s+',  # continuous verbs
            r'\b(?:grabbed|took|opened|closed|entered|exited|searched|looked|moved|ran|walked)\b',
            r'\baction\s*:\s*([^\n]+)',  # action: description
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            actions.extend([match.strip() for match in matches if match.strip()])
        
        # Use spaCy for verb extraction
        doc = self.nlp(text)
        for token in doc:
            if token.pos_ == "VERB" and not token.is_stop:
                # Get the full verb phrase
                verb_phrase = [token.text]
                for child in token.children:
                    if child.dep_ in ["dobj", "prep", "prt"]:
                        verb_phrase.append(child.text)
                actions.append(" ".join(verb_phrase))
        
        return list(set(actions))
    
    def extract_lighting(self, text: str) -> List[str]:
        """Extract lighting conditions"""
        lighting = []
        
        lighting_patterns = [
            r'\b(?:bright|dark|dim|well-lit|poorly lit|shadowy|sunlit|moonlit)\b',
            r'\b(?:daylight|sunlight|artificial light|fluorescent|lamplight)\b',
            r'\blighting\s*:\s*([^\n]+)',
            r'\b(?:lights?\s+(?:on|off|bright|dim))\b',
            r'\b(?:it was|room was)\s+([a-zA-Z\s]+?)(?:\s+lit|lighted)',
        ]
        
        for pattern in lighting_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            lighting.extend([match.strip() for match in matches if match.strip()])
        
        return list(set(lighting))
    
    def extract_clothing(self, text: str) -> List[str]:
        """Extract clothing descriptions"""
        clothing = []
        
        clothing_patterns = [
            r'\b(?:wearing|dressed in|had on)\s+([^\n.]+)',
            r'\b(?:shirt|pants|jeans|dress|jacket|hoodie|coat|shoes|hat|cap)\b',
            r'\bclothing\s*:\s*([^\n]+)',
            r'\b(?:blue|red|black|white|green|yellow|brown|gray|grey)\s+(?:shirt|pants|jeans|dress|jacket|hoodie)',
            r'\b(?:dark|light|bright)\s+(?:clothing|clothes|attire)',
        ]
        
        for pattern in clothing_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            clothing.extend([match.strip() for match in matches if match.strip()])
        
        return list(set(clothing))
    
    def extract_facial_expressions(self, data: Dict) -> List[str]:
        """Extract facial expressions from structured data"""
        expressions = []
        
        # Check if facial expressions are explicitly provided
        if "facial_expressions" in data and data["facial_expressions"]:
            expressions.extend(data["facial_expressions"].split(','))
        
        # Pattern matching in text
        text = str(data.get("content", ""))
        expression_patterns = [
            r'\bfacial[_\s]expressions?\s*:\s*([^\n]+)',
            r'\b(?:looked|appeared|seemed)\s+(angry|sad|happy|surprised|confused|worried|scared|neutral)',
            r'\b(?:smiled|frowned|grimaced|squinted|stared|glared)\b',
        ]
        
        for pattern in expression_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            expressions.extend([match.strip() for match in matches if match.strip()])
        
        return list(set([expr.strip() for expr in expressions if expr.strip()]))
    
    def extract_body_language(self, data: Dict) -> List[str]:
        """Extract body language from structured data"""
        body_lang = []
        
        # Check if body language is explicitly provided
        if "body_language" in data and data["body_language"]:
            body_lang.extend(data["body_language"].split(','))
        
        # Pattern matching in text
        text = str(data.get("content", ""))
        body_patterns = [
            r'\bbody[_\s]language\s*:\s*([^\n]+)',
            r'\b(?:posture|stance|gestures?)\s*:\s*([^\n]+)',
            r'\b(?:slouched|upright|tense|relaxed|fidgeting|still)\b',
            r'\b(?:crossed arms|hands on hips|pointed|gestured|shrugged)\b',
        ]
        
        for pattern in body_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            body_lang.extend([match.strip() for match in matches if match.strip()])
        
        return list(set([bl.strip() for bl in body_lang if bl.strip()]))
    
    def extract_timeline(self, text: str) -> List[str]:
        """Extract timeline and sequence information"""
        timeline = []
        
        time_patterns = [
            r'\b(?:at|around|approximately)\s+(\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?)',
            r'\b(?:first|then|next|after|finally|meanwhile)\s+([^.]+)',
            r'\b(?:morning|afternoon|evening|night|dawn|dusk)\b',
            r'\b(?:before|after|during)\s+([^,.]+)',
            r'\btimeline\s*:\s*([^\n]+)',
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            timeline.extend([match.strip() for match in matches if match.strip()])
        
        return list(set(timeline))
    
    def extract_objects_involved(self, text: str) -> List[str]:
        """Extract objects and items involved in the scene"""
        objects = []
        
        # Use spaCy for noun extraction
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "WORK_OF_ART", "ORG"]:
                objects.append(ent.text)
        
        # Pattern-based extraction
        object_patterns = [
            r'\b(?:grabbed|took|touched|moved|broke|opened|closed)\s+(?:the\s+)?([a-zA-Z\s]+?)(?:\s|$|\.)',
            r'\b(?:weapon|tool|item|object)\s*:\s*([^\n]+)',
            r'\b(?:knife|gun|phone|wallet|bag|keys|documents|jewelry|laptop|camera)\b',
        ]
        
        for pattern in object_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            objects.extend([match.strip() for match in matches if match.strip()])
        
        return list(set(objects))
    
    def extract_emotions(self, text: str) -> List[str]:
        """Extract emotions using AI models"""
        emotions = []
        
        try:
            # Use emotion classifier
            emotion_result = self.emotion_classifier(text)
            if isinstance(emotion_result, list) and len(emotion_result) > 0:
                emotions.append(emotion_result[0]['label'])
            
            # Use sentiment analyzer
            sentiment_result = self.sentiment_analyzer(text)
            if isinstance(sentiment_result, list) and len(sentiment_result) > 0:
                emotions.append(sentiment_result[0]['label'])
                
        except Exception as e:
            logger.warning(f"⚠️ Emotion extraction failed: {e}")
        
        return emotions
    
    def extract_all_features(self, file_path: str) -> ForensicFeatures:
        """Extract all forensic features from the text file"""
        logger.info(f"🔍 Starting feature extraction from {file_path}")
        
        # Load the structured data
        data = self.load_structured_text(file_path)
        text_content = str(data.get("content", ""))
        
        # Extract all features
        features = ForensicFeatures(
            person_behavior=self.extract_person_behavior(text_content),
            background_setting=self.extract_background_setting(text_content),
            actions=self.extract_actions(text_content),
            lighting=self.extract_lighting(text_content),
            clothing=self.extract_clothing(text_content),
            facial_expressions=self.extract_facial_expressions(data),
            body_language=self.extract_body_language(data),
            timeline=self.extract_timeline(text_content),
            objects_involved=self.extract_objects_involved(text_content),
            emotions_detected=self.extract_emotions(text_content)
        )
        
        logger.info("✅ Feature extraction completed")
        return features

class DeeVidPromptGenerator:
    """Generates DeeVid AI prompts using OpenAI"""
    
    def __init__(self):
        """Initialize OpenAI client"""
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("✅ OpenAI client initialized")
    
    def generate_known_scenario_prompt(self, features: ForensicFeatures) -> str:
        """Generate prompt for known/current information scenario"""
        
        system_prompt = """You are a forensic video reconstruction specialist. Create a detailed DeeVid AI text-to-video prompt that recreates the crime scene exactly as described in the evidence. 

Requirements:
- Keep under 2000 characters
- Be specific about visual elements
- Include camera angles and movements
- Focus on factual recreation
- Use cinematic language suitable for video generation

Format: Create a single, detailed prompt that DeeVid AI can use to generate a forensic reconstruction video."""
        
        user_content = f"""
Based on this forensic evidence, create a DeeVid text-to-video prompt for EXACT recreation:

Person Behavior: {', '.join(features.person_behavior)}
Actions: {', '.join(features.actions)}
Background/Setting: {', '.join(features.background_setting)}
Lighting: {', '.join(features.lighting)}
Clothing: {', '.join(features.clothing)}
Facial Expressions: {', '.join(features.facial_expressions)}
Body Language: {', '.join(features.body_language)}
Timeline: {', '.join(features.timeline)}
Objects Involved: {', '.join(features.objects_involved)}
Emotions: {', '.join(features.emotions_detected)}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using cheaper model as requested
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=800,
                temperature=0.3  # Lower temperature for factual recreation
            )
            
            prompt = response.choices[0].message.content.strip()
            logger.info("✅ Known scenario prompt generated")
            return prompt[:1900]  # Ensure under 2000 characters
            
        except Exception as e:
            logger.error(f"❌ Error generating known scenario prompt: {e}")
            return ""
    
    def generate_alternative_scenario_prompt(self, features: ForensicFeatures) -> str:
        """Generate prompt for alternative/new possibilities scenario"""
        
        system_prompt = """You are a forensic analyst exploring alternative theories. Create a DeeVid AI text-to-video prompt showing a plausible alternative scenario based on the same evidence, but with different interpretations or missing details filled in creatively yet logically.

Requirements:
- Keep under 2000 characters
- Stay plausible based on evidence
- Explore "what if" scenarios
- Include different motivations or sequences
- Use cinematic language suitable for video generation

Format: Create a single, detailed prompt that DeeVid AI can use to generate an alternative forensic theory video."""
        
        user_content = f"""
Based on this forensic evidence, create a DeeVid text-to-video prompt for an ALTERNATIVE scenario:

Person Behavior: {', '.join(features.person_behavior)}
Actions: {', '.join(features.actions)}
Background/Setting: {', '.join(features.background_setting)}
Lighting: {', '.join(features.lighting)}
Clothing: {', '.join(features.clothing)}
Facial Expressions: {', '.join(features.facial_expressions)}
Body Language: {', '.join(features.body_language)}
Timeline: {', '.join(features.timeline)}
Objects Involved: {', '.join(features.objects_involved)}
Emotions: {', '.join(features.emotions_detected)}

Consider: What if the sequence was different? What if there was a different motive? What if some details were misinterpreted?
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using cheaper model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=800,
                temperature=0.7  # Higher temperature for creative alternatives
            )
            
            prompt = response.choices[0].message.content.strip()
            logger.info("✅ Alternative scenario prompt generated")
            return prompt[:1900]  # Ensure under 2000 characters
            
        except Exception as e:
            logger.error(f"❌ Error generating alternative scenario prompt: {e}")
            return ""

class ForensicT2VPipeline:
    """Main pipeline class that orchestrates the entire process"""
    
    def __init__(self):
        """Initialize the complete pipeline"""
        self.extractor = ForensicTextExtractor()
        self.prompt_generator = DeeVidPromptGenerator()
        logger.info("🚀 Forensic T2V Pipeline initialized")
    
    def process_forensic_text(self, file_path: str) -> Tuple[str, str]:
        """
        Process forensic text file and generate both DeeVid prompts
        
        Returns:
            Tuple[str, str]: (known_scenario_prompt, alternative_scenario_prompt)
        """
        logger.info(f"🔥 Processing forensic text: {file_path}")
        
        # Extract features
        features = self.extractor.extract_all_features(file_path)
        
        # Log extracted features for debugging
        logger.info("📊 Extracted Features Summary:")
        for field, value in asdict(features).items():
            if value:
                logger.info(f"  {field}: {len(value)} items")
        
        # Generate both prompts
        known_prompt = self.prompt_generator.generate_known_scenario_prompt(features)
        alternative_prompt = self.prompt_generator.generate_alternative_scenario_prompt(features)
        
        return known_prompt, alternative_prompt
    
    def save_prompts(self, known_prompt: str, alternative_prompt: str, output_dir: str = "output"):
        """Save generated prompts to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/known_scenario_prompt.txt", 'w', encoding='utf-8') as f:
            f.write(known_prompt)
        
        with open(f"{output_dir}/alternative_scenario_prompt.txt", 'w', encoding='utf-8') as f:
            f.write(alternative_prompt)
        
        logger.info(f"💾 Prompts saved to {output_dir}/")

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    pipeline = ForensicT2VPipeline()
    
    # Create sample structured text file for testing
    sample_data = {
        "content": "The suspect was seen entering through the back door around 10 PM. He was wearing dark clothing and appeared nervous. The kitchen light was on but the rest of the house was dark. He searched through drawers frantically before leaving through the same door.",
        "facial_expressions": "nervous, worried, focused",
        "body_language": "tense posture, quick movements, looking over shoulder",
        "visual_observations": "dark clothing, average height, male"
    }
    
    # Save sample data
    with open("sample_forensic_data.json", 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    # Process the sample file
    known_prompt, alternative_prompt = pipeline.process_forensic_text("sample_forensic_data.json")
    
    print("\n" + "="*60)
    print("🎬 KNOWN SCENARIO PROMPT (DeeVid AI)")
    print("="*60)
    print(known_prompt)
    print(f"\nCharacter count: {len(known_prompt)}")
    
    print("\n" + "="*60)
    print("🔮 ALTERNATIVE SCENARIO PROMPT (DeeVid AI)")
    print("="*60)
    print(alternative_prompt)
    print(f"\nCharacter count: {len(alternative_prompt)}")
    
    # Save prompts
    pipeline.save_prompts(known_prompt, alternative_prompt)
    print("\n✅ Prompts saved to output/ directory")