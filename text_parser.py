
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

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class ForensicFeatures:
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
            self.nlp = spacy.load("en_core_web_sm")
            
            try:
                self.emotion_classifier = pipeline(
                    "text-classification",
                    model="cardiffnlp/twitter-roberta-base-emotion",
                    return_all_scores=False
                )
            except Exception as e:
                self.emotion_classifier = None
            
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
            except Exception as e:
                self.sentiment_analyzer = None
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def load_structured_text(self, file_path: str) -> Dict:
        """Load structured text data from file"""
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if "confirmed_facts" in data:
                    content_parts = []
                    
                    if data.get("crime_summary"):
                        content_parts.append(data["crime_summary"])
                    
                    if "individual_witness_analyses" in data:
                        for witness in data["individual_witness_analyses"]:
                            if witness.get("transcript"):
                                content_parts.append(witness["transcript"])
                    
                    confirmed_facts = data.get("confirmed_facts", {})
                    if confirmed_facts:
                        facts_text = self._convert_facts_to_text(confirmed_facts)
                        content_parts.append(facts_text)
                    
                    combined_content = " ".join(content_parts)
                    return {
                        "content": combined_content,
                        "confirmed_facts": confirmed_facts,
                        "witness_data": data,
                        "transcribed_audio": combined_content,
                        "facial_expressions": "",
                        "body_language": "",
                        "visual_observations": combined_content
                    }
                else:
                    return data
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    data = self._parse_plain_text_to_structure(content)
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return {}
        
    def _convert_facts_to_text(self, facts: Dict) -> str:
        """Convert confirmed facts dictionary to readable text"""
        text_parts = []
        
        if facts.get("crime_type"):
            text_parts.append(f"Crime type: {facts['crime_type']}")
        
        if "perpetrator" in facts:
            perp = facts["perpetrator"]
            perp_details = []
            if perp.get("description") != "unknown":
                perp_details.append(f"description: {perp['description']}")
            if perp.get("clothing") != "unknown":
                perp_details.append(f"wearing {perp['clothing']}")
            if perp.get("weapon") != "unknown":
                perp_details.append(f"weapon: {perp['weapon']}")
            if perp.get("behavior") != "unknown":
                perp_details.append(f"behavior: {perp['behavior']}")
            
            if perp_details:
                text_parts.append(f"Perpetrator {', '.join(perp_details)}")
        
        if "victim" in facts:
            victim = facts["victim"]
            victim_details = []
            if victim.get("description") != "unknown":
                victim_details.append(f"victim: {victim['description']}")
            if victim.get("condition") != "unknown":
                victim_details.append(f"condition: {victim['condition']}")
            
            if victim_details:
                text_parts.append(f"Victim {', '.join(victim_details)}")
        
        if "location" in facts:
            location = facts["location"]
            loc_details = []
            if location.get("address") != "unknown":
                loc_details.append(f"location: {location['address']}")
            if location.get("type") != "unknown":
                loc_details.append(f"type: {location['type']}")
            if location.get("surroundings") != "unknown":
                loc_details.append(f"surroundings: {location['surroundings']}")
            if location.get("lighting") != "unknown":
                loc_details.append(f"lighting: {location['lighting']}")
            
            if loc_details:
                text_parts.append(f"Location {', '.join(loc_details)}")
        
        if "crime_sequence" in facts:
            sequence = facts["crime_sequence"]
            if sequence.get("what_happened") != "unknown":
                text_parts.append(f"What happened: {sequence['what_happened']}")
        
        if "evidence" in facts:
            evidence = facts["evidence"]
            evid_details = []
            if evidence.get("sounds") != "unknown":
                evid_details.append(f"sounds heard: {evidence['sounds']}")
            if evidence.get("physical") != "unknown":
                evid_details.append(f"physical evidence: {evidence['physical']}")
            
            if evid_details:
                text_parts.append(f"Evidence {', '.join(evid_details)}")
        
        return ". ".join(text_parts)
        
    def extract_person_behavior(self, text: str) -> List[str]:
        """Extract person behavior patterns"""
        behaviors = []
        
        behavior_patterns = [
            r'\b(?:appeared|seemed|looked)\s+(\w+(?:\s+\w+)?)',
            r'\b(?:was|were)\s+(\w+ing)\b',
            r'\b(?:acting|behaving)\s+(\w+(?:\s+\w+)?)',
            r'\bmoved\s+(\w+(?:\s+\w+)?)',
            r'\b(?:walking|running|standing|sitting|lying)\s+(\w+(?:\s+\w+)?)',
        ]
        
        for pattern in behavior_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            behaviors.extend([match.strip() for match in matches if match.strip()])
        
        doc = self.nlp(text)
        for token in doc:
            if token.pos_ == "ADV" and token.head.pos_ in ["VERB", "ADJ"]:
                behaviors.append(f"{token.head.text} {token.text}")
        
        return list(set(behaviors))
    
    def extract_background_setting(self, text: str) -> List[str]:
        """Extract background and setting information"""
        settings = []
        
        location_patterns = [
            r'\b(?:in|at|inside|outside)\s+(?:the\s+)?([a-zA-Z\s]+?)(?:\s+(?:room|area|building|house|apartment))',
            r'\b(?:bedroom|living room|bathroom|garage|basement|attic|office|hallway|convenience store)\b',
            r'\b(?:restaurant|store|park|street|alley|parking lot|mall|school)\b',
            r'\benvironment\s*:\s*([^\n]+)',
            r'\bsetting\s*:\s*([^\n]+)',
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            settings.extend([match.strip() for match in matches if match.strip()])
        
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC", "FAC"]:
                settings.append(ent.text)
        
        return list(set(settings))
    
    def extract_actions(self, text: str) -> List[str]:
        """Extract specific actions performed"""
        actions = []
        
        action_patterns = [
            r'\b(?:he|she|they|suspect|person)\s+(\w+ed)\s+',
            r'\b(?:was|were)\s+(\w+ing)\s+',
            r'\b(?:grabbed|took|opened|closed|entered|exited|searched|looked|moved|ran|walked)\b',
            r'\baction\s*:\s*([^\n]+)',
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            actions.extend([match.strip() for match in matches if match.strip()])
        
        doc = self.nlp(text)
        for token in doc:
            if token.pos_ == "VERB" and not token.is_stop:
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
        
        if "facial_expressions" in data and data["facial_expressions"]:
            expressions.extend(data["facial_expressions"].split(','))
        
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
        
        if "body_language" in data and data["body_language"]:
            body_lang.extend(data["body_language"].split(','))
        
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
        
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "WORK_OF_ART", "ORG"]:
                objects.append(ent.text)
        
        object_patterns = [
            r'\b(?:grabbed|took|touched|moved|broke|opened|closed)\s+(?:the\s+)?([a-zA-Z\s]+?)(?:\s|$|\.)',
            r'\b(?:weapon|tool|item|object|knife)\s*:\s*([^\n]+)',
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
            if self.emotion_classifier:
                emotion_result = self.emotion_classifier(text)
                if isinstance(emotion_result, list) and len(emotion_result) > 0:
                    emotions.append(emotion_result[0]['label'])
                elif isinstance(emotion_result, dict):
                    emotions.append(emotion_result['label'])
            
            if self.sentiment_analyzer:
                sentiment_result = self.sentiment_analyzer(text)
                if isinstance(sentiment_result, list) and len(sentiment_result) > 0:
                    emotions.append(sentiment_result[0]['label'])
                elif isinstance(sentiment_result, dict):
                    emotions.append(sentiment_result['label'])
                    
        except Exception as e:
            pass
        
        return emotions
    
    def extract_all_features(self, file_path: str) -> ForensicFeatures:
        """Extract all forensic features from the text file"""
        data = self.load_structured_text(file_path)
        text_content = str(data.get("content", ""))
        
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
        
        return features

class DeeVidPromptGenerator:
    """Generates DeeVid AI prompts using OpenAI"""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment variables or .env file")
        
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_known_scenario_prompt(self, features: ForensicFeatures) -> str:
        system_prompt = (
            "You are generating factual descriptions for law enforcement crime scene reconstruction. "
            "Write objective, professional descriptions based on witness testimony. "
            "State only observable facts: location, person, clothing, actions, timing. "
            "Use direct police report language. No camera angles, no dramatic elements, no speculation. "
            "This is for official forensic video reconstruction by law enforcement officers."
            "Prompt it to receive a prompt for a Dee Vid AI text to video generation"
            "State facts in a paragraph form without repeating information, just like a witness recount"
            "Generated prompt should not exceed 600 characters long"
            "You must be able to interpret actions from witness reports using contextual tools"
            "Include all observable facts about the crime: who did what, with what, where, when, and the victim's reactions. Be objective, explicit and precise without speculation or embellishment."
            "Mention any gory details"
        )
        
        user_content = (
            f"Create a factual crime scene description from witness testimony data:\n\n"
            f"Person Behavior: {', '.join(features.person_behavior)}. "
            f"Actions: {', '.join(features.actions)}. "
            f"Background/Setting: {', '.join(features.background_setting)}. "
            f"Clothing: {', '.join(features.clothing)}. "
            f"Facial Expressions: {', '.join(features.facial_expressions)}. "
            f"Body Language: {', '.join(features.body_language)}. "
            f"Timeline: {', '.join(features.timeline)}. "
            f"Objects Involved: {', '.join(features.objects_involved)}. "
            f"Emotions: {', '.join(features.emotions_detected)}."
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_completion_tokens=800,
                temperature=0.7
            )
            
            prompt = response.choices[0].message.content.strip()
            return prompt[:1900]
            
        except Exception as e:
            return f"Error generating known scenario prompt: {e}"
    
    def generate_alternative_scenario_prompt(self, features: ForensicFeatures) -> str:
        system_prompt = (
            "You are interpreting additional contextual facts for a law enforcement crime scene reconstruction. "
            "Write objective, professional descriptions considering alternative interpretations or context of witness testimony. "
            "Only state these facts: location, person, clothing, actions, timing, motive. "
            "Use direct police report language. No camera angles, no dramatic elements, no speculation. "
            "This is for official forensic video reconstruction by law enforcement officers."
            "Prompt it to receive a prompt for a Dee Vid AI text to video generation"
            "State facts in a paragraph form without repeating information, just like a witness recount"
            "Generated prompt should not exceed 600 characters long"
            "Using informatin from the witness.json file, come up with an alternate CRIMINAL reasoning for the CRIMINAL actions"
            "Alternate scenarios can include possible motives, that could have influenced the fixed actions"
            "You are to interpret the facts given and think of possible reasons why the crime was committed"
            "Be imaginative when you prompt for additional contextual information"
        ) 
        
        user_content = (
            f"Create an alternative crime scene description from witness testimony data.\n"
            f"Events must be different from that in the known scenario prompt.\n"
            f"Consider plausible alternative interpretations:\n\n"
            f"Person Behavior: {', '.join(features.person_behavior)}. "
            f"Actions: {', '.join(features.actions)}. "
            f"Background/Setting: {', '.join(features.background_setting)}. "
            f"Clothing: {', '.join(features.clothing)}. "
            f"Facial Expressions: {', '.join(features.facial_expressions)}. "
            f"Body Language: {', '.join(features.body_language)}. "
            f"Timeline: {', '.join(features.timeline)}. "
            f"Objects Involved: {', '.join(features.objects_involved)}. "
            f"Emotions: {', '.join(features.emotions_detected)}."
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_completion_tokens=800,
                temperature=0.8
            )
            
            prompt = response.choices[0].message.content.strip()
            return prompt[:1900]
            
        except Exception as e:
            return f"Error generating alternative scenario prompt: {e}"

class ForensicT2VPipeline:
    """Main pipeline class that orchestrates the entire process"""
    
    def __init__(self):
        """Initialize the complete pipeline"""
        print("Initializing forensic text-to-video pipeline...")
        self.extractor = ForensicTextExtractor()
        self.prompt_generator = DeeVidPromptGenerator()
        print("Pipeline initialization complete.")
    
    def process_forensic_text(self, file_path: str) -> Tuple[str, str]:
        """
        Process forensic text file and generate both DeeVid prompts
        
        Returns:
            Tuple[str, str]: (known_scenario_prompt, alternative_scenario_prompt)
        """
        print(f"Processing file: {file_path}")
        
        print("Extracting features...")
        features = self.extractor.extract_all_features(file_path)
        
        print("Generating video prompts...")
        known_prompt = self.prompt_generator.generate_known_scenario_prompt(features)
        alternative_prompt = self.prompt_generator.generate_alternative_scenario_prompt(features)
        
        print("Processing complete.")
        return known_prompt, alternative_prompt
    
    def save_prompts(self, known_prompt: str, alternative_prompt: str, output_dir: str = "output"):
        """Save generated prompts to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/known_scenario_prompt.txt", 'w', encoding='utf-8') as f:
            f.write(known_prompt)
        
        with open(f"{output_dir}/alternative_scenario_prompt.txt", 'w', encoding='utf-8') as f:
            f.write(alternative_prompt)
        
        print(f"Prompts saved to {output_dir}/")

# Example usage and testing
if __name__ == "__main__":
    try:
        pipeline = ForensicT2VPipeline()

        if not os.path.exists("witness.json"):
            print("Error: witness.json file not found!")
            exit(1)

        known_prompt, alternative_prompt = pipeline.process_forensic_text("witness.json")
        
        print("\nKNOWN SCENARIO PROMPT")
        print("=" * 60)
        print(known_prompt)
        print(f"Character count: {len(known_prompt)}")
        
        print("\nALTERNATIVE SCENARIO PROMPT")
        print("=" * 60)
        print(alternative_prompt)
        print(f"Character count: {len(alternative_prompt)}")
        
        pipeline.save_prompts(known_prompt, alternative_prompt)
        
    except Exception as e:
        print(f"Error: {e}")
      
