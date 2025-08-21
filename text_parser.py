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
    
    def __init__(self):
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
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                content_parts = []
                
                # Handle first JSON structure (confirmed_evidence)
                if "confirmed_evidence" in data:
                    if "case_summary" in data:
                        case_summary = data["case_summary"]
                        if "incident_summary" in case_summary:
                            content_parts.append(case_summary["incident_summary"])
                    
                    evidence = data["confirmed_evidence"]
                    evidence_text = self._convert_evidence_to_text(evidence)
                    content_parts.append(evidence_text)
                    
                    if "witness_reliability" in data:
                        for witness in data["witness_reliability"]:
                            witness_text = self._convert_witness_to_text(witness)
                            content_parts.append(witness_text)
                    
                    if "conflicting_accounts" in data:
                        for conflict in data["conflicting_accounts"]:
                            content_parts.append(f"Conflict: {conflict.get('conflicting_details', '')}")
                
                # Handle second JSON structure (crime_facts)
                elif "crime_facts" in data:
                    crime_facts = data["crime_facts"]
                    crime_text = self._convert_crime_facts_to_text(crime_facts)
                    content_parts.append(crime_text)
                    
                    if "raw_transcript" in data:
                        content_parts.append(data["raw_transcript"])
                
                combined_content = " ".join(content_parts)
                
                print(f"DEBUG - Combined content: {combined_content[:200]}...")
                
                return {
                    "content": combined_content,
                    "raw_data": data
                }
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    return {"content": content}
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return {}
    
       
    def _convert_witness_to_text(self, witness: Dict) -> str:
        text_parts = []
        
        if "key_observations" in witness:
            obs = witness["key_observations"]
            for key, value in obs.items():
                if value and value != "unknown":
                    text_parts.append(f"{key}: {value}")
        
        return ". ".join(text_parts)
    
    def _convert_evidence_to_text(self, evidence: Dict) -> str:
        text_parts = []
        
        if evidence.get("incident_type"):
            text_parts.append(f"Incident type: {evidence['incident_type']}")
        
        if "suspect_description" in evidence:
            suspect = evidence["suspect_description"]
            for key, value in suspect.items():
                if value and value != "unknown":
                    text_parts.append(f"Suspect {key.replace('_', ' ')}: {value}")
        
        if "incident_location" in evidence:
            location = evidence["incident_location"]
            for key, value in location.items():
                if value and value != "unknown":
                    text_parts.append(f"Location {key.replace('_', ' ')}: {value}")
        
        if "timeline" in evidence:
            timeline = evidence["timeline"]
            for key, value in timeline.items():
                if value and value != "unknown":
                    text_parts.append(f"Timeline {key}: {value}")
        
        if "victim_information" in evidence:
            victim = evidence["victim_information"]
            for key, value in victim.items():
                if value and value != "unknown":
                    text_parts.append(f"Victim {key}: {value}")
        
        if evidence.get("physical_evidence") and evidence["physical_evidence"] != "unknown":
            text_parts.append(f"Physical evidence: {evidence['physical_evidence']}")
        
        if evidence.get("sequence_of_events") and evidence["sequence_of_events"] != "unknown":
            text_parts.append(f"Sequence of events: {evidence['sequence_of_events']}")
        
        return ". ".join(text_parts)
    
    def _convert_crime_facts_to_text(self, crime_facts: Dict) -> str:
        text_parts = []
        
        if crime_facts.get("crime_type"):
            text_parts.append(f"Crime type: {crime_facts['crime_type']}")
        
        if "perpetrator" in crime_facts:
            perp = crime_facts["perpetrator"]
            for key, value in perp.items():
                if value and value != "unknown":
                    text_parts.append(f"Perpetrator {key}: {value}")
        
        if "victim" in crime_facts:
            victim = crime_facts["victim"]
            for key, value in victim.items():
                if value and value != "unknown":
                    text_parts.append(f"Victim {key}: {value}")
        
        if "crime_details" in crime_facts:
            details = crime_facts["crime_details"]
            if details.get("what_happened"):
                text_parts.append(f"What happened: {details['what_happened']}")
            if details.get("sequence_of_events"):
                text_parts.append(f"Sequence: {', '.join(details['sequence_of_events'])}")
        
        if "location" in crime_facts:
            location = crime_facts["location"]
            for key, value in location.items():
                if value and value != "unknown":
                    text_parts.append(f"Location {key}: {value}")
        
        if "time" in crime_facts:
            time_info = crime_facts["time"]
            for key, value in time_info.items():
                if value and value != "unknown":
                    text_parts.append(f"Time {key}: {value}")
        
        if "evidence" in crime_facts:
            evidence = crime_facts["evidence"]
            for key, value in evidence.items():
                if value and value != "unknown":
                    text_parts.append(f"Evidence {key}: {value}")
        
        return ". ".join(text_parts)
        
    def extract_person_behavior(self, text: str) -> List[str]:
        behaviors = []
        
        behavior_patterns = [
            r'\b(?:appeared|seemed|looked)\s+(\w+(?:\s+\w+)?)',
            r'\b(?:was|were)\s+(\w+ing)\b',
            r'\b(?:acting|behaving)\s+(\w+(?:\s+\w+)?)',
            r'\bmoved\s+(\w+(?:\s+\w+)?)',
            r'\b(?:walking|running|standing|sitting|lying)\s+(\w+(?:\s+\w+)?)',
            r'\bbehavior[:\s]+([^.]+)',
            r'\bfast movement\b',
            r'\bshouting\b',
            r'\bcollapsed\b'
        ]
        
        for pattern in behavior_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            behaviors.extend([match.strip() for match in matches if match.strip()])
        
        doc = self.nlp(text)
        for token in doc:
            if token.pos_ == "ADV" and token.head.pos_ in ["VERB", "ADJ"]:
                behaviors.append(f"{token.head.text} {token.text}")
        
        return list(set([b for b in behaviors if len(b) > 2]))
    
    def extract_background_setting(self, text: str) -> List[str]:
        settings = []
        
        location_patterns = [
            r'\b(?:in|at|inside|outside)\s+(?:the\s+)?([a-zA-Z\s]+?)(?:\s+(?:room|area|building|house|apartment|park))',
            r'\b(?:bedroom|living room|bathroom|garage|basement|attic|office|hallway|convenience store)\b',
            r'\b(?:restaurant|store|park|street|alley|parking lot|mall|school)\b',
            r'\blocation_type[:\s]+([^.]+)',
            r'\boutdoor\b',
            r'\bpark\b'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            settings.extend([match.strip() for match in matches if match.strip()])
        
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC", "FAC"]:
                settings.append(ent.text)
        
        return list(set([s for s in settings if len(s) > 2]))
    
    def extract_actions(self, text: str) -> List[str]:
        actions = []
        
        action_patterns = [
            r'\b(?:he|she|they|suspect|person|man)\s+(\w+ed)\s+',
            r'\b(?:was|were)\s+(\w+ing)\s+',
            r'\b(?:grabbed|took|opened|closed|entered|exited|searched|looked|moved|ran|walked|stabbed|attacked|collapsed|shouted)\b',
            r'\bsequence_of_events[:\s]+([^.]+)',
            r'\bstabbed\b',
            r'\bcollapsed\b',
            r'\bshouting\b',
            r'\bran away\b'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            actions.extend([match.strip() for match in matches if match.strip()])
        
        doc = self.nlp(text)
        for token in doc:
            if token.pos_ == "VERB" and not token.is_stop and len(token.text) > 2:
                verb_phrase = [token.text]
                for child in token.children:
                    if child.dep_ in ["dobj", "prep", "prt"]:
                        verb_phrase.append(child.text)
                actions.append(" ".join(verb_phrase))
        
        return list(set([a for a in actions if len(a) > 2]))
    
    def extract_lighting(self, text: str) -> List[str]:
        lighting = []
        
        lighting_patterns = [
            r'\b(?:bright|dark|dim|well-lit|poorly lit|shadowy|sunlit|moonlit)\b',
            r'\b(?:daylight|sunlight|artificial light|fluorescent|lamplight)\b',
            r'\blighting_conditions[:\s]+([^.]+)',
            r'\b(?:lights?\s+(?:on|off|bright|dim))\b',
            r'\b(?:it was|room was)\s+([a-zA-Z\s]+?)(?:\s+lit|lighted)'
        ]
        
        for pattern in lighting_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            lighting.extend([match.strip() for match in matches if match.strip()])
        
        return list(set([l for l in lighting if len(l) > 2]))
    
    def extract_clothing(self, text: str) -> List[str]:
        clothing = []
        
        clothing_patterns = [
            r'\b(?:wearing|dressed in|had on)\s+([^\n.]+)',
            r'\b(?:shirt|pants|jeans|dress|jacket|hoodie|coat|shoes|hat|cap)\b',
            r'\bclothing[:\s]+([^.]+)',
            r'\b(?:blue|red|black|white|green|yellow|brown|gray|grey)\s+(?:shirt|pants|jeans|dress|jacket|hoodie)',
            r'\b(?:dark|light|bright)\s+(?:clothing|clothes|attire)',
            r'\bdark clothing\b',
            r'\bblack hoodie\b',
            r'\bgrey hoodie\b'
        ]
        
        for pattern in clothing_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            clothing.extend([match.strip() for match in matches if match.strip()])
        
        return list(set([c for c in clothing if len(c) > 2]))
    
    def extract_facial_expressions(self, data: Dict) -> List[str]:
        expressions = []
        
        text = str(data.get("content", ""))
        expression_patterns = [
            r'\bfacial[_\s]expressions?\s*:\s*([^\n]+)',
            r'\b(?:looked|appeared|seemed)\s+(angry|sad|happy|surprised|confused|worried|scared|neutral)',
            r'\b(?:smiled|frowned|grimaced|squinted|stared|glared)\b',
            r'\bemotional_state[:\s]+([^.]+)'
        ]
        
        for pattern in expression_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            expressions.extend([match.strip() for match in matches if match.strip()])
        
        return list(set([expr.strip() for expr in expressions if expr.strip() and len(expr) > 2]))
    
    def extract_body_language(self, data: Dict) -> List[str]:
        body_lang = []
        
        text = str(data.get("content", ""))
        body_patterns = [
            r'\bbody[_\s]language\s*:\s*([^\n]+)',
            r'\b(?:posture|stance|gestures?)\s*:\s*([^\n]+)',
            r'\b(?:slouched|upright|tense|relaxed|fidgeting|still)\b',
            r'\b(?:crossed arms|hands on hips|pointed|gestured|shrugged)\b',
            r'\bfast movement\b'
        ]
        
        for pattern in body_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            body_lang.extend([match.strip() for match in matches if match.strip()])
        
        return list(set([bl.strip() for bl in body_lang if bl.strip() and len(bl) > 2]))
    
    def extract_timeline(self, text: str) -> List[str]:
        timeline = []
        
        time_patterns = [
            r'\b(?:at|around|approximately)\s+(\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?)',
            r'\b(?:first|then|next|after|finally|meanwhile)\s+([^.]+)',
            r'\b(?:morning|afternoon|evening|night|dawn|dusk)\b',
            r'\b(?:before|after|during)\s+([^,.]+)',
            r'\btime_of_incident[:\s]+([^.]+)',
            r'\b8:45 PM\b',
            r'\b9 PM\b',
            r'\b8:50 PM\b'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            timeline.extend([match.strip() for match in matches if match.strip()])
        
        return list(set([t for t in timeline if len(t) > 2]))
    
    def extract_objects_involved(self, text: str) -> List[str]:
        objects = []
        
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "WORK_OF_ART", "ORG"]:
                objects.append(ent.text)
        
        object_patterns = [
            r'\b(?:grabbed|took|touched|moved|broke|opened|closed)\s+(?:the\s+)?([a-zA-Z\s]+?)(?:\s|$|\.)',
            r'\b(?:weapon|tool|item|object|knife)\s*:\s*([^\n]+)',
            r'\b(?:knife|gun|phone|wallet|bag|keys|documents|jewelry|laptop|camera)\b',
            r'\bweapon_used[:\s]+([^.]+)',
            r'\bsomething shiny\b',
            r'\bdeadly weapon\b'
        ]
        
        for pattern in object_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            objects.extend([match.strip() for match in matches if match.strip()])
        
        return list(set([o for o in objects if len(o) > 2]))
    
    def extract_emotions(self, text: str) -> List[str]:
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
        
        emotion_patterns = [
            r'\bemotional_state[:\s]+([^.]+)',
            r'\bnegative\b',
            r'\bmedium reliability\b'
        ]
        
        for pattern in emotion_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            emotions.extend([match.strip() for match in matches if match.strip()])
        
        return list(set([e for e in emotions if len(e) > 2]))
    
    def extract_all_features(self, file_path: str) -> ForensicFeatures:
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
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment variables or .env file")
        
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_known_scenario_prompt(self, features: ForensicFeatures) -> str:
        system_prompt = (
            "You are generating factual descriptions for law enforcement crime scene reconstruction. "
            "Write objective, professional descriptions based ONLY on the provided witness testimony data. "
            "Do NOT invent or add any details not present in the input data. "
            "If information is missing or unknown, acknowledge it as such. "
            "State only observable facts from the witness data: location, person, clothing, actions, timing. "
            "Use direct police report language. No camera angles, no dramatic elements, no speculation. "
            "Generated prompt should not exceed 600 characters long. "
            "Focus on the actual incident details provided in the witness testimony."
        )
        
        user_content = (
            f"Create a factual crime scene description using ONLY the following witness testimony data:\n\n"
            f"Person Behavior: {', '.join(features.person_behavior) if features.person_behavior else 'Not specified'}. "
            f"Actions: {', '.join(features.actions) if features.actions else 'Not specified'}. "
            f"Background/Setting: {', '.join(features.background_setting) if features.background_setting else 'Not specified'}. "
            f"Clothing: {', '.join(features.clothing) if features.clothing else 'Not specified'}. "
            f"Facial Expressions: {', '.join(features.facial_expressions) if features.facial_expressions else 'Not specified'}. "
            f"Body Language: {', '.join(features.body_language) if features.body_language else 'Not specified'}. "
            f"Timeline: {', '.join(features.timeline) if features.timeline else 'Not specified'}. "
            f"Objects Involved: {', '.join(features.objects_involved) if features.objects_involved else 'Not specified'}. "
            f"Emotions: {', '.join(features.emotions_detected) if features.emotions_detected else 'Not specified'}.\n\n"
            f"Use only the information provided above. Do not add fictional details, times, locations, or names not present in the data."
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_completion_tokens=200,
                temperature=0.1
            )
            
            prompt = response.choices[0].message.content.strip()
            return prompt[:600]
            
        except Exception as e:
            return f"Error generating known scenario prompt: {e}"
    
    def generate_alternative_scenario_prompt(self, features: ForensicFeatures) -> str:
        system_prompt = (
            "You are interpreting witness testimony for law enforcement crime scene reconstruction. "
            "Write objective descriptions considering alternative interpretations of the SAME witness testimony data. "
            "Do NOT invent new details. Use the same facts but consider different possible interpretations or contexts. "
            "State only these facts with alternative context: location, person, clothing, actions, timing, possible motive. "
            "Use direct police report language. No camera angles, no dramatic elements. "
            "Generated prompt should not exceed 600 characters long. "
            "Focus on alternative interpretations of the provided witness data, not new fictional scenarios."
        ) 
        
        user_content = (
            f"Create an alternative interpretation of the SAME crime scene using the following witness testimony data:\n\n"
            f"Person Behavior: {', '.join(features.person_behavior) if features.person_behavior else 'Not specified'}. "
            f"Actions: {', '.join(features.actions) if features.actions else 'Not specified'}. "
            f"Background/Setting: {', '.join(features.background_setting) if features.background_setting else 'Not specified'}. "
            f"Clothing: {', '.join(features.clothing) if features.clothing else 'Not specified'}. "
            f"Facial Expressions: {', '.join(features.facial_expressions) if features.facial_expressions else 'Not specified'}. "
            f"Body Language: {', '.join(features.body_language) if features.body_language else 'Not specified'}. "
            f"Timeline: {', '.join(features.timeline) if features.timeline else 'Not specified'}. "
            f"Objects Involved: {', '.join(features.objects_involved) if features.objects_involved else 'Not specified'}. "
            f"Emotions: {', '.join(features.emotions_detected) if features.emotions_detected else 'Not specified'}.\n\n"
            f"Provide an alternative interpretation of these same facts. Do not add new details not present in the witness data."
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_completion_tokens=200,
                temperature=0.3
            )
            
            prompt = response.choices[0].message.content.strip()
            return prompt[:600]
            
        except Exception as e:
            return f"Error generating alternative scenario prompt: {e}"

class ForensicT2VPipeline:
    
    def __init__(self):
        print("Initializing forensic text-to-video pipeline...")
        self.extractor = ForensicTextExtractor()
        self.prompt_generator = DeeVidPromptGenerator()
        print("Pipeline initialization complete.")
    
    def process_forensic_text(self, file_path: str) -> Tuple[str, str]:
        print(f"Processing file: {file_path}")
        
        print("Extracting features...")
        features = self.extractor.extract_all_features(file_path)
        
        print("Features extracted:")
        print(f"Person Behavior: {features.person_behavior}")
        print(f"Actions: {features.actions}")
        print(f"Setting: {features.background_setting}")
        print(f"Clothing: {features.clothing}")
        print(f"Timeline: {features.timeline}")
        print(f"Objects: {features.objects_involved}")
        
        print("Generating video prompts...")
        known_prompt = self.prompt_generator.generate_known_scenario_prompt(features)
        alternative_prompt = self.prompt_generator.generate_alternative_scenario_prompt(features)
        
        print("Processing complete.")
        return known_prompt, alternative_prompt
    
    def save_prompts(self, known_prompt: str, alternative_prompt: str, output_dir: str = "output"):
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/known_scenario_prompt.txt", 'w', encoding='utf-8') as f:
            f.write(known_prompt)
        
        with open(f"{output_dir}/alternative_scenario_prompt.txt", 'w', encoding='utf-8') as f:
            f.write(alternative_prompt)
        
        print(f"Prompts saved to {output_dir}/")

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
